"""
Transport layer for inter-node communication.

Provides pluggable transport: 
- InProcessTransport: for embedded/testing (queues)
- TCPTransport: for real distributed deployment
"""

import json
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger("vecgrid.transport")


@dataclass
class Message:
    """Inter-node message."""
    msg_type: str       # "insert", "search", "search_result", "delete", "migrate", "heartbeat"
    sender: str         # Node ID
    payload: dict       # Message-specific data

    def serialize(self) -> bytes:
        data = {
            "msg_type": self.msg_type,
            "sender": self.sender,
            "payload": self._serialize_payload(self.payload),
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, raw: bytes) -> "Message":
        data = json.loads(raw.decode("utf-8"))
        payload = cls._deserialize_payload(data["payload"])
        return cls(
            msg_type=data["msg_type"],
            sender=data["sender"],
            payload=payload,
        )

    @staticmethod
    def _serialize_payload(payload: dict) -> dict:
        result = {}
        for k, v in payload.items():
            if isinstance(v, np.ndarray):
                result[k] = {"__ndarray__": True, "data": v.tolist()}
            elif isinstance(v, list) and v and isinstance(v[0], tuple):
                result[k] = [list(t) for t in v]
            else:
                result[k] = v
        return result

    @staticmethod
    def _deserialize_payload(payload: dict) -> dict:
        result = {}
        for k, v in payload.items():
            if isinstance(v, dict) and v.get("__ndarray__"):
                result[k] = np.array(v["data"], dtype=np.float32)
            else:
                result[k] = v
        return result


class Transport(ABC):
    """Abstract transport layer."""

    @abstractmethod
    def start(self, node_id: str, handler: Callable[[Message], Optional[Message]]):
        """Start listening for messages."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the transport."""
        pass

    @abstractmethod
    def send(self, target_node: str, message: Message) -> Optional[Message]:
        """Send a message to a target node and optionally get response."""
        pass

    @abstractmethod
    def broadcast(self, message: Message) -> list[Optional[Message]]:
        """Broadcast a message to all known nodes."""
        pass


class InProcessTransport(Transport):
    """
    In-process transport using shared memory.
    
    All nodes in the same process communicate via direct method calls.
    This is the embedded mode - like Hazelcast embedded.
    """

    # Class-level registry of all nodes
    _registry: dict[str, Callable] = {}
    _lock = threading.Lock()

    def start(self, node_id: str, handler: Callable[[Message], Optional[Message]]):
        with self._lock:
            InProcessTransport._registry[node_id] = handler
        self._node_id = node_id
        self._handler = handler

    def stop(self):
        with self._lock:
            InProcessTransport._registry.pop(self._node_id, None)

    def send(self, target_node: str, message: Message) -> Optional[Message]:
        with self._lock:
            handler = InProcessTransport._registry.get(target_node)
        if handler:
            return handler(message)
        return None

    def broadcast(self, message: Message) -> list[Optional[Message]]:
        results = []
        with self._lock:
            targets = {k: v for k, v in InProcessTransport._registry.items()
                       if k != self._node_id}
        for node_id, handler in targets.items():
            try:
                result = handler(message)
                results.append(result)
            except Exception:
                results.append(None)
        return results

    @classmethod
    def reset(cls):
        """Reset the global registry. Useful for tests."""
        with cls._lock:
            cls._registry.clear()


class TCPTransport(Transport):
    """
    TCP-based transport for real distributed deployment.
    
    Each node listens on a port and communicates via TCP.
    Uses a simple length-prefixed protocol.
    
    Also handles:
    - Discovery protocol (member list requests from seed discovery)
    - Heartbeat pings
    - Auto peer registration on discovery
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 0):
        self.host = host
        self.port = port
        self._node_id: Optional[str] = None
        self._handler: Optional[Callable] = None
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._peers: dict[str, tuple[str, int]] = {}  # node_id -> (host, port)
        self._lock = threading.Lock()

    def register_peer(self, node_id: str, host: str, port: int):
        """Register a peer node's address."""
        with self._lock:
            self._peers[node_id] = (host, port)

    def unregister_peer(self, node_id: str):
        """Remove a peer."""
        with self._lock:
            self._peers.pop(node_id, None)

    def get_peers(self) -> dict[str, tuple[str, int]]:
        """Get a copy of the current peer map."""
        with self._lock:
            return dict(self._peers)

    def start(self, node_id: str, handler: Callable[[Message], Optional[Message]]):
        self._node_id = node_id
        self._handler = handler
        
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self.port = self._server_socket.getsockname()[1]  # Get actual port if 0
        self._server_socket.listen(32)
        self._server_socket.settimeout(1.0)
        self._running = True
        
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        if self._thread:
            self._thread.join(timeout=3)

    def _listen_loop(self):
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_connection(self, conn: socket.socket):
        try:
            data = self._recv_message(conn)
            if not data:
                self._send_bytes(conn, b"")
                return

            # Try to detect discovery protocol vs normal messages
            raw = data.decode("utf-8")
            parsed = json.loads(raw)

            if parsed.get("type") == "discovery":
                # Discovery protocol: return member list
                self._handle_discovery_request(conn, parsed)
                return

            if parsed.get("type") == "ping":
                # Heartbeat ping: respond with pong
                pong = json.dumps({"type": "pong", "node_id": self._node_id}).encode("utf-8")
                self._send_bytes(conn, pong)
                return

            # Normal VecGrid message
            if self._handler:
                msg = Message.deserialize(data)
                response = self._handler(msg)
                if response:
                    self._send_bytes(conn, response.serialize())
                else:
                    self._send_bytes(conn, b"")
        except Exception as e:
            logger.debug(f"Error handling connection: {e}")
        finally:
            conn.close()

    def _handle_discovery_request(self, conn: socket.socket, request: dict):
        """Respond with current member list including ourselves."""
        members = [{"node_id": self._node_id, "host": self.host, "port": self.port}]
        with self._lock:
            for node_id, (host, port) in self._peers.items():
                members.append({"node_id": node_id, "host": host, "port": port})

        # Also register the requester as a peer
        req_node = request.get("node_id")
        req_host = request.get("host")
        req_port = request.get("port")
        if req_node and req_host and req_port and req_node != self._node_id:
            self.register_peer(req_node, req_host, req_port)

        response = json.dumps({"members": members}).encode("utf-8")
        self._send_bytes(conn, response)

    def send_ping(self, node_id: str, host: str, port: int) -> bool:
        """Send a heartbeat ping to a specific address. Returns True if pong received."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))

            ping = json.dumps({"type": "ping", "node_id": self._node_id}).encode("utf-8")
            self._send_bytes(sock, ping)

            resp_data = self._recv_message(sock)
            sock.close()

            if resp_data:
                resp = json.loads(resp_data.decode("utf-8"))
                return resp.get("type") == "pong"
            return False
        except Exception:
            return False

    def send(self, target_node: str, message: Message) -> Optional[Message]:
        with self._lock:
            addr = self._peers.get(target_node)
        if not addr:
            return None
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(addr)
            self._send_bytes(sock, message.serialize())
            
            response_data = self._recv_message(sock)
            sock.close()
            
            if response_data:
                return Message.deserialize(response_data)
            return None
        except socket.timeout:
            logger.debug(f"Timeout sending to {target_node} at {addr}")
            return None
        except ConnectionRefusedError:
            logger.debug(f"Connection refused to {target_node} at {addr}")
            return None
        except Exception as e:
            logger.debug(f"Send to {target_node} failed: {e}")
            return None

    def broadcast(self, message: Message) -> list[Optional[Message]]:
        results = []
        with self._lock:
            targets = dict(self._peers)
        for node_id in targets:
            if node_id != self._node_id:
                result = self.send(node_id, message)
                results.append(result)
        return results

    @staticmethod
    def _send_bytes(sock: socket.socket, data: bytes):
        length = struct.pack("!I", len(data))
        sock.sendall(length + data)

    @staticmethod
    def _recv_message(sock: socket.socket) -> Optional[bytes]:
        length_data = b""
        while len(length_data) < 4:
            chunk = sock.recv(4 - len(length_data))
            if not chunk:
                return None
            length_data += chunk
        
        length = struct.unpack("!I", length_data)[0]
        if length == 0:
            return None
        
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(length - len(data), 65536))
            if not chunk:
                return None
            data += chunk
        return data
