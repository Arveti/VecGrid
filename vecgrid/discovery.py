"""
Node Discovery for VecGrid.

Three discovery strategies (like Hazelcast):

1. Multicast UDP Discovery (default)
   - Nodes broadcast presence on a multicast group (224.2.2.3:54327)
   - Other nodes listening on the same group auto-discover peers
   - Works on local networks without any configuration

2. Seed Node Discovery
   - Connect to a list of known seed addresses
   - Ask each seed for the full member list
   - Works across networks / cloud environments

3. Heartbeat Failure Detector
   - Periodic heartbeat pings between nodes
   - If a node misses N heartbeats, it's declared dead
   - Triggers backup promotion automatically
"""

import json
import socket
import struct
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger("vecgrid.discovery")

# Defaults matching Hazelcast conventions
DEFAULT_MULTICAST_GROUP = "224.2.2.3"
DEFAULT_MULTICAST_PORT = 54327
DEFAULT_HEARTBEAT_INTERVAL = 2.0    # seconds
DEFAULT_HEARTBEAT_TIMEOUT = 8.0     # seconds (4 missed heartbeats = dead)


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    port: int
    last_heartbeat: float = 0.0

    def address(self) -> tuple[str, int]:
        return (self.host, self.port)

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "host": self.host, "port": self.port}

    @classmethod
    def from_dict(cls, d: dict) -> "NodeInfo":
        return cls(node_id=d["node_id"], host=d["host"], port=d["port"])


class MulticastDiscovery:
    """
    UDP multicast discovery — Hazelcast style.

    Each node periodically broadcasts its presence on a multicast group.
    Other nodes on the same network segment hear the broadcast and
    register the sender as a peer.

    Usage:
        discovery = MulticastDiscovery(
            node_id="node-1",
            service_port=5701,
            on_node_discovered=lambda info: print(f"Found {info.node_id}")
        )
        discovery.start()
    """

    def __init__(self, node_id: str, service_port: int,
                 on_node_discovered: Callable[[NodeInfo], None],
                 multicast_group: str = DEFAULT_MULTICAST_GROUP,
                 multicast_port: int = DEFAULT_MULTICAST_PORT,
                 broadcast_interval: float = 2.0,
                 advertise_host: Optional[str] = None):
        self.node_id = node_id
        self.service_port = service_port
        self.on_node_discovered = on_node_discovered
        self.multicast_group = multicast_group
        self.multicast_port = multicast_port
        self.broadcast_interval = broadcast_interval

        self._running = False
        self._send_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._send_sock: Optional[socket.socket] = None
        self._recv_sock: Optional[socket.socket] = None
        self._local_ip: str = advertise_host or self._get_local_ip()

    def _get_local_ip(self) -> str:
        """Get the local IP that can reach the network."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            s.close()

    def start(self):
        """Start broadcasting and listening."""
        self._running = True

        # Send socket
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        # Receive socket
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # Not available on all platforms

        self._recv_sock.bind(("", self.multicast_port))

        # Join multicast group
        mreq = struct.pack(
            "4sl",
            socket.inet_aton(self.multicast_group),
            socket.INADDR_ANY,
        )
        self._recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self._recv_sock.settimeout(1.0)

        self._send_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._recv_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()

        logger.info(f"Multicast discovery started: {self.multicast_group}:{self.multicast_port}")

    def stop(self):
        """Stop discovery."""
        self._running = False

        # Send a leave announcement
        try:
            msg = json.dumps({
                "type": "leave",
                "node_id": self.node_id,
            }).encode("utf-8")
            if self._send_sock:
                self._send_sock.sendto(msg, (self.multicast_group, self.multicast_port))
        except Exception:
            pass

        if self._send_thread:
            self._send_thread.join(timeout=3)
        if self._recv_thread:
            self._recv_thread.join(timeout=3)
        if self._send_sock:
            self._send_sock.close()
        if self._recv_sock:
            self._recv_sock.close()

    def _broadcast_loop(self):
        """Periodically broadcast our presence."""
        while self._running:
            try:
                msg = json.dumps({
                    "type": "alive",
                    "node_id": self.node_id,
                    "host": self._local_ip,
                    "port": self.service_port,
                }).encode("utf-8")
                self._send_sock.sendto(msg, (self.multicast_group, self.multicast_port))
            except Exception as e:
                logger.debug(f"Multicast send error: {e}")
            time.sleep(self.broadcast_interval)

    def _listen_loop(self):
        """Listen for multicast announcements from other nodes."""
        while self._running:
            try:
                data, addr = self._recv_sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))

                if msg.get("node_id") == self.node_id:
                    continue  # Ignore our own broadcasts

                if msg.get("type") == "alive":
                    info = NodeInfo(
                        node_id=msg["node_id"],
                        host=msg["host"],
                        port=msg["port"],
                        last_heartbeat=time.time(),
                    )
                    self.on_node_discovered(info)

                elif msg.get("type") == "leave":
                    # Node announced graceful departure — immediately notify
                    info = NodeInfo(
                        node_id=msg["node_id"],
                        host="",
                        port=0,
                    )
                    logger.info(f"Node {msg['node_id']} announced leave")
                    # Dispatch to the same callback; the VecGrid layer
                    # detects host="" as a departure signal
                    self.on_node_discovered(info)

            except socket.timeout:
                continue
            except Exception as e:
                logger.debug(f"Multicast recv error: {e}")


class SeedNodeDiscovery:
    """
    Seed node discovery — for environments where multicast doesn't work.

    Connects to a list of known seed addresses, asks each for the
    full cluster member list, and registers all discovered peers.

    Usage:
        discovery = SeedNodeDiscovery(
            node_id="node-1",
            service_port=5701,
            seeds=["192.168.1.10:5701", "192.168.1.11:5701"],
            on_node_discovered=lambda info: register(info)
        )
        discovery.start()
    """

    def __init__(self, node_id: str, service_port: int,
                 seeds: list[str],
                 on_node_discovered: Callable[[NodeInfo], None],
                 poll_interval: float = 5.0,
                 advertise_host: Optional[str] = None):
        self.node_id = node_id
        self.service_port = service_port
        self.seeds = self._parse_seeds(seeds)
        self.on_node_discovered = on_node_discovered
        self.poll_interval = poll_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._local_ip = advertise_host or self._get_local_ip()

    @staticmethod
    def _parse_seeds(seeds: list[str]) -> list[tuple[str, int]]:
        result = []
        for s in seeds:
            if ":" in s:
                host, port = s.rsplit(":", 1)
                result.append((host, int(port)))
            else:
                result.append((s, 5701))
        return result

    def _get_local_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            s.close()

    def start(self):
        self._running = True
        # Do an immediate discovery round
        self._discover_once()
        # Then keep polling in background
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"Seed node discovery started with {len(self.seeds)} seeds")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _poll_loop(self):
        while self._running:
            time.sleep(self.poll_interval)
            if self._running:
                self._discover_once()

    def _discover_once(self):
        """Connect to each seed and ask for member list."""
        for host, port in self.seeds:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3.0)
                sock.connect((host, port))

                # Send discovery request
                request = json.dumps({
                    "type": "discovery",
                    "node_id": self.node_id,
                    "host": self._local_ip,
                    "port": self.service_port,
                }).encode("utf-8")
                length = struct.pack("!I", len(request))
                sock.sendall(length + request)

                # Read response
                length_data = sock.recv(4)
                if len(length_data) == 4:
                    resp_len = struct.unpack("!I", length_data)[0]
                    if resp_len > 0:
                        resp_data = b""
                        while len(resp_data) < resp_len:
                            chunk = sock.recv(min(resp_len - len(resp_data), 65536))
                            if not chunk:
                                break
                            resp_data += chunk
                        if resp_data:
                            members = json.loads(resp_data.decode("utf-8"))
                            for m in members.get("members", []):
                                if m.get("node_id") != self.node_id:
                                    info = NodeInfo(
                                        node_id=m["node_id"],
                                        host=m["host"],
                                        port=m["port"],
                                        last_heartbeat=time.time(),
                                    )
                                    self.on_node_discovered(info)
                sock.close()
            except Exception as e:
                logger.debug(f"Seed {host}:{port} unreachable: {e}")


class HeartbeatFailureDetector:
    """
    Periodic heartbeat ping between cluster nodes.

    Each node sends heartbeat pings to all known peers. If a peer
    misses heartbeats beyond the timeout, it's declared dead and
    the on_node_failed callback fires.

    This is the crash detector — complements graceful leave messages.
    """

    def __init__(self, node_id: str,
                 get_peers: Callable[[], list[NodeInfo]],
                 send_ping: Callable[[str, str, int], bool],
                 on_node_failed: Callable[[str], None],
                 heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
                 heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT):
        self.node_id = node_id
        self.get_peers = get_peers
        self.send_ping = send_ping
        self.on_node_failed = on_node_failed
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout

        self._peer_heartbeats: dict[str, float] = {}  # node_id -> last_seen
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info(f"Heartbeat detector started (interval={self.heartbeat_interval}s, "
                     f"timeout={self.heartbeat_timeout}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def record_heartbeat(self, node_id: str):
        """Record that we heard from a node (any message counts)."""
        with self._lock:
            self._peer_heartbeats[node_id] = time.time()

    def remove_node(self, node_id: str):
        """Remove a node from tracking (after confirmed leave)."""
        with self._lock:
            self._peer_heartbeats.pop(node_id, None)

    def _heartbeat_loop(self):
        while self._running:
            try:
                self._check_and_ping()
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
            time.sleep(self.heartbeat_interval)

    def _check_and_ping(self):
        now = time.time()
        peers = self.get_peers()

        # Send ping to all peers
        for peer in peers:
            if peer.node_id == self.node_id:
                continue

            # Register peer if not tracked
            with self._lock:
                if peer.node_id not in self._peer_heartbeats:
                    self._peer_heartbeats[peer.node_id] = now

            # Send ping
            success = self.send_ping(peer.node_id, peer.host, peer.port)
            if success:
                self.record_heartbeat(peer.node_id)

        # Check for dead nodes
        dead_nodes = []
        with self._lock:
            for node_id, last_seen in list(self._peer_heartbeats.items()):
                if now - last_seen > self.heartbeat_timeout:
                    dead_nodes.append(node_id)
                    del self._peer_heartbeats[node_id]

        # Fire callbacks for dead nodes
        for node_id in dead_nodes:
            logger.warning(f"Node {node_id} failed (no heartbeat for "
                           f"{self.heartbeat_timeout}s)")
            self.on_node_failed(node_id)


@dataclass
class DiscoveryConfig:
    """Configuration for node discovery."""
    # Discovery mode: "multicast", "seed", or "none"
    mode: str = "multicast"

    # Multicast settings
    multicast_group: str = DEFAULT_MULTICAST_GROUP
    multicast_port: int = DEFAULT_MULTICAST_PORT
    broadcast_interval: float = 2.0

    # Seed node settings
    seeds: list[str] = field(default_factory=list)
    seed_poll_interval: float = 5.0

    # Heartbeat settings
    heartbeat_enabled: bool = True
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
