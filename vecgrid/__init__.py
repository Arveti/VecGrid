"""
VecGrid - Distributed Embedded Vector Database

A Hazelcast-style distributed in-memory vector database that embeds
directly in your application process. No separate infrastructure needed.

Usage (embedded / in-process):
    from vecgrid import VecGrid
    
    grid = VecGrid(node_id="app-1", dim=384)
    grid.start()
    grid.put("doc-1", vector, {"title": "Hello"})
    results = grid.search(query_vector, k=10)

Usage (TCP with multicast discovery):
    grid = VecGrid(node_id="app-1", dim=384, transport="tcp", port=5701,
                   discovery="multicast")
    grid.start()  # Auto-discovers other nodes on the network

Usage (TCP with seed node discovery):
    grid = VecGrid(node_id="app-1", dim=384, transport="tcp", port=5701,
                   discovery="seed", seeds=["192.168.1.10:5701", "192.168.1.11:5701"])
    grid.start()
"""

from typing import Optional

from .hnsw import HNSWIndex, HNSWConfig, NumpyHNSWIndex, create_index, get_backend_name, compile_filter
from .hash_ring import ConsistentHashRing
from .node import EmbeddedNode, VecGridConfig, SearchResult, PartitionRole, LocalPartition
from .transport import InProcessTransport, TCPTransport, Transport, Message
from .persistence import PersistenceEngine
from .discovery import (
    MulticastDiscovery, SeedNodeDiscovery, HeartbeatFailureDetector,
    DiscoveryConfig, NodeInfo,
)
from .admin import AdminAPI

__version__ = "0.1.2"


class VecGrid:
    """
    High-level API for the distributed embedded vector database.

    This is the main entry point. Each application instance creates
    one VecGrid instance, which automatically participates in the
    distributed cluster.
    """

    def __init__(self, node_id: str, dim: int = 128,
                 num_partitions: int = 271,
                 backup_count: int = 1,
                 transport: str = "embedded",
                 host: str = "0.0.0.0", port: int = 5701,
                 advertise_host: Optional[str] = None,
                 hnsw_config: Optional[dict] = None,
                 data_dir: Optional[str] = None,
                 snapshot_interval: int = 1000,
                 discovery: str = "none",
                 seeds: Optional[list[str]] = None,
                 multicast_group: str = "224.2.2.3",
                 multicast_port: int = 54327,
                 heartbeat: bool = True,
                 heartbeat_interval: float = 2.0,
                 heartbeat_timeout: float = 8.0,
                 **kwargs):
        """
        Create a VecGrid node.

        Args:
            node_id: Unique identifier for this node
            dim: Vector dimensions
            num_partitions: Total partitions (same across all nodes)
            transport: "embedded" for in-process, "tcp" for network
            host: Listen host for TCP transport
            port: Listen port for TCP transport (default 5701, like Hazelcast)
            advertise_host: Externally-reachable IP to advertise to peers.
                If None, auto-detects via default route. Set this when using
                VPNs (e.g. Tailscale) or when the bind address differs from
                the reachable address.
            hnsw_config: HNSW configuration overrides
            data_dir: Persistence directory (None = pure in-memory)
            snapshot_interval: WAL entries before auto-snapshot
            discovery: "none", "multicast", or "seed"
            seeds: Seed node addresses for seed discovery (e.g. ["10.0.0.1:5701"])
            multicast_group: Multicast group address
            multicast_port: Multicast UDP port
            heartbeat: Enable heartbeat failure detector
            heartbeat_interval: Heartbeat ping interval (seconds)
            heartbeat_timeout: Declare node dead after this many seconds
        """
        hnsw = HNSWConfig(**(hnsw_config or {}))

        config = VecGridConfig(
            dim=dim,
            num_partitions=num_partitions,
            backup_count=backup_count,
            hnsw=hnsw,
            data_dir=data_dir,
            snapshot_interval=snapshot_interval,
            **kwargs,
        )

        if transport == "tcp":
            self._transport = TCPTransport(host=host, port=port,
                                          advertise_host=advertise_host)
        else:
            self._transport = InProcessTransport()

        self._node = EmbeddedNode(
            node_id=node_id,
            config=config,
            transport=self._transport,
        )
        self.node_id = node_id
        self.dim = dim
        self._host = host
        self._port = port
        
        # Admin API
        self.admin = AdminAPI(self)

        # Discovery config
        self._discovery_mode = discovery if transport == "tcp" else "none"
        self._seeds = seeds or []
        self._multicast_group = multicast_group
        self._multicast_port = multicast_port
        self._advertise_host = (advertise_host if transport == "tcp"
                                else None)
        self._heartbeat_enabled = heartbeat and transport == "tcp"
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout

        # Discovery components (created on start)
        self._discovery = None
        self._failure_detector = None

    def start(self):
        """Start the node, run discovery, and join the cluster."""
        self._node.start()

        # Actual port (may differ from requested if port=0)
        if isinstance(self._transport, TCPTransport):
            self._port = self._transport.port

        # Start discovery
        if self._discovery_mode == "multicast":
            self._discovery = MulticastDiscovery(
                node_id=self.node_id,
                service_port=self._port,
                on_node_discovered=self._on_peer_discovered,
                multicast_group=self._multicast_group,
                multicast_port=self._multicast_port,
                advertise_host=self._advertise_host,
            )
            self._discovery.start()

        elif self._discovery_mode == "seed":
            self._discovery = SeedNodeDiscovery(
                node_id=self.node_id,
                service_port=self._port,
                seeds=self._seeds,
                on_node_discovered=self._on_peer_discovered,
                advertise_host=self._advertise_host,
            )
            self._discovery.start()

        # Start heartbeat failure detector
        if self._heartbeat_enabled and isinstance(self._transport, TCPTransport):
            self._failure_detector = HeartbeatFailureDetector(
                node_id=self.node_id,
                get_peers=self._get_peer_infos,
                send_ping=self._transport.send_ping,
                on_node_failed=self._on_peer_failed,
                heartbeat_interval=self._heartbeat_interval,
                heartbeat_timeout=self._heartbeat_timeout,
            )
            self._failure_detector.start()

        return self

    def stop(self):
        """Stop the node and leave the cluster."""
        if self._failure_detector:
            self._failure_detector.stop()
        if self._discovery:
            self._discovery.stop()
        self._node.stop()

    def _on_peer_discovered(self, info: NodeInfo):
        """Called by discovery when a new peer is found."""
        if info.node_id == self.node_id:
            return

        # Leave signal (host="") — handle as node departure
        if not info.host:
            self._on_peer_failed(info.node_id)
            return

        if not isinstance(self._transport, TCPTransport):
            return

        # Register transport peer
        self._transport.register_peer(info.node_id, info.host, info.port)

        # Join the peer into the cluster (triggers partition rebalancing)
        if info.node_id not in self._node._cluster_nodes:
            join_msg = Message(
                msg_type="join",
                sender=self.node_id,
                payload={"node_id": self.node_id},
            )
            resp = self._transport.send(info.node_id, join_msg)
            if resp and resp.msg_type == "join_ack":
                self._node._handle_join(Message(
                    msg_type="join",
                    sender=info.node_id,
                    payload={"node_id": info.node_id},
                ))

        # Record heartbeat
        if self._failure_detector:
            self._failure_detector.record_heartbeat(info.node_id)

    def _on_peer_failed(self, node_id: str):
        """Called by heartbeat detector when a node is declared dead."""
        if isinstance(self._transport, TCPTransport):
            self._transport.unregister_peer(node_id)
        # Trigger the same leave handling as graceful shutdown
        self._node._handle_leave(Message(
            msg_type="leave",
            sender=node_id,
            payload={"node_id": node_id},
        ))

    def _get_peer_infos(self) -> list[NodeInfo]:
        """Get peer list for heartbeat detector."""
        if not isinstance(self._transport, TCPTransport):
            return []
        peers = self._transport.get_peers()
        return [
            NodeInfo(node_id=nid, host=host, port=port)
            for nid, (host, port) in peers.items()
        ]

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    def put(self, vector_id: str, vector, metadata: Optional[dict] = None):
        """Insert or update a vector."""
        import numpy as np
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        self._node.put(vector_id, vector, metadata)

    def search(self, query, k: int = 10, ef: Optional[int] = None,
               filter: Optional = None) -> list[SearchResult]:
        """Search for k nearest neighbors across the cluster.

        Args:
            query: Query vector (numpy array or list).
            k: Number of results to return.
            ef: HNSW search beam width override.
            filter: Metadata filter. Either:
                - Dict spec: {"field": "source", "op": "eq", "value": "Biology"}
                - List of specs (AND): [spec1, spec2, ...]
                - Callable(meta) -> bool (single-node only)
        """
        import numpy as np
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        return self._node.search(query, k=k, ef=ef, filter=filter)

    def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        return self._node.delete(vector_id)

    def get(self, vector_id: str):
        """Get a vector by ID (returns (vector, metadata) or None)."""
        return self._node.get(vector_id)

    def local_size(self) -> int:
        """Number of primary vectors stored on this node."""
        return self._node.local_size()

    def local_backup_size(self) -> int:
        """Number of backup vectors stored on this node."""
        return self._node.local_backup_size()

    def cluster_size(self) -> int:
        """Total primary vectors across the entire cluster."""
        return self._node.cluster_size()

    def stats(self) -> dict:
        """Node and cluster statistics."""
        return self._node.stats()

    def add_peer(self, node_id: str, host: str = None, port: int = None):
        """Manually register a TCP peer (not needed with discovery)."""
        if isinstance(self._transport, TCPTransport) and host and port:
            self._transport.register_peer(node_id, host, port)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __repr__(self):
        return repr(self._node)
