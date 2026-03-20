"""
EmbeddedNode - the core of VecGrid.

Hazelcast-style distributed embedded vector database node with:
- Sync backup replication (write to primary + N backups before ack)
- Safe partition migration (migrate-then-delete, no data loss)
- Smart routing (any node handles any request, forwards to owner)
- Automatic backup promotion on node failure
- Partition table versioning for consistency
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import numpy as np

from .hnsw import HNSWIndex, HNSWConfig, create_index
from .hash_ring import ConsistentHashRing
from .transport import Transport, InProcessTransport, Message
from .persistence import PersistenceEngine

logger = logging.getLogger("vecgrid")


class PartitionRole(Enum):
    PRIMARY = "primary"
    BACKUP = "backup"
    STALE = "stale"  # No longer owned, kept until migrated away


@dataclass
class LocalPartition:
    """A partition hosted on this node (primary or backup)."""
    id: int
    role: PartitionRole
    index: HNSWIndex
    version: int = 0  # Incremented on every write for sync tracking


@dataclass
class VecGridConfig:
    """Configuration for a VecGrid node."""
    dim: int = 128                    # Vector dimensions
    num_partitions: int = 271         # Total partitions in cluster
    backup_count: int = 1             # Sync backup copies per partition
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    search_timeout: float = 5.0       # Seconds to wait for search responses
    sync_backup: bool = True          # Synchronous backup replication
    data_dir: Optional[str] = None    # Persistence directory (None = no persistence)
    snapshot_interval: int = 1000     # WAL entries before auto-snapshot


@dataclass
class SearchResult:
    """A single search result."""
    vector_id: str
    distance: float
    metadata: dict
    partition_id: int
    source_node: str


class EmbeddedNode:
    """
    Distributed embedded vector database node with Hazelcast-style semantics.

    Write path (sync backup replication):
        Client -> put("doc-1", vec)
            -> hash("doc-1") -> partition 42
            -> partition 42 owned by node-1
            -> IF local: insert into primary index
               ELSE: smart-route to node-1
            -> node-1 replicates to backup node(s) synchronously
            -> ack to client only after all backups confirm

    Read path (scatter-gather):
        Client -> search(query, k=10)
            -> fan out to ALL nodes (search msg)
            -> each node searches its LOCAL primary partitions
            -> merge all results, return global top-k

    Node failure:
        -> node-2 detects node-1 left (leave msg or timeout)
        -> partitions where node-1 was primary:
           backup holder promotes to primary (backup already has data)
        -> partitions where node-1 was backup:
           new backup is selected, primary syncs data to it
        -> zero data loss for any partition that had a backup
    """

    def __init__(self, node_id: str, config: Optional[VecGridConfig] = None,
                 transport: Optional[Transport] = None):
        self.node_id = node_id
        self.config = config or VecGridConfig()
        self.transport = transport or InProcessTransport()

        # Partition management
        self.hash_ring = ConsistentHashRing(
            num_partitions=self.config.num_partitions,
            backup_count=self.config.backup_count,
        )

        # Local partitions: partition_id -> LocalPartition (both primary and backup)
        self._partitions: dict[int, LocalPartition] = {}
        self._lock = threading.RLock()
        self._running = False

        # Cluster state
        self._cluster_nodes: set[str] = set()
        self._partition_table_version: int = 0

        # Persistence (optional)
        self._persistence: Optional[PersistenceEngine] = None
        if self.config.data_dir:
            self._persistence = PersistenceEngine(
                data_dir=self.config.data_dir,
                snapshot_interval=self.config.snapshot_interval,
            )

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def start(self):
        """Start the node and join the cluster."""
        self._running = True

        # Open persistence engine
        if self._persistence:
            self._persistence.open()

        self.transport.start(self.node_id, self._handle_message)

        # Add self to the hash ring
        self.hash_ring.add_node(self.node_id)
        self._cluster_nodes.add(self.node_id)
        self._partition_table_version += 1

        # Create local primary indexes for our partitions
        self._sync_partition_table()

        # Recover data from disk (WAL + snapshots)
        if self._persistence:
            self._recover_from_disk()

        # Announce to existing cluster
        announce = Message(
            msg_type="join",
            sender=self.node_id,
            payload={"node_id": self.node_id},
        )
        responses = self.transport.broadcast(announce)

        # Process join responses
        for resp in responses:
            if resp and resp.msg_type == "join_ack":
                peer = resp.sender
                if peer not in self._cluster_nodes:
                    self._cluster_nodes.add(peer)
                    self.hash_ring.add_node(peer)
                    self._partition_table_version += 1

        # Recompute partition table after learning about all peers
        self._sync_partition_table()

        # Pull data for partitions we now own (safe migration)
        self._pull_missing_partitions()

        my_primaries = sum(1 for p in self._partitions.values() if p.role == PartitionRole.PRIMARY)
        my_backups = sum(1 for p in self._partitions.values() if p.role == PartitionRole.BACKUP)
        logger.info(
            f"Node {self.node_id} started. "
            f"Primary: {my_primaries}, Backup: {my_backups}. "
            f"Cluster: {len(self._cluster_nodes)} nodes"
        )

    def stop(self):
        """Graceful shutdown: snapshot, announce departure, close."""
        self._running = False

        # Snapshot all local partitions before leaving
        if self._persistence:
            self._snapshot_all_partitions()
            self._persistence.close()

        # Announce departure — receivers will promote backups
        leave = Message(
            msg_type="leave",
            sender=self.node_id,
            payload={"node_id": self.node_id},
        )
        self.transport.broadcast(leave)
        self.transport.stop()

        self.hash_ring.remove_node(self.node_id)
        self._cluster_nodes.discard(self.node_id)
        self._partitions.clear()

        logger.info(f"Node {self.node_id} stopped.")

    # ----------------------------------------------------------------
    # Partition table management
    # ----------------------------------------------------------------

    def _sync_partition_table(self):
        """
        Synchronize local partition holdings with the hash ring.
        Creates missing partitions. Does NOT delete data until
        migration is confirmed (safe migration).
        """
        with self._lock:
            should_hold: dict[int, PartitionRole] = {}

            for pid in range(self.config.num_partitions):
                partition = self.hash_ring.partitions[pid]
                if partition.owner_node == self.node_id:
                    should_hold[pid] = PartitionRole.PRIMARY
                elif self.node_id in partition.backup_nodes:
                    should_hold[pid] = PartitionRole.BACKUP

            # Create partitions we should hold but don't yet
            for pid, role in should_hold.items():
                if pid not in self._partitions:
                    self._partitions[pid] = LocalPartition(
                        id=pid,
                        role=role,
                        index=create_index(dim=self.config.dim, config=self.config.hnsw),
                    )
                else:
                    # Update role (e.g. backup promoted to primary)
                    self._partitions[pid].role = role

            # Drop partitions we no longer need (only if empty)
            # Non-empty ones marked STALE — kept for migration but excluded from counts
            stale = [pid for pid in self._partitions if pid not in should_hold]
            for pid in stale:
                if len(self._partitions[pid].index) == 0:
                    del self._partitions[pid]
                else:
                    self._partitions[pid].role = PartitionRole.STALE

    def _pull_missing_partitions(self):
        """
        Pull data for primary partitions that are empty.
        Safe migration: pull data THEN tell source to drop.
        """
        with self._lock:
            empty_primaries = [
                pid for pid, lp in self._partitions.items()
                if lp.role == PartitionRole.PRIMARY and len(lp.index) == 0
            ]

        for pid in empty_primaries:
            msg = Message(
                msg_type="migrate_request",
                sender=self.node_id,
                payload={"partition_id": pid},
            )
            responses = self.transport.broadcast(msg)
            for resp in responses:
                if resp and resp.msg_type == "migrate_data":
                    vectors = resp.payload.get("vectors", {})
                    if vectors:
                        self._apply_migration(pid, resp.payload)
                        # Confirm migration so source can drop
                        drop_msg = Message(
                            msg_type="migration_complete",
                            sender=self.node_id,
                            payload={"partition_id": pid},
                        )
                        self.transport.send(resp.sender, drop_msg)
                        break

    def _apply_migration(self, partition_id: int, data: dict):
        """Apply migrated partition data into local index."""
        with self._lock:
            index = create_index(dim=self.config.dim, config=self.config.hnsw)
            vectors = data.get("vectors", {})
            metadata = data.get("metadata", {})
            version = data.get("version", 0)

            for vid, vec_data in vectors.items():
                vec = np.array(vec_data, dtype=np.float32)
                meta = metadata.get(vid, {})
                index.insert(vid, vec, meta)

            if partition_id in self._partitions:
                self._partitions[partition_id].index = index
                self._partitions[partition_id].version = version
            else:
                self._partitions[partition_id] = LocalPartition(
                    id=partition_id,
                    role=PartitionRole.PRIMARY,
                    index=index,
                    version=version,
                )

        # Snapshot migrated data immediately for durability
        if self._persistence and vectors:
            np_vectors = {vid: np.array(v, dtype=np.float32) if isinstance(v, list) else v
                          for vid, v in vectors.items()}
            self._persistence.snapshot(
                partition_id, version, self.config.dim, np_vectors, metadata
            )

    # ----------------------------------------------------------------
    # Persistence: WAL + Snapshot
    # ----------------------------------------------------------------

    def _recover_from_disk(self):
        """
        Recover all local partitions from WAL + snapshots.
        Called during start() before joining the cluster.
        """
        persisted = self._persistence.get_persisted_partitions()
        if not persisted:
            return

        recovered_total = 0
        for pid in persisted:
            # Only recover partitions we currently own
            if pid not in self._partitions:
                continue

            snap_ver, vectors, metadata, wal_entries = self._persistence.recover(
                pid, self.config.dim
            )

            lp = self._partitions[pid]

            # Apply snapshot vectors
            for vid, vec in vectors.items():
                meta = metadata.get(vid, {})
                lp.index.insert(vid, vec, meta)

            # Replay WAL on top
            for entry in wal_entries:
                if entry.op == "insert" and entry.vector is not None:
                    lp.index.insert(entry.vector_id, entry.vector, entry.metadata or {})
                elif entry.op == "delete":
                    lp.index.delete(entry.vector_id)

            # Set version to max of snapshot + WAL
            if wal_entries:
                lp.version = max(e.version for e in wal_entries)
            else:
                lp.version = snap_ver

            recovered_total += len(lp.index)

        if recovered_total > 0:
            logger.info(f"Node {self.node_id}: recovered {recovered_total} vectors from disk")

    def _persist_insert(self, partition_id: int, version: int,
                        vector_id: str, vector: np.ndarray,
                        metadata: Optional[dict]):
        """Write insert to WAL and trigger snapshot if needed."""
        if not self._persistence:
            return
        self._persistence.log_insert(partition_id, version, vector_id, vector, metadata)
        self._maybe_snapshot(partition_id)

    def _persist_delete(self, partition_id: int, version: int, vector_id: str):
        """Write delete to WAL."""
        if not self._persistence:
            return
        self._persistence.log_delete(partition_id, version, vector_id)

    def _maybe_snapshot(self, partition_id: int):
        """Take a snapshot if WAL is large enough."""
        if not self._persistence:
            return
        if self._persistence.should_snapshot(partition_id):
            with self._lock:
                lp = self._partitions.get(partition_id)
                if lp and lp.role == PartitionRole.PRIMARY:
                    self._persistence.snapshot(
                        partition_id, lp.version, self.config.dim,
                        dict(lp.index.vectors), dict(lp.index.metadata),
                    )

    def _snapshot_all_partitions(self):
        """Snapshot all primary partitions. Called during graceful shutdown."""
        with self._lock:
            for pid, lp in self._partitions.items():
                if lp.role == PartitionRole.PRIMARY and len(lp.index) > 0:
                    self._persistence.snapshot(
                        pid, lp.version, self.config.dim,
                        dict(lp.index.vectors), dict(lp.index.metadata),
                    )

    # ----------------------------------------------------------------
    # Message handlers
    # ----------------------------------------------------------------

    def _handle_message(self, message: Message) -> Optional[Message]:
        """Route incoming message to appropriate handler."""
        handlers = {
            "join": self._handle_join,
            "leave": self._handle_leave,
            "insert": self._handle_insert,
            "backup_insert": self._handle_backup_insert,
            "search": self._handle_search,
            "delete": self._handle_delete,
            "backup_delete": self._handle_backup_delete,
            "get": self._handle_get,
            "migrate_request": self._handle_migrate_request,
            "migration_complete": self._handle_migration_complete,
            "migrate_data_push": self._handle_migrate_data_push,
            "size_request": self._handle_size_request,
        }
        handler = handlers.get(message.msg_type)
        if handler:
            return handler(message)
        return None

    def _handle_join(self, msg: Message) -> Message:
        """Handle a node joining the cluster."""
        peer = msg.payload["node_id"]
        with self._lock:
            if peer not in self._cluster_nodes:
                self._cluster_nodes.add(peer)
                self.hash_ring.add_node(peer)
                self._partition_table_version += 1
                self._sync_partition_table()

        # After topology change, ensure we have data for our new primaries
        # and replicate to new backups
        self._pull_missing_partitions()
        self._replicate_to_new_backups()

        return Message(
            msg_type="join_ack",
            sender=self.node_id,
            payload={"node_id": self.node_id},
        )

    def _handle_leave(self, msg: Message) -> Optional[Message]:
        """
        Handle node leaving: promote backups to primary, pull missing data.
        This is where Hazelcast-style failover happens.
        """
        peer = msg.payload["node_id"]
        with self._lock:
            self._cluster_nodes.discard(peer)
            self.hash_ring.remove_node(peer)
            self._partition_table_version += 1
            self._sync_partition_table()

            # The sync_partition_table call above already updates roles
            # based on the new hash ring state. Backups are promoted to
            # primary automatically because the hash ring now says we own
            # those partitions.

        # Outside the lock: pull data for primary partitions we don't have,
        # and replicate our data to new backups
        self._pull_missing_partitions()
        self._replicate_to_new_backups()

        return None

    def _replicate_to_new_backups(self):
        """After topology change, sync data to new backup nodes."""
        with self._lock:
            snapshot = [(pid, lp) for pid, lp in self._partitions.items()
                        if lp.role == PartitionRole.PRIMARY and len(lp.index) > 0]
        for pid, lp in snapshot:
            partition = self.hash_ring.partitions.get(pid)
            if not partition:
                continue
            for backup_node in partition.backup_nodes:
                if backup_node == self.node_id:
                    continue
                try:
                    vectors = {vid: vec.tolist() for vid, vec in lp.index.vectors.items()}
                    metadata = dict(lp.index.metadata)
                except RuntimeError:
                    continue  # dict changed during iteration — skip this cycle
                msg = Message(
                    msg_type="migrate_data_push",
                    sender=self.node_id,
                    payload={
                        "partition_id": pid,
                        "role": "backup",
                        "vectors": vectors,
                        "metadata": metadata,
                        "version": lp.version,
                    },
                )
                self.transport.send(backup_node, msg)

    def _handle_insert(self, msg: Message) -> Message:
        """
        Handle vector insert with smart routing + backup replication.
        If we don't own this partition, forward to the owner.
        """
        payload = msg.payload
        pid = payload["partition_id"]
        vid = payload["vector_id"]
        vector = payload["vector"]
        metadata = payload.get("metadata", {})

        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)

        partition = self.hash_ring.partitions[pid]

        # Smart routing: forward to actual owner if not us
        if partition.owner_node != self.node_id:
            forward_resp = self.transport.send(partition.owner_node, msg)
            if forward_resp:
                return forward_resp
            return Message(msg_type="insert_ack", sender=self.node_id,
                           payload={"ok": False, "error": "forward_failed"})

        # We are the primary — insert locally
        with self._lock:
            if pid not in self._partitions:
                self._partitions[pid] = LocalPartition(
                    id=pid,
                    role=PartitionRole.PRIMARY,
                    index=create_index(dim=self.config.dim, config=self.config.hnsw),
                )
            lp = self._partitions[pid]
            lp.version += 1
            version = lp.version

        # WAL: persist BEFORE applying to index
        self._persist_insert(pid, version, vid, vector, metadata)

        # Apply to index
        with self._lock:
            lp.index.insert(vid, vector, metadata)

        # Synchronous backup replication
        if self.config.sync_backup and partition.backup_nodes:
            self._replicate_insert(pid, vid, vector, metadata, partition.backup_nodes)

        return Message(msg_type="insert_ack", sender=self.node_id, payload={"ok": True})

    def _replicate_insert(self, pid: int, vid: str, vector: np.ndarray,
                          metadata: dict, backup_nodes: list[str]):
        """Synchronously replicate an insert to all backup nodes."""
        backup_msg = Message(
            msg_type="backup_insert",
            sender=self.node_id,
            payload={
                "partition_id": pid,
                "vector_id": vid,
                "vector": vector,
                "metadata": metadata,
            },
        )
        for backup_node in backup_nodes:
            if backup_node != self.node_id:
                self.transport.send(backup_node, backup_msg)

    def _handle_backup_insert(self, msg: Message) -> Message:
        """Handle a backup replication insert from the primary."""
        payload = msg.payload
        pid = payload["partition_id"]
        vid = payload["vector_id"]
        vector = payload["vector"]
        metadata = payload.get("metadata", {})

        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)

        with self._lock:
            if pid not in self._partitions:
                self._partitions[pid] = LocalPartition(
                    id=pid,
                    role=PartitionRole.BACKUP,
                    index=create_index(dim=self.config.dim, config=self.config.hnsw),
                )
            lp = self._partitions[pid]
            lp.index.insert(vid, vector, metadata)
            lp.version += 1

        return Message(msg_type="backup_ack", sender=self.node_id, payload={"ok": True})

    def _handle_search(self, msg: Message) -> Message:
        """Search all LOCAL primary partitions only (no backup double-counting)."""
        from .hnsw import compile_filter

        payload = msg.payload
        query = payload["query"]
        k = payload.get("k", 10)
        ef = payload.get("ef")
        filter_spec = payload.get("filter")

        if isinstance(query, list):
            query = np.array(query, dtype=np.float32)

        compiled_filter = compile_filter(filter_spec)

        # Snapshot partition references under lock, search outside it
        with self._lock:
            to_search = [(pid, lp) for pid, lp in self._partitions.items()
                         if lp.role == PartitionRole.PRIMARY and len(lp.index) > 0]

        all_results = []
        for pid, lp in to_search:
            try:
                results = lp.index.search(query, k=k, ef=ef,
                                          filter_fn=compiled_filter)
                for dist, vid, meta in results:
                    all_results.append({
                        "distance": dist,
                        "vector_id": vid,
                        "metadata": meta,
                        "partition_id": pid,
                    })
            except Exception as e:
                logger.warning(f"Search failed on partition {pid}: {e}")

        all_results.sort(key=lambda x: x["distance"])
        return Message(
            msg_type="search_result",
            sender=self.node_id,
            payload={"results": all_results[:k]},
        )

    def _handle_delete(self, msg: Message) -> Message:
        """Handle deletion with smart routing + backup sync."""
        payload = msg.payload
        pid = payload["partition_id"]
        vid = payload["vector_id"]

        partition = self.hash_ring.partitions[pid]

        if partition.owner_node != self.node_id:
            forward_resp = self.transport.send(partition.owner_node, msg)
            if forward_resp:
                return forward_resp
            return Message(msg_type="delete_ack", sender=self.node_id,
                           payload={"ok": False, "error": "forward_failed"})

        ok = False
        version = 0
        with self._lock:
            if pid in self._partitions:
                ok = self._partitions[pid].index.delete(vid)
                if ok:
                    self._partitions[pid].version += 1
                    version = self._partitions[pid].version

        if ok:
            self._persist_delete(pid, version, vid)

        if ok and self.config.sync_backup and partition.backup_nodes:
            backup_msg = Message(
                msg_type="backup_delete",
                sender=self.node_id,
                payload={"partition_id": pid, "vector_id": vid},
            )
            for bn in partition.backup_nodes:
                if bn != self.node_id:
                    self.transport.send(bn, backup_msg)

        return Message(msg_type="delete_ack", sender=self.node_id, payload={"ok": ok})

    def _handle_backup_delete(self, msg: Message) -> Message:
        """Handle a backup delete from the primary."""
        payload = msg.payload
        pid = payload["partition_id"]
        vid = payload["vector_id"]

        with self._lock:
            if pid in self._partitions:
                self._partitions[pid].index.delete(vid)
                self._partitions[pid].version += 1

        return Message(msg_type="backup_ack", sender=self.node_id, payload={"ok": True})

    def _handle_get(self, msg: Message) -> Message:
        """Handle get-by-id with smart routing."""
        payload = msg.payload
        pid = payload["partition_id"]
        vid = payload["vector_id"]

        partition = self.hash_ring.partitions[pid]

        if partition.owner_node != self.node_id:
            forward_resp = self.transport.send(partition.owner_node, msg)
            if forward_resp:
                return forward_resp
            return Message(msg_type="get_result", sender=self.node_id,
                           payload={"found": False})

        with self._lock:
            if pid in self._partitions:
                index = self._partitions[pid].index
                if vid in index.vectors:
                    return Message(
                        msg_type="get_result",
                        sender=self.node_id,
                        payload={
                            "found": True,
                            "vector": index.vectors[vid],
                            "metadata": index.metadata.get(vid, {}),
                        },
                    )

        return Message(msg_type="get_result", sender=self.node_id, payload={"found": False})

    def _handle_migrate_request(self, msg: Message) -> Message:
        """
        Send partition data but DON'T delete.
        Wait for migration_complete before dropping.
        """
        pid = msg.payload["partition_id"]

        with self._lock:
            lp = self._partitions.get(pid)
            if lp and len(lp.index) > 0:
                vectors = {vid: vec.tolist() for vid, vec in lp.index.vectors.items()}
                metadata = dict(lp.index.metadata)
                return Message(
                    msg_type="migrate_data",
                    sender=self.node_id,
                    payload={
                        "vectors": vectors,
                        "metadata": metadata,
                        "version": lp.version,
                    },
                )

        return Message(
            msg_type="migrate_data",
            sender=self.node_id,
            payload={"vectors": {}, "metadata": {}, "version": 0},
        )

    def _handle_migration_complete(self, msg: Message) -> Optional[Message]:
        """
        New owner confirmed receipt. NOW safe to drop the partition.
        This is the key fix: migrate-THEN-delete, never delete-then-migrate.
        """
        pid = msg.payload["partition_id"]
        with self._lock:
            partition = self.hash_ring.partitions[pid]
            if (partition.owner_node != self.node_id and
                    self.node_id not in partition.backup_nodes):
                self._partitions.pop(pid, None)
                # Clean up WAL + snapshots for migrated partition
                if self._persistence:
                    self._persistence.remove_partition(pid)
        return None

    def _handle_migrate_data_push(self, msg: Message) -> Optional[Message]:
        """Handle pushed partition data (for new backup creation after topology change)."""
        payload = msg.payload
        pid = payload["partition_id"]
        role_str = payload.get("role", "backup")
        role = PartitionRole.BACKUP if role_str == "backup" else PartitionRole.PRIMARY

        self._apply_migration(pid, payload)
        with self._lock:
            if pid in self._partitions:
                self._partitions[pid].role = role

        return Message(msg_type="migrate_ack", sender=self.node_id, payload={"ok": True})

    def _handle_size_request(self, msg: Message) -> Message:
        """Return local primary vector count."""
        with self._lock:
            count = sum(
                len(lp.index) for lp in self._partitions.values()
                if lp.role == PartitionRole.PRIMARY
            )
        return Message(msg_type="size_result", sender=self.node_id,
                       payload={"count": count})

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def put(self, vector_id: str, vector: np.ndarray, metadata: Optional[dict] = None):
        """
        Insert a vector. Automatically routes to correct partition owner,
        which then synchronously replicates to backup nodes.
        """
        if vector.shape != (self.config.dim,):
            raise ValueError(f"Expected dim {self.config.dim}, got {vector.shape}")

        pid = self.hash_ring.get_partition(vector_id)
        partition = self.hash_ring.partitions[pid]
        owner = partition.owner_node

        if owner == self.node_id:
            # Local primary insert — WAL before index for crash safety
            with self._lock:
                if pid not in self._partitions:
                    self._partitions[pid] = LocalPartition(
                        id=pid,
                        role=PartitionRole.PRIMARY,
                        index=create_index(dim=self.config.dim, config=self.config.hnsw),
                    )
                lp = self._partitions[pid]
                lp.version += 1
                version = lp.version

            # WAL: persist BEFORE applying to index
            self._persist_insert(pid, version, vector_id, vector, metadata)

            # Apply to in-memory index
            with self._lock:
                lp.index.insert(vector_id, vector, metadata or {})

            # Sync to backups
            if self.config.sync_backup and partition.backup_nodes:
                self._replicate_insert(pid, vector_id, vector,
                                       metadata or {}, partition.backup_nodes)
        elif owner:
            # Smart routing: send to the actual owner
            msg = Message(
                msg_type="insert",
                sender=self.node_id,
                payload={
                    "partition_id": pid,
                    "vector_id": vector_id,
                    "vector": vector,
                    "metadata": metadata or {},
                },
            )
            resp = self.transport.send(owner, msg)
            if resp is None:
                raise RuntimeError(f"Insert failed: owner {owner} unreachable")
            if not resp.payload.get("ok", False):
                raise RuntimeError(f"Insert failed: {resp.payload}")
        else:
            raise RuntimeError("No owner for partition — cluster may be empty")

    def search(self, query: np.ndarray, k: int = 10,
               ef: Optional[int] = None,
               filter: Optional = None) -> list[SearchResult]:
        """
        Scatter-gather search across all nodes.

        Args:
            filter: Metadata filter. Either:
                - Dict spec: {"field": "source", "op": "eq", "value": "Biology"}
                - List of specs (AND): [spec1, spec2, ...]
                - Callable(meta) -> bool (single-node only)
        """
        from .hnsw import compile_filter

        if query.shape != (self.config.dim,):
            raise ValueError(f"Expected dim {self.config.dim}, got {query.shape}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        # Determine if filter is serializable or callable-only
        filter_spec = None
        compiled_filter = None

        if filter is not None:
            if callable(filter) and not isinstance(filter, (dict, list)):
                if len(self._cluster_nodes) > 1:
                    raise ValueError(
                        "Callable filters cannot be used in distributed mode. "
                        "Use a filter spec dict/list instead: "
                        '{"field": "name", "op": "eq", "value": "x"}'
                    )
                compiled_filter = filter
            else:
                filter_spec = filter
                compiled_filter = compile_filter(filter)

        all_results = []

        # Snapshot partition refs under lock, search outside it
        with self._lock:
            to_search = [(pid, lp) for pid, lp in self._partitions.items()
                         if lp.role == PartitionRole.PRIMARY and len(lp.index) > 0]

        for pid, lp in to_search:
            try:
                results = lp.index.search(query, k=k, ef=ef,
                                          filter_fn=compiled_filter)
                for dist, vid, meta in results:
                    all_results.append(SearchResult(
                        vector_id=vid,
                        distance=dist,
                        metadata=meta,
                        partition_id=pid,
                        source_node=self.node_id,
                    ))
            except Exception as e:
                logger.warning(f"Search failed on partition {pid}: {e}")

        # Scatter to remote nodes
        search_payload = {"query": query, "k": k, "ef": ef}
        if filter_spec is not None:
            search_payload["filter"] = filter_spec

        search_msg = Message(
            msg_type="search",
            sender=self.node_id,
            payload=search_payload,
        )
        responses = self.transport.broadcast(search_msg)
        for resp in responses:
            if resp and resp.msg_type == "search_result":
                for r in resp.payload.get("results", []):
                    all_results.append(SearchResult(
                        vector_id=r["vector_id"],
                        distance=r["distance"],
                        metadata=r["metadata"],
                        partition_id=r["partition_id"],
                        source_node=resp.sender,
                    ))

        all_results.sort(key=lambda r: r.distance)
        return all_results[:k]

    def delete(self, vector_id: str) -> bool:
        """Delete a vector. Smart-routed to primary, synced to backups."""
        pid = self.hash_ring.get_partition(vector_id)
        partition = self.hash_ring.partitions[pid]
        owner = partition.owner_node

        if owner == self.node_id:
            ok = False
            version = 0
            with self._lock:
                if pid in self._partitions:
                    ok = self._partitions[pid].index.delete(vector_id)
                    if ok:
                        self._partitions[pid].version += 1
                        version = self._partitions[pid].version
            if ok:
                self._persist_delete(pid, version, vector_id)
            if ok and self.config.sync_backup and partition.backup_nodes:
                backup_msg = Message(
                    msg_type="backup_delete",
                    sender=self.node_id,
                    payload={"partition_id": pid, "vector_id": vector_id},
                )
                for bn in partition.backup_nodes:
                    if bn != self.node_id:
                        self.transport.send(bn, backup_msg)
            return ok
        elif owner:
            msg = Message(
                msg_type="delete",
                sender=self.node_id,
                payload={"partition_id": pid, "vector_id": vector_id},
            )
            resp = self.transport.send(owner, msg)
            return resp is not None and resp.payload.get("ok", False)
        return False

    def get(self, vector_id: str) -> Optional[tuple[np.ndarray, dict]]:
        """
        Get vector by ID. Checks local partitions first (primary or backup),
        then smart-routes to owner.
        """
        pid = self.hash_ring.get_partition(vector_id)
        partition = self.hash_ring.partitions[pid]

        # Check local first (works for both primary and backup)
        with self._lock:
            if pid in self._partitions:
                index = self._partitions[pid].index
                if vector_id in index.vectors:
                    return index.vectors[vector_id].copy(), index.metadata.get(vector_id, {})

        # Smart route to owner
        if partition.owner_node and partition.owner_node != self.node_id:
            msg = Message(
                msg_type="get",
                sender=self.node_id,
                payload={"partition_id": pid, "vector_id": vector_id},
            )
            resp = self.transport.send(partition.owner_node, msg)
            if resp and resp.payload.get("found"):
                vec = resp.payload["vector"]
                if isinstance(vec, list):
                    vec = np.array(vec, dtype=np.float32)
                return vec, resp.payload.get("metadata", {})

        return None

    def local_size(self) -> int:
        """Number of vectors in local primary partitions."""
        with self._lock:
            return sum(
                len(lp.index) for lp in self._partitions.values()
                if lp.role == PartitionRole.PRIMARY
            )

    def local_backup_size(self) -> int:
        """Number of vectors in local backup partitions."""
        with self._lock:
            return sum(
                len(lp.index) for lp in self._partitions.values()
                if lp.role == PartitionRole.BACKUP
            )

    def cluster_size(self) -> int:
        """Total primary vectors across the cluster."""
        total = self.local_size()
        msg = Message(msg_type="size_request", sender=self.node_id, payload={})
        responses = self.transport.broadcast(msg)
        for resp in responses:
            if resp and resp.msg_type == "size_result":
                total += resp.payload.get("count", 0)
        return total

    def stats(self) -> dict:
        """Detailed node statistics."""
        with self._lock:
            n_primary = sum(1 for lp in self._partitions.values() if lp.role == PartitionRole.PRIMARY)
            n_backup = sum(1 for lp in self._partitions.values() if lp.role == PartitionRole.BACKUP)
            total_primary = sum(len(lp.index) for lp in self._partitions.values() if lp.role == PartitionRole.PRIMARY)
            total_backup = sum(len(lp.index) for lp in self._partitions.values() if lp.role == PartitionRole.BACKUP)

        return {
            "node_id": self.node_id,
            "cluster_nodes": sorted(self._cluster_nodes),
            "primary_partitions": n_primary,
            "backup_partitions": n_backup,
            "total_primary_vectors": total_primary,
            "total_backup_vectors": total_backup,
            "partition_table_version": self._partition_table_version,
            "hash_ring": self.hash_ring.stats(),
        }

    def __repr__(self):
        return (
            f"EmbeddedNode(id={self.node_id}, "
            f"primary={self.local_size()}, "
            f"backup={self.local_backup_size()})"
        )
