from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .__init__ import VecGrid


class AdminAPI:
    """Administration API for advanced partition management."""

    def __init__(self, grid: "VecGrid"):
        self._grid = grid

    def partition_ids(self) -> list[int]:
        """List all partition IDs in the cluster (same across all nodes)."""
        return list(range(self._grid._node.config.num_partitions))

    def local_partitions(self, role: str = "primary") -> list[int]:
        """
        List partition IDs currently managed by this local node.

        Args:
            role: "primary" or "backup"
        """
        from .node import PartitionRole

        target_role = PartitionRole.PRIMARY if role == "primary" else PartitionRole.BACKUP
        # Take the lock to guard against concurrent topology changes
        with self._grid._node._lock:
            return [
                pid
                for pid, lp in self._grid._node._partitions.items()
                if lp.role == target_role
            ]

    def export_partition(self, partition_id: int) -> dict:
        """
        Export all vectors and metadata for a partition.

        Routes to the primary owner automatically.

        Returns:
            dict with keys "vectors", "metadata", "version".
            "vectors" is empty if the partition has no data or does not exist.
        """
        from .transport import Message

        partition = self._grid._node.hash_ring.partitions.get(partition_id)
        if not partition:
            return {"vectors": {}, "metadata": {}, "version": 0}

        owner = partition.owner_node
        if not owner:
            return {"vectors": {}, "metadata": {}, "version": 0}

        msg = Message(
            msg_type="admin_export",
            sender=self._grid.node_id,
            payload={"partition_id": partition_id},
        )

        if owner == self._grid.node_id:
            resp = self._grid._node._handle_admin_export(msg)
        else:
            resp = self._grid._node.transport.send(owner, msg)

        if resp and resp.payload.get("found"):
            return {
                "vectors": resp.payload.get("vectors", {}),
                "metadata": resp.payload.get("metadata", {}),
                "version": resp.payload.get("version", 0),
            }
        # Partition exists but is empty
        return {"vectors": {}, "metadata": {}, "version": 0}

    def rebuild_partition_hnsw(self, partition_id: int) -> bool:
        """
        Trigger a zero-downtime background HNSW index rebuild for a partition.

        Routes to the primary owner automatically. Inserts and deletes that
        arrive while the new graph is being built are buffered and replayed
        before the index pointer is swapped, so no writes are lost.

        Returns:
            True if the rebuild was accepted (started in background).
            False if the partition does not exist or has no owner.
        """
        from .transport import Message

        partition = self._grid._node.hash_ring.partitions.get(partition_id)
        if not partition:
            return False

        owner = partition.owner_node
        if not owner:
            return False

        msg = Message(
            msg_type="admin_rebuild",
            sender=self._grid.node_id,
            payload={"partition_id": partition_id},
        )

        if owner == self._grid.node_id:
            resp = self._grid._node._handle_admin_rebuild(msg)
        else:
            resp = self._grid._node.transport.send(owner, msg)

        return resp is not None and resp.payload.get("ok", False)
