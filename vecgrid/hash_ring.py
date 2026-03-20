"""
Consistent Hash Ring for partition ownership.

Maps vector IDs to partitions, and partitions to owning nodes.
Supports dynamic node join/leave with minimal data movement.
"""

import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class Partition:
    """Represents a single partition."""
    id: int
    owner_node: str
    backup_nodes: list[str]


class ConsistentHashRing:
    """
    Consistent hash ring that maps keys to partitions,
    and partitions to nodes.
    
    Similar to Hazelcast's partition model:
    - Fixed number of partitions (default 271, prime for better distribution)
    - Partitions are assigned to nodes round-robin initially
    - When nodes join/leave, partitions are rebalanced
    """

    def __init__(self, num_partitions: int = 271, backup_count: int = 1):
        self.num_partitions = num_partitions
        self.backup_count = backup_count
        self.nodes: list[str] = []
        self.partitions: dict[int, Partition] = {}
        
        # Initialize empty partitions
        for i in range(num_partitions):
            self.partitions[i] = Partition(id=i, owner_node="", backup_nodes=[])

    def _hash_key(self, key: str) -> int:
        """Hash a key to a partition ID."""
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h, 16) % self.num_partitions

    def _rebalance(self):
        """Rebalance partition ownership across nodes."""
        if not self.nodes:
            for p in self.partitions.values():
                p.owner_node = ""
                p.backup_nodes = []
            return
        
        n = len(self.nodes)
        
        for pid, partition in self.partitions.items():
            # Primary owner: round-robin
            owner_idx = pid % n
            partition.owner_node = self.nodes[owner_idx]
            
            # Backup owners
            partition.backup_nodes = []
            for b in range(1, self.backup_count + 1):
                if b < n:
                    backup_idx = (pid + b) % n
                    partition.backup_nodes.append(self.nodes[backup_idx])

    def add_node(self, node_id: str) -> dict[int, tuple[str, str]]:
        """
        Add a node to the ring. Returns partition migration map:
        {partition_id: (from_node, to_node)}
        """
        if node_id in self.nodes:
            return {}
        
        # Capture current ownership
        old_owners = {pid: p.owner_node for pid, p in self.partitions.items()}
        
        self.nodes.append(node_id)
        self.nodes.sort()  # Deterministic ordering
        self._rebalance()
        
        # Compute migrations
        migrations = {}
        for pid, partition in self.partitions.items():
            old_owner = old_owners[pid]
            if old_owner and old_owner != partition.owner_node:
                migrations[pid] = (old_owner, partition.owner_node)
        
        return migrations

    def remove_node(self, node_id: str) -> dict[int, tuple[str, str]]:
        """
        Remove a node from the ring. Returns partition migration map.
        """
        if node_id not in self.nodes:
            return {}
        
        old_owners = {pid: p.owner_node for pid, p in self.partitions.items()}
        
        self.nodes.remove(node_id)
        self._rebalance()
        
        migrations = {}
        for pid, partition in self.partitions.items():
            old_owner = old_owners[pid]
            if old_owner == node_id and partition.owner_node:
                migrations[pid] = (old_owner, partition.owner_node)
        
        return migrations

    def get_partition(self, key: str) -> int:
        """Get partition ID for a key."""
        return self._hash_key(key)

    def get_owner(self, key: str) -> Optional[str]:
        """Get the owning node for a key."""
        pid = self._hash_key(key)
        return self.partitions[pid].owner_node or None

    def get_partition_owner(self, partition_id: int) -> Optional[str]:
        """Get the owning node for a partition."""
        p = self.partitions.get(partition_id)
        return p.owner_node if p and p.owner_node else None

    def get_node_partitions(self, node_id: str) -> list[int]:
        """Get all partition IDs owned by a node."""
        return [
            pid for pid, p in self.partitions.items()
            if p.owner_node == node_id
        ]

    def get_all_nodes(self) -> list[str]:
        """Get all nodes in the ring."""
        return list(self.nodes)

    def stats(self) -> dict:
        """Distribution statistics."""
        if not self.nodes:
            return {"nodes": 0, "partitions": self.num_partitions}
        
        dist = {node: 0 for node in self.nodes}
        for p in self.partitions.values():
            if p.owner_node:
                dist[p.owner_node] += 1
        
        counts = list(dist.values())
        return {
            "nodes": len(self.nodes),
            "partitions": self.num_partitions,
            "distribution": dist,
            "min_partitions_per_node": min(counts),
            "max_partitions_per_node": max(counts),
            "ideal_per_node": self.num_partitions / len(self.nodes),
        }
