"""
Persistence layer for VecGrid.

Implements Write-Ahead Log (WAL) + periodic snapshots for durability.

Design:
    data_dir/
        wal/
            partition_000.wal     # Append-only log of mutations
            partition_005.wal
            ...
        snapshots/
            snapshot_000_v42.bin  # Full index dump at version 42
            snapshot_005_v38.bin
            ...
        meta.json               # Cluster metadata, last snapshot versions

Recovery flow:
    1. Load latest snapshot for each partition
    2. Replay WAL entries AFTER the snapshot version
    3. Index is fully restored

WAL entry format (one JSON line per mutation):
    {"op": "insert", "vid": "doc-1", "vec": [...], "meta": {...}, "ver": 42}
    {"op": "delete", "vid": "doc-1", "ver": 43}
"""

import json
import os
import struct
import threading
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger("vecgrid.persistence")


@dataclass
class WALEntry:
    """A single WAL entry."""
    op: str             # "insert" or "delete"
    vector_id: str
    version: int
    vector: Optional[np.ndarray] = None
    metadata: Optional[dict] = None

    def to_bytes(self) -> bytes:
        """Serialize to length-prefixed JSON line."""
        data = {
            "op": self.op,
            "vid": self.vector_id,
            "ver": self.version,
        }
        if self.op == "insert":
            data["vec"] = self.vector.tolist()
            if self.metadata:
                data["meta"] = self.metadata
        line = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return struct.pack("!I", len(line)) + line

    @classmethod
    def from_bytes(cls, raw: bytes) -> "WALEntry":
        data = json.loads(raw.decode("utf-8"))
        vec = None
        if "vec" in data:
            vec = np.array(data["vec"], dtype=np.float32)
        return cls(
            op=data["op"],
            vector_id=data["vid"],
            version=data["ver"],
            vector=vec,
            metadata=data.get("meta"),
        )


class WALWriter:
    """
    Append-only WAL writer for a single partition.
    
    Thread-safe. Writes are fsynced to disk before returning
    to guarantee durability.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._file = None
        self._entry_count = 0

    def open(self):
        """Open WAL file for appending."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "ab")
        # Count existing entries
        self._entry_count = self._count_entries()

    def append(self, entry: WALEntry):
        """Append an entry and fsync."""
        with self._lock:
            if self._file is None:
                raise RuntimeError("WAL not open")
            data = entry.to_bytes()
            self._file.write(data)
            self._file.flush()
            os.fsync(self._file.fileno())
            self._entry_count += 1

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None

    def truncate(self):
        """Truncate WAL after snapshot (entries are now in snapshot)."""
        with self._lock:
            if self._file:
                self._file.close()
            self._file = open(self.filepath, "wb")  # Truncate
            self._file.flush()
            os.fsync(self._file.fileno())
            self._entry_count = 0

    def _count_entries(self) -> int:
        count = 0
        try:
            with open(self.filepath, "rb") as f:
                while True:
                    length_data = f.read(4)
                    if len(length_data) < 4:
                        break
                    length = struct.unpack("!I", length_data)[0]
                    f.seek(length, 1)
                    count += 1
        except FileNotFoundError:
            pass
        return count

    @property
    def entry_count(self) -> int:
        return self._entry_count


class WALReader:
    """Read WAL entries from disk."""

    @staticmethod
    def read_all(filepath: Path) -> list[WALEntry]:
        """Read all entries from a WAL file."""
        entries = []
        try:
            with open(filepath, "rb") as f:
                while True:
                    length_data = f.read(4)
                    if len(length_data) < 4:
                        break
                    length = struct.unpack("!I", length_data)[0]
                    if length > 100_000_000:  # 100MB sanity limit
                        logger.warning(f"WAL entry too large ({length} bytes) in {filepath}, stopping read")
                        break
                    raw = f.read(length)
                    if len(raw) < length:
                        logger.warning(f"Truncated WAL entry in {filepath}")
                        break
                    try:
                        entries.append(WALEntry.from_bytes(raw))
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Corrupt WAL entry in {filepath}, skipping: {e}")
                        continue
        except FileNotFoundError:
            pass
        return entries

    @staticmethod
    def read_after_version(filepath: Path, after_version: int) -> list[WALEntry]:
        """Read WAL entries with version > after_version."""
        all_entries = WALReader.read_all(filepath)
        return [e for e in all_entries if e.version > after_version]


class SnapshotManager:
    """
    Manages full index snapshots.
    
    Snapshot format (binary):
        Header: 
            4 bytes magic "VGSN"
            4 bytes version (uint32)
            4 bytes dim (uint32)
            4 bytes num_vectors (uint32)
        Per vector:
            4 bytes vid_length (uint32)
            vid_length bytes vector_id (utf-8)
            dim * 4 bytes vector (float32)
            4 bytes meta_length (uint32)
            meta_length bytes metadata (json utf-8)
    """

    MAGIC = b"VGSN"

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save(self, partition_id: int, version: int, dim: int,
             vectors: dict[str, np.ndarray], metadata: dict[str, dict]):
        """Save a full snapshot of a partition."""
        filename = f"snapshot_{partition_id:04d}_v{version}.bin"
        filepath = self.snapshot_dir / filename
        tmp_path = filepath.with_suffix(".tmp")

        with open(tmp_path, "wb") as f:
            # Header
            f.write(self.MAGIC)
            f.write(struct.pack("!III", version, dim, len(vectors)))

            # Vectors
            for vid, vec in vectors.items():
                vid_bytes = vid.encode("utf-8")
                f.write(struct.pack("!I", len(vid_bytes)))
                f.write(vid_bytes)
                f.write(vec.astype(np.float32).tobytes())

                meta = metadata.get(vid, {})
                meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
                f.write(struct.pack("!I", len(meta_bytes)))
                f.write(meta_bytes)

            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(tmp_path, filepath)

        # Clean up old snapshots for this partition
        self._cleanup_old_snapshots(partition_id, keep_version=version)

        logger.debug(f"Snapshot saved: partition {partition_id} v{version} "
                     f"({len(vectors)} vectors)")

    def load_latest(self, partition_id: int) -> Optional[tuple[int, int, dict, dict]]:
        """
        Load the latest snapshot for a partition.
        Returns (version, dim, vectors_dict, metadata_dict) or None.
        """
        prefix = f"snapshot_{partition_id:04d}_v"
        best_file = None
        best_version = -1

        for f in self.snapshot_dir.iterdir():
            if f.name.startswith(prefix) and f.suffix == ".bin":
                try:
                    ver_str = f.stem.split("_v")[-1]
                    ver = int(ver_str)
                    if ver > best_version:
                        best_version = ver
                        best_file = f
                except (ValueError, IndexError):
                    continue

        if best_file is None:
            return None

        return self._read_snapshot(best_file)

    def _read_snapshot(self, filepath: Path) -> Optional[tuple[int, int, dict, dict]]:
        """Read a snapshot file."""
        try:
            with open(filepath, "rb") as f:
                magic = f.read(4)
                if magic != self.MAGIC:
                    logger.error(f"Invalid snapshot magic: {filepath}")
                    return None

                header = f.read(12)
                if len(header) < 12:
                    logger.error(f"Truncated snapshot header: {filepath}")
                    return None
                version, dim, num_vectors = struct.unpack("!III", header)

                if dim == 0 or dim > 100_000:  # Sanity check
                    logger.error(f"Invalid dim {dim} in snapshot: {filepath}")
                    return None

                vectors = {}
                metadata = {}

                for i in range(num_vectors):
                    vid_len_data = f.read(4)
                    if len(vid_len_data) < 4:
                        logger.warning(f"Truncated snapshot at vector {i}: {filepath}")
                        break
                    vid_len = struct.unpack("!I", vid_len_data)[0]
                    if vid_len > 10_000:  # Sanity check for vector ID length
                        logger.warning(f"Unreasonable vid_len {vid_len} at vector {i}: {filepath}")
                        break
                    vid = f.read(vid_len).decode("utf-8")

                    vec_bytes = f.read(dim * 4)
                    if len(vec_bytes) < dim * 4:
                        logger.warning(f"Truncated vector data at {vid}: {filepath}")
                        break
                    vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                    vectors[vid] = vec

                    meta_len_data = f.read(4)
                    if len(meta_len_data) < 4:
                        logger.warning(f"Truncated metadata length at {vid}: {filepath}")
                        break
                    meta_len = struct.unpack("!I", meta_len_data)[0]
                    if meta_len > 0:
                        meta_bytes = f.read(meta_len)
                        if len(meta_bytes) < meta_len:
                            logger.warning(f"Truncated metadata at {vid}: {filepath}")
                            break
                        try:
                            metadata[vid] = json.loads(meta_bytes.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            logger.warning(f"Corrupt metadata for {vid}: {e}")
                            metadata[vid] = {}
                    else:
                        metadata[vid] = {}

                return version, dim, vectors, metadata

        except Exception as e:
            logger.error(f"Failed to read snapshot {filepath}: {e}")
            return None

    def _cleanup_old_snapshots(self, partition_id: int, keep_version: int):
        """Remove old snapshots, keep only the latest."""
        prefix = f"snapshot_{partition_id:04d}_v"
        for f in self.snapshot_dir.iterdir():
            if f.name.startswith(prefix) and f.suffix == ".bin":
                try:
                    ver_str = f.stem.split("_v")[-1]
                    ver = int(ver_str)
                    if ver < keep_version:
                        f.unlink()
                except (ValueError, IndexError):
                    continue


class PersistenceEngine:
    """
    High-level persistence manager for VecGrid.

    Coordinates WAL writes, snapshot creation, and recovery
    for all partitions on a node.

    Usage:
        engine = PersistenceEngine("/var/data/vecgrid/node-1")
        engine.open()
        
        # On every write:
        engine.log_insert(partition_id, version, vector_id, vector, metadata)
        
        # Periodically:
        engine.snapshot(partition_id, version, vectors, metadata)
        
        # On recovery:
        entries = engine.recover(partition_id, dim)
        # Returns (snapshot_version, vectors, metadata, wal_entries_after_snapshot)
    """

    def __init__(self, data_dir: str, snapshot_interval: int = 1000):
        """
        Args:
            data_dir: Root directory for this node's data
            snapshot_interval: Take snapshot every N WAL entries per partition
        """
        self.data_dir = Path(data_dir)
        self.wal_dir = self.data_dir / "wal"
        self.snapshot_dir = self.data_dir / "snapshots"
        self.snapshot_interval = snapshot_interval

        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._wal_writers: dict[int, WALWriter] = {}
        self._snapshot_mgr = SnapshotManager(self.snapshot_dir)
        self._lock = threading.Lock()
        self._running = False

    def open(self):
        """Open the persistence engine."""
        self._running = True
        logger.info(f"Persistence engine opened: {self.data_dir}")

    def close(self):
        """Close all WAL writers."""
        self._running = False
        with self._lock:
            for writer in self._wal_writers.values():
                writer.close()
            self._wal_writers.clear()

    def _get_wal_writer(self, partition_id: int) -> WALWriter:
        """Get or create WAL writer for a partition."""
        if partition_id not in self._wal_writers:
            filepath = self.wal_dir / f"partition_{partition_id:04d}.wal"
            writer = WALWriter(filepath)
            writer.open()
            self._wal_writers[partition_id] = writer
        return self._wal_writers[partition_id]

    def log_insert(self, partition_id: int, version: int,
                   vector_id: str, vector: np.ndarray,
                   metadata: Optional[dict] = None):
        """Log an insert to the WAL. Called BEFORE applying to index."""
        with self._lock:
            writer = self._get_wal_writer(partition_id)
            entry = WALEntry(
                op="insert",
                vector_id=vector_id,
                version=version,
                vector=vector,
                metadata=metadata,
            )
            writer.append(entry)

    def log_delete(self, partition_id: int, version: int, vector_id: str):
        """Log a delete to the WAL."""
        with self._lock:
            writer = self._get_wal_writer(partition_id)
            entry = WALEntry(
                op="delete",
                vector_id=vector_id,
                version=version,
            )
            writer.append(entry)

    def should_snapshot(self, partition_id: int) -> bool:
        """Check if this partition should take a snapshot based on WAL size."""
        with self._lock:
            writer = self._wal_writers.get(partition_id)
            if writer:
                return writer.entry_count >= self.snapshot_interval
        return False

    def snapshot(self, partition_id: int, version: int, dim: int,
                 vectors: dict[str, np.ndarray], metadata: dict[str, dict]):
        """
        Take a full snapshot of a partition and truncate its WAL.
        """
        # Save snapshot first
        self._snapshot_mgr.save(partition_id, version, dim, vectors, metadata)

        # Then truncate WAL (snapshot has all the data)
        with self._lock:
            writer = self._wal_writers.get(partition_id)
            if writer:
                writer.truncate()

    def recover(self, partition_id: int, dim: int
                ) -> tuple[int, dict[str, np.ndarray], dict[str, dict], list[WALEntry]]:
        """
        Recover a partition from snapshot + WAL.

        Returns:
            (snapshot_version, vectors, metadata, wal_entries_after_snapshot)
        """
        # Load latest snapshot
        snapshot_data = self._snapshot_mgr.load_latest(partition_id)

        if snapshot_data:
            snap_version, snap_dim, vectors, metadata = snapshot_data
            if snap_dim != dim:
                logger.warning(f"Snapshot dim {snap_dim} != expected {dim}")
        else:
            snap_version = 0
            vectors = {}
            metadata = {}

        # Replay WAL entries after snapshot
        wal_path = self.wal_dir / f"partition_{partition_id:04d}.wal"
        wal_entries = WALReader.read_after_version(wal_path, snap_version)

        return snap_version, vectors, metadata, wal_entries

    def get_persisted_partitions(self) -> set[int]:
        """Get set of partition IDs that have data on disk."""
        partitions = set()

        # Check WAL files
        for f in self.wal_dir.iterdir():
            if f.name.startswith("partition_") and f.suffix == ".wal":
                try:
                    pid = int(f.stem.split("_")[1])
                    if f.stat().st_size > 0:
                        partitions.add(pid)
                except (ValueError, IndexError):
                    continue

        # Check snapshots
        for f in self.snapshot_dir.iterdir():
            if f.name.startswith("snapshot_") and f.suffix == ".bin":
                try:
                    pid = int(f.stem.split("_")[1])
                    partitions.add(pid)
                except (ValueError, IndexError):
                    continue

        return partitions

    def remove_partition(self, partition_id: int):
        """Remove all persistence data for a partition (after migration)."""
        with self._lock:
            writer = self._wal_writers.pop(partition_id, None)
            if writer:
                writer.close()

        # Remove WAL
        wal_path = self.wal_dir / f"partition_{partition_id:04d}.wal"
        if wal_path.exists():
            wal_path.unlink()

        # Remove snapshots
        prefix = f"snapshot_{partition_id:04d}_v"
        for f in self.snapshot_dir.iterdir():
            if f.name.startswith(prefix):
                f.unlink()

    def stats(self) -> dict:
        """Persistence statistics."""
        wal_sizes = {}
        total_wal_bytes = 0
        for f in self.wal_dir.iterdir():
            if f.suffix == ".wal":
                size = f.stat().st_size
                total_wal_bytes += size
                try:
                    pid = int(f.stem.split("_")[1])
                    wal_sizes[pid] = size
                except (ValueError, IndexError):
                    pass

        snapshot_count = sum(1 for f in self.snapshot_dir.iterdir() if f.suffix == ".bin")

        return {
            "data_dir": str(self.data_dir),
            "wal_files": len(wal_sizes),
            "total_wal_bytes": total_wal_bytes,
            "snapshot_count": snapshot_count,
            "snapshot_interval": self.snapshot_interval,
        }
