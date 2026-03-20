"""
VecGrid test suite.

Run with: python -m pytest tests/ -v
Or directly: python tests/test_vecgrid.py
"""

import os
import shutil
import tempfile
import numpy as np

def compute_exact_knn(query, vectors_dict, k, metric="cosine", metadatas_dict=None, filter_ext=None):
    from vecgrid.hnsw import compile_filter
    filter_fn = compile_filter(filter_ext)
    
    candidates = []
    for vid, vec in vectors_dict.items():
        if filter_fn is not None:
            meta = metadatas_dict.get(vid, {}) if metadatas_dict else {}
            if not filter_fn(meta):
                continue
                
        if metric == "cosine":
            norm_q = float(np.linalg.norm(query))
            norm_v = float(np.linalg.norm(vec))
            if norm_q > 0 and norm_v > 0:
                dist = 1.0 - float(np.dot(query, vec) / (norm_q * norm_v))
            else:
                dist = 1.0
        else:
            dist = float(np.linalg.norm(query - vec))
        candidates.append((dist, vid))
        
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [vid for dist, vid in candidates[:k]]

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from vecgrid import VecGrid, InProcessTransport
from vecgrid.persistence import PersistenceEngine, WALEntry, WALWriter, WALReader, SnapshotManager
from pathlib import Path


class _raises:
    """Minimal pytest.raises replacement."""
    def __init__(self, exc_type):
        self.exc_type = exc_type
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} but nothing was raised")
        if issubclass(exc_type, self.exc_type):
            return True  # Suppress the exception
        return False


# ----------------------------------------------------------------
# HNSW + Single Node
# ----------------------------------------------------------------

class TestSingleNode:
    def test_insert_and_search(self):
        with VecGrid("solo", dim=32) as grid:
            np.random.seed(1)
            vectors = {}
            for i in range(100):
                vec = np.random.randn(32).astype(np.float32)
                grid.put(f"v-{i}", vec, {"i": i})
                vectors[f"v-{i}"] = vec

            query = np.random.randn(32).astype(np.float32)
            results = grid.search(query, k=5)
            assert len(results) == 5
            
            exact_ids = compute_exact_knn(query, vectors, k=5)
            assert [r.vector_id for r in results] == exact_ids

    def test_delete(self):
        with VecGrid("solo", dim=16) as grid:
            vec = np.random.randn(16).astype(np.float32)
            grid.put("x", vec, {"a": 1})
            assert grid.local_size() == 1
            assert grid.delete("x") is True
            assert grid.local_size() == 0
            assert grid.delete("x") is False

    def test_get(self):
        with VecGrid("solo", dim=8) as grid:
            vec = np.random.randn(8).astype(np.float32)
            grid.put("doc-1", vec, {"title": "hello"})
            result = grid.get("doc-1")
            assert result is not None
            recovered_vec, meta = result
            assert np.allclose(recovered_vec, vec)
            assert meta["title"] == "hello"
            assert grid.get("nonexistent") is None

    def test_dimension_mismatch_raises(self):
        with VecGrid("solo", dim=16) as grid:
            with _raises(ValueError):
                grid.put("bad", np.random.randn(32).astype(np.float32))


# ----------------------------------------------------------------
# Multi-Node Cluster
# ----------------------------------------------------------------

class TestCluster:
    def test_three_node_cluster(self):
        nodes = [VecGrid(f"n-{i}", dim=32).start() for i in range(3)]
        np.random.seed(10)

        vectors = {}
        for i in range(300):
            vec = np.random.randn(32).astype(np.float32)
            nodes[0].put(f"v-{i}", vec)
            vectors[f"v-{i}"] = vec

        total = sum(n.local_size() for n in nodes)
        assert total == 300

        query = np.random.randn(32).astype(np.float32)
        r0 = [r.vector_id for r in nodes[0].search(query, k=10)]
        r1 = [r.vector_id for r in nodes[1].search(query, k=10)]
        r2 = [r.vector_id for r in nodes[2].search(query, k=10)]
        assert r0 == r1 == r2  # Consistency
        
        exact_ids = compute_exact_knn(query, vectors, k=10)
        assert r0 == exact_ids

        for n in nodes:
            n.stop()

    def test_cluster_size(self):
        nodes = [VecGrid(f"n-{i}", dim=16).start() for i in range(3)]
        for i in range(90):
            nodes[i % 3].put(f"v-{i}", np.random.randn(16).astype(np.float32))
        assert nodes[0].cluster_size() == 90
        for n in nodes:
            n.stop()


# ----------------------------------------------------------------
# Sync Backup Replication
# ----------------------------------------------------------------

class TestReplication:
    def test_backup_count(self):
        nodes = [VecGrid(f"n-{i}", dim=16, backup_count=1).start() for i in range(3)]

        for i in range(100):
            nodes[0].put(f"v-{i}", np.random.randn(16).astype(np.float32))

        total_primary = sum(n.local_size() for n in nodes)
        total_backup = sum(n.local_backup_size() for n in nodes)
        assert total_primary == 100
        assert total_backup == 100  # 1:1 backup ratio

        for n in nodes:
            n.stop()

    def test_zero_data_loss_on_node_failure(self):
        nodes = [VecGrid(f"n-{i}", dim=16, backup_count=1).start() for i in range(3)]
        np.random.seed(42)

        for i in range(200):
            nodes[0].put(f"v-{i}", np.random.randn(16).astype(np.float32))

        size_before = nodes[0].cluster_size()
        assert size_before == 200

        # Kill node-1
        nodes[1].stop()

        size_after = nodes[0].cluster_size()
        assert size_after == 200  # Zero data loss

        nodes[0].stop()
        nodes[2].stop()


# ----------------------------------------------------------------
# Smart Routing
# ----------------------------------------------------------------

class TestSmartRouting:
    def test_insert_from_any_node(self):
        nodes = [VecGrid(f"n-{i}", dim=16).start() for i in range(3)]

        for i in range(60):
            nodes[i % 3].put(f"v-{i}", np.random.randn(16).astype(np.float32))

        assert nodes[0].cluster_size() == 60
        for n in nodes:
            n.stop()

    def test_get_from_any_node(self):
        nodes = [VecGrid(f"n-{i}", dim=16).start() for i in range(3)]

        vec = np.random.randn(16).astype(np.float32)
        nodes[0].put("target", vec, {"found": True})

        # Should be retrievable from any node
        for n in nodes:
            result = n.get("target")
            assert result is not None
            assert np.allclose(result[0], vec)

        for n in nodes:
            n.stop()


# ----------------------------------------------------------------
# Safe Migration
# ----------------------------------------------------------------

class TestMigration:
    def test_no_data_loss_on_node_join(self):
        nodes = [VecGrid(f"n-{i}", dim=16, backup_count=1).start() for i in range(2)]

        for i in range(300):
            nodes[0].put(f"v-{i}", np.random.randn(16).astype(np.float32))

        size_before = nodes[0].cluster_size()
        assert size_before == 300

        # Add third node
        node2 = VecGrid("n-2", dim=16, backup_count=1).start()
        nodes.append(node2)

        size_after = nodes[0].cluster_size()
        assert size_after == 300  # No data loss

        for n in nodes:
            n.stop()


# ----------------------------------------------------------------
# Filtered Search (Predicate Pushdown)
# ----------------------------------------------------------------

class TestFilteredSearch:
    """Test predicate pushdown / filter-aware search."""

    def _make_grid(self):
        grid = VecGrid("solo", dim=16, num_partitions=7)
        grid.start()
        np.random.seed(42)
        categories = ["Biology", "Physics", "Math", "History"]
        self.vectors = {}
        self.metadatas = {}
        for i in range(200):
            vec = np.random.randn(16).astype(np.float32)
            meta = {
                "category": categories[i % 4],
                "year": 2018 + (i % 6),
                "score": float(i),
            }
            grid.put(f"v-{i}", vec, meta)
            self.vectors[f"v-{i}"] = vec
            self.metadatas[f"v-{i}"] = meta
        return grid

    def test_eq_filter(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        f_spec = {"field": "category", "op": "eq", "value": "Biology"}
        results = grid.search(query, k=10, filter=f_spec)
        exact_ids = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_ne_filter(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        f_spec = {"field": "category", "op": "ne", "value": "History"}
        results = grid.search(query, k=10, filter=f_spec)
        exact_ids = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_in_filter(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        f_spec = {"field": "category", "op": "in", "value": ["Biology", "Physics"]}
        results = grid.search(query, k=10, filter=f_spec)
        exact_ids = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_gt_filter(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        f_spec = {"field": "year", "op": "gt", "value": 2020}
        results = grid.search(query, k=10, filter=f_spec)
        exact_ids = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_gte_lte_filter(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)

        f_spec_gte = {"field": "year", "op": "gte", "value": 2021}
        results_gte = grid.search(query, k=10, filter=f_spec_gte)
        exact_ids_gte = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec_gte)
        assert [r.vector_id for r in results_gte] == exact_ids_gte

        f_spec_lte = {"field": "year", "op": "lte", "value": 2019}
        results_lte = grid.search(query, k=10, filter=f_spec_lte)
        exact_ids_lte = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec_lte)
        assert [r.vector_id for r in results_lte] == exact_ids_lte

        grid.stop()

    def test_lt_filter(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        f_spec = {"field": "year", "op": "lt", "value": 2020}
        results = grid.search(query, k=10, filter=f_spec)
        exact_ids = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_and_conditions(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        f_spec = [
            {"field": "category", "op": "eq", "value": "Biology"},
            {"field": "year", "op": "gt", "value": 2020},
        ]
        results = grid.search(query, k=10, filter=f_spec)
        exact_ids = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_no_matches_returns_empty(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        results = grid.search(query, k=10,
                              filter={"field": "category", "op": "eq",
                                      "value": "Nonexistent"})
        assert len(results) == 0
        grid.stop()

    def test_no_filter_backward_compat(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        results = grid.search(query, k=10)
        assert len(results) == 10
        exact_ids = compute_exact_knn(query, self.vectors, k=10)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_callable_filter_local(self):
        grid = self._make_grid()
        query = np.random.randn(16).astype(np.float32)
        f_spec = lambda meta: meta.get("category") == "Math"
        results = grid.search(query, k=10, filter=f_spec)
        exact_ids = compute_exact_knn(query, self.vectors, k=10, metadatas_dict=self.metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids
        grid.stop()

    def test_filtered_search_cluster(self):
        nodes = [VecGrid(f"n-{i}", dim=16, num_partitions=7).start()
                 for i in range(3)]
        np.random.seed(10)
        categories = ["A", "B", "C"]
        vectors = {}
        metadatas = {}
        for i in range(300):
            vec = np.random.randn(16).astype(np.float32)
            meta = {"cat": categories[i % 3]}
            nodes[0].put(f"v-{i}", vec, meta)
            vectors[f"v-{i}"] = vec
            metadatas[f"v-{i}"] = meta

        query = np.random.randn(16).astype(np.float32)
        f_spec = {"field": "cat", "op": "eq", "value": "A"}
        results = nodes[0].search(query, k=10, filter=f_spec)
        assert len(results) > 0
        
        exact_ids = compute_exact_knn(query, vectors, k=10, metadatas_dict=metadatas, filter_ext=f_spec)
        assert [r.vector_id for r in results] == exact_ids

        # All nodes should return same filtered results
        r1 = [r.vector_id for r in nodes[1].search(query, k=10, filter=f_spec)]
        r2 = [r.vector_id for r in nodes[2].search(query, k=10, filter=f_spec)]
        assert r1 == exact_ids and r2 == exact_ids

        for n in nodes:
            n.stop()


# ----------------------------------------------------------------
# Persistence: WAL
# ----------------------------------------------------------------

class TestWAL:
    def test_wal_write_read(self, tmp_dir):
        filepath = Path(tmp_dir) / "test.wal"
        writer = WALWriter(filepath)
        writer.open()

        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        writer.append(WALEntry(op="insert", vector_id="v-1", version=1,
                               vector=vec, metadata={"a": 1}))
        writer.append(WALEntry(op="delete", vector_id="v-1", version=2))
        writer.close()

        entries = WALReader.read_all(filepath)
        assert len(entries) == 2
        assert entries[0].op == "insert"
        assert entries[0].vector_id == "v-1"
        assert np.allclose(entries[0].vector, vec)
        assert entries[0].metadata == {"a": 1}
        assert entries[1].op == "delete"

    def test_wal_read_after_version(self, tmp_dir):
        filepath = Path(tmp_dir) / "test.wal"
        writer = WALWriter(filepath)
        writer.open()

        for i in range(10):
            writer.append(WALEntry(op="insert", vector_id=f"v-{i}", version=i+1,
                                   vector=np.zeros(3, dtype=np.float32)))
        writer.close()

        entries = WALReader.read_after_version(filepath, 7)
        assert len(entries) == 3
        assert entries[0].version == 8


# ----------------------------------------------------------------
# Persistence: Snapshots
# ----------------------------------------------------------------

class TestSnapshot:
    def test_save_and_load(self, tmp_dir):
        mgr = SnapshotManager(Path(tmp_dir))

        vectors = {
            "v-0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "v-1": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        }
        metadata = {"v-0": {"a": 1}, "v-1": {"b": 2}}

        mgr.save(partition_id=0, version=10, dim=3,
                 vectors=vectors, metadata=metadata)

        result = mgr.load_latest(partition_id=0)
        assert result is not None
        version, dim, loaded_vecs, loaded_meta = result
        assert version == 10
        assert dim == 3
        assert len(loaded_vecs) == 2
        assert np.allclose(loaded_vecs["v-0"], vectors["v-0"])
        assert loaded_meta["v-1"] == {"b": 2}

    def test_latest_snapshot_wins(self, tmp_dir):
        mgr = SnapshotManager(Path(tmp_dir))
        vec = np.zeros(2, dtype=np.float32)

        mgr.save(0, 5, 2, {"a": vec}, {"a": {}})
        mgr.save(0, 10, 2, {"a": vec, "b": vec}, {"a": {}, "b": {}})

        result = mgr.load_latest(0)
        assert result[0] == 10  # Version
        assert len(result[2]) == 2  # Vectors


# ----------------------------------------------------------------
# Persistence: Full Recovery
# ----------------------------------------------------------------

class TestPersistenceRecovery:
    def test_survive_restart(self, tmp_dir):
        """Insert, shutdown cleanly, restart, verify all data recovered."""
        data_dir = os.path.join(tmp_dir, "node-0")
        dim = 16
        np.random.seed(42)

        # Phase 1: Insert and shutdown
        grid = VecGrid("node-0", dim=dim, data_dir=data_dir, snapshot_interval=50)
        grid.start()

        vectors = {}
        for i in range(150):
            vec = np.random.randn(dim).astype(np.float32)
            vectors[f"v-{i}"] = vec
            grid.put(f"v-{i}", vec, {"i": i})

        query = np.random.randn(dim).astype(np.float32)
        results_before = [r.vector_id for r in grid.search(query, k=10)]
        grid.stop()

        # Phase 2: Restart
        InProcessTransport.reset()
        grid2 = VecGrid("node-0", dim=dim, data_dir=data_dir, snapshot_interval=50)
        grid2.start()

        assert grid2.local_size() == 150
        results_after = [r.vector_id for r in grid2.search(query, k=10)]
        assert results_before == results_after

        # Spot check vectors
        for vid in ["v-0", "v-50", "v-149"]:
            result = grid2.get(vid)
            assert result is not None
            assert np.allclose(result[0], vectors[vid], atol=1e-6)

        grid2.stop()

    def test_survive_crash_wal_replay(self, tmp_dir):
        """Insert, simulate crash (no clean shutdown), recover from WAL."""
        data_dir = os.path.join(tmp_dir, "node-0")
        dim = 8
        np.random.seed(99)

        grid = VecGrid("node-0", dim=dim, data_dir=data_dir, snapshot_interval=50)
        grid.start()

        # Insert enough to trigger snapshot
        for i in range(80):
            grid.put(f"v-{i}", np.random.randn(dim).astype(np.float32))

        # Insert more (WAL only, no snapshot)
        extra = {}
        for i in range(80, 100):
            vec = np.random.randn(dim).astype(np.float32)
            extra[f"v-{i}"] = vec
            grid.put(f"v-{i}", vec)

        # Simulate crash: close persistence without snapshot
        grid._node._persistence.close()
        InProcessTransport.reset()

        # Recover
        grid2 = VecGrid("node-0", dim=dim, data_dir=data_dir, snapshot_interval=50)
        grid2.start()

        assert grid2.local_size() == 100

        # WAL-replayed vectors must be correct
        for vid, original in extra.items():
            result = grid2.get(vid)
            assert result is not None, f"{vid} not recovered"
            assert np.allclose(result[0], original, atol=1e-6)

        grid2.stop()

    def test_delete_persists(self, tmp_dir):
        """Delete a vector, restart, verify it's gone."""
        data_dir = os.path.join(tmp_dir, "node-0")
        dim = 8

        grid = VecGrid("node-0", dim=dim, data_dir=data_dir, snapshot_interval=1000)
        grid.start()

        grid.put("keep", np.ones(dim, dtype=np.float32))
        grid.put("remove", np.ones(dim, dtype=np.float32))
        assert grid.local_size() == 2

        grid.delete("remove")
        assert grid.local_size() == 1
        grid.stop()

        InProcessTransport.reset()
        grid2 = VecGrid("node-0", dim=dim, data_dir=data_dir, snapshot_interval=1000)
        grid2.start()

        assert grid2.local_size() == 1
        assert grid2.get("keep") is not None
        assert grid2.get("remove") is None
        grid2.stop()


if __name__ == "__main__":
    import sys, traceback, inspect

    def run_test(cls_name, test_name, fn):
        """Run a test method, providing tmp_dir if needed."""
        InProcessTransport.reset()
        tmp = None
        try:
            sig = inspect.signature(fn)
            if "tmp_dir" in sig.parameters:
                tmp = tempfile.mkdtemp(prefix="vecgrid_test_")
                fn(tmp)
            else:
                fn()
            print(f"  ✓ {cls_name}.{test_name}")
            return True
        except Exception as e:
            print(f"  ✗ {cls_name}.{test_name}: {e}")
            traceback.print_exc()
            return False
        finally:
            InProcessTransport.reset()
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    test_classes = [
        TestSingleNode, TestCluster, TestReplication,
        TestSmartRouting, TestMigration, TestFilteredSearch,
        TestWAL, TestSnapshot, TestPersistenceRecovery,
    ]

    results = []
    for cls in test_classes:
        print(f"\n{cls.__name__}:")
        instance = cls()
        for name in sorted(dir(instance)):
            if name.startswith("test_"):
                results.append(run_test(cls.__name__, name, getattr(instance, name)))

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"{passed}/{total} tests passed")
    if passed == total:
        print("✓ ALL TESTS PASS")
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
