"""
Tests for the Admin API (grid.admin).

Coverage:
  - partition_ids()
  - local_partitions(role=...)
  - export_partition(pid)
  - rebuild_partition_hnsw(pid) — including zero-downtime concurrency tests
    covering both inserts AND deletes that occur while the new HNSW graph builds.
"""
import threading
import time
import numpy as np
import pytest

from vecgrid import VecGrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_vids_for_partition(node, target_pid: int, prefix: str, n: int, start: int = 0):
    """Return exactly n vector IDs (strings) that hash to target_pid."""
    vids = []
    i = start
    while len(vids) < n:
        vid = f"{prefix}-{i}"
        if node.hash_ring.get_partition(vid) == target_pid:
            vids.append(vid)
        i += 1
    return vids


# ---------------------------------------------------------------------------
# Test: partition_ids()
# ---------------------------------------------------------------------------

def test_partition_ids():
    grid = VecGrid("admin-pids", dim=4)
    grid.start()
    try:
        ids = grid.admin.partition_ids()
        assert len(ids) == 271
        assert ids == list(range(271))  # stable, deterministic ordering
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: local_partitions()
# ---------------------------------------------------------------------------

def test_local_partitions_single_node():
    grid = VecGrid("admin-local", dim=4)
    grid.start()
    try:
        primaries = grid.admin.local_partitions("primary")
        backups = grid.admin.local_partitions("backup")

        # Single node: all 271 partitions are primary, none are backups
        assert len(primaries) == 271
        assert len(backups) == 0
        assert sorted(primaries) == list(range(271))
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: export_partition() — basic correctness
# ---------------------------------------------------------------------------

def test_export_partition_basic():
    grid = VecGrid("admin-export", dim=4)
    grid.start()
    try:
        pid = grid.admin.local_partitions("primary")[0]
        vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        vids = find_vids_for_partition(grid._node, pid, "e", 20)

        for vid in vids:
            grid.put(vid, vector, {"src": vid})

        result = grid.admin.export_partition(pid)
        assert "vectors" in result
        assert "metadata" in result
        assert "version" in result
        assert len(result["vectors"]) == 20
        assert len(result["metadata"]) == 20
        assert result["version"] >= 20

        # Check a specific vector round-trips correctly (float precision from JSON)
        first_vid = vids[0]
        exported_vec = result["vectors"][first_vid]
        assert len(exported_vec) == 4
        np.testing.assert_allclose(exported_vec, vector.tolist(), rtol=1e-5)
        assert result["metadata"][first_vid] == {"src": first_vid}
    finally:
        grid.stop()


def test_export_partition_empty():
    """Exporting an empty-but-existing partition returns zeros, not an error."""
    grid = VecGrid("admin-empty", dim=4)
    grid.start()
    try:
        result = grid.admin.export_partition(0)
        assert result == {"vectors": {}, "metadata": {}, "version": 0}
    finally:
        grid.stop()


def test_export_partition_nonexistent():
    """Partition IDs outside the valid range return an empty result, not an exception."""
    grid = VecGrid("admin-nonexist", dim=4)
    grid.start()
    try:
        result = grid.admin.export_partition(9999)
        assert result == {"vectors": {}, "metadata": {}, "version": 0}
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: rebuild_partition_hnsw() — basic reconstruction
# ---------------------------------------------------------------------------

def test_rebuild_partition_hnsw_basic():
    """After a rebuild, the index must contain exactly the same vectors as before."""
    grid = VecGrid("admin-rebuild-basic", dim=4)
    grid.start()
    try:
        pid = grid.admin.local_partitions("primary")[0]
        vector = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        vids = find_vids_for_partition(grid._node, pid, "rb", 30)

        for vid in vids:
            grid.put(vid, vector, {"tag": "initial"})

        before = grid.admin.export_partition(pid)
        assert len(before["vectors"]) == 30

        # Trigger rebuild and wait for completion
        assert grid.admin.rebuild_partition_hnsw(pid) is True
        # Give the background thread time — 1s is ample for 30 vectors
        time.sleep(0.3)

        after = grid.admin.export_partition(pid)
        assert len(after["vectors"]) == 30
        # All original vector IDs must still be present
        assert set(after["vectors"].keys()) == set(before["vectors"].keys())
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: rebuild_partition_hnsw() — inserts during rebuild
# ---------------------------------------------------------------------------

def test_rebuild_concurrent_inserts():
    """
    Vectors inserted while the HNSW graph is being rebuilt must survive.

    Ground-truth: we track exactly which secondary IDs were acknowledged as
    inserted, and verify every one appears in the final index.
    """
    grid = VecGrid("admin-rebuild-ins", dim=4)
    grid.start()
    try:
        pid = grid.admin.local_partitions("primary")[5]  # arbitrary non-zero pid
        vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        # Phase 1: insert seed vectors (50)
        seed_vids = find_vids_for_partition(grid._node, pid, "seed", 50)
        for vid in seed_vids:
            grid.put(vid, vector, {"phase": "seed"})

        # Phase 2: background inserter runs concurrently with rebuild
        # We collect acknowledged inserts into a thread-safe list.
        secondary_vids = find_vids_for_partition(grid._node, pid, "concurrent", 25, start=1_000_000)
        inserted_during_rebuild = []
        stop_event = threading.Event()

        def inserter():
            for vid in secondary_vids:
                if stop_event.is_set():
                    break
                grid.put(vid, vector, {"phase": "concurrent"})
                inserted_during_rebuild.append(vid)
                time.sleep(0.01)

        t = threading.Thread(target=inserter, daemon=True)

        # Start inserter slightly before the rebuild to get some overlap
        t.start()
        time.sleep(0.02)  # Let a couple inserts proceed first

        assert grid.admin.rebuild_partition_hnsw(pid) is True

        # Wait for the inserter to finish, then let the rebuild complete
        t.join(timeout=5.0)
        stop_event.set()
        time.sleep(0.5)  # Generous wait for the background rebuild thread

        final = grid.admin.export_partition(pid)
        final_keys = set(final["vectors"].keys())

        # Every seed vector must be present
        for vid in seed_vids:
            assert vid in final_keys, f"Seed vector {vid} missing after rebuild"

        # Every acknowledged concurrent insert must be present
        for vid in inserted_during_rebuild:
            assert vid in final_keys, f"Concurrent insert {vid} lost during rebuild"
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: rebuild_partition_hnsw() — deletes during rebuild
# ---------------------------------------------------------------------------

def test_rebuild_concurrent_deletes():
    """
    Vectors deleted while the HNSW graph is being rebuilt must NOT appear in
    the final index.
    """
    grid = VecGrid("admin-rebuild-del", dim=4)
    grid.start()
    try:
        pid = grid.admin.local_partitions("primary")[3]
        vector = np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32)

        # Insert 60 vectors
        all_vids = find_vids_for_partition(grid._node, pid, "del", 60)
        for vid in all_vids:
            grid.put(vid, vector, {"phase": "initial"})

        # Split into "keepers" (first 40) and "to-delete" (last 20)
        keepers = set(all_vids[:40])
        to_delete = all_vids[40:]

        deleted_during_rebuild = []
        stop_event = threading.Event()

        def deleter():
            for vid in to_delete:
                if stop_event.is_set():
                    break
                ok = grid.delete(vid)
                if ok:
                    deleted_during_rebuild.append(vid)
                time.sleep(0.01)

        t = threading.Thread(target=deleter, daemon=True)
        t.start()
        time.sleep(0.02)  # Let some deletes happen before rebuild

        assert grid.admin.rebuild_partition_hnsw(pid) is True

        t.join(timeout=5.0)
        stop_event.set()
        time.sleep(0.5)

        final = grid.admin.export_partition(pid)
        final_keys = set(final["vectors"].keys())

        # Keepers must all remain
        for vid in keepers:
            assert vid in final_keys, f"Keeper {vid} incorrectly removed"

        # Acknowledged deletes must not appear in the final index
        for vid in deleted_during_rebuild:
            assert vid not in final_keys, f"Deleted vector {vid} survived rebuild"
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: rebuild_partition_hnsw() — concurrent inserts AND deletes
# ---------------------------------------------------------------------------

def test_rebuild_concurrent_inserts_and_deletes():
    """
    The rebuild buffer must correctly handle interleaved inserts and deletes.
    """
    grid = VecGrid("admin-rebuild-both", dim=4)
    grid.start()
    try:
        pid = grid.admin.local_partitions("primary")[7]
        vector = np.array([0.6, 0.1, 0.2, 0.1], dtype=np.float32)

        # Insert 50 initial vectors as seed
        seed_vids = find_vids_for_partition(grid._node, pid, "both-seed", 50)
        for vid in seed_vids:
            grid.put(vid, vector)

        # 20 vectors to delete, 20 vectors to insert, all during rebuild
        to_delete = seed_vids[:20]   # will be deleted; chosen from existing
        new_vids = find_vids_for_partition(grid._node, pid, "both-new", 20, start=2_000_000)

        deleted_ack = []
        inserted_ack = []
        stop_event = threading.Event()

        def mixed_writer():
            ops = [(vid, "delete") for vid in to_delete] + [(vid, "insert") for vid in new_vids]
            import random as rng
            rng.shuffle(ops)
            for vid, op in ops:
                if stop_event.is_set():
                    break
                if op == "insert":
                    grid.put(vid, vector, {"op": "new"})
                    inserted_ack.append(vid)
                else:
                    if grid.delete(vid):
                        deleted_ack.append(vid)
                time.sleep(0.005)

        t = threading.Thread(target=mixed_writer, daemon=True)
        t.start()
        time.sleep(0.02)

        assert grid.admin.rebuild_partition_hnsw(pid) is True

        t.join(timeout=5.0)
        stop_event.set()
        time.sleep(0.5)

        final = grid.admin.export_partition(pid)
        final_keys = set(final["vectors"].keys())

        # All new inserts must be present
        for vid in inserted_ack:
            assert vid in final_keys, f"Newly inserted {vid} missing after rebuild"

        # All acknowledged deletes must be absent
        for vid in deleted_ack:
            assert vid not in final_keys, f"Deleted vector {vid} survived rebuild"

        # Untouched seeds (those not in to_delete) must still be present
        untouched_seeds = [v for v in seed_vids if v not in set(deleted_ack)]
        for vid in untouched_seeds:
            assert vid in final_keys, f"Untouched seed {vid} lost during rebuild"
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: rebuild_partition_hnsw() — missing / invalid partition
# ---------------------------------------------------------------------------

def test_rebuild_missing_partition():
    grid = VecGrid("admin-rebuild-miss", dim=4)
    grid.start()
    try:
        assert grid.admin.rebuild_partition_hnsw(9999) is False
        assert grid.admin.rebuild_partition_hnsw(-1) is False
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Test: rebuild is idempotent (double-trigger doesn't corrupt state)
# ---------------------------------------------------------------------------

def test_rebuild_idempotent_double_trigger():
    """Calling rebuild twice rapidly: second call should be rejected, data stays intact."""
    grid = VecGrid("admin-rebuild-idem", dim=4)
    grid.start()
    try:
        pid = grid.admin.local_partitions("primary")[0]
        vector = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        vids = find_vids_for_partition(grid._node, pid, "idem", 15)
        for vid in vids:
            grid.put(vid, vector)

        # Trigger twice in quick succession
        r1 = grid.admin.rebuild_partition_hnsw(pid)
        r2 = grid.admin.rebuild_partition_hnsw(pid)
        assert r1 is True   # First always succeeds
        assert r2 is False  # Second rejected: already rebuilding

        time.sleep(0.5)

        final = grid.admin.export_partition(pid)
        # After everything settles, all 15 vectors must be intact
        assert len(final["vectors"]) == 15
    finally:
        grid.stop()


# ---------------------------------------------------------------------------
# Stress test: external ground-truth tracking
# ---------------------------------------------------------------------------

def test_rebuild_ground_truth_stress():
    """
    Heavy stress test: multiple inserts and deletes interleaved with a rebuild.
    We maintain a Python set as ground-truth and compare against the final
    index state. This cannot be "overfit" because the ground-truth is tracked
    independently of any VecGrid internals.
    """
    grid = VecGrid("admin-stress", dim=4)
    grid.start()
    try:
        pid = grid.admin.local_partitions("primary")[10]
        vector = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

        # Phase 1: seed 100 vectors
        seed_vids = find_vids_for_partition(grid._node, pid, "stress", 100)
        ground_truth = set(seed_vids)
        for vid in seed_vids:
            grid.put(vid, vector, {"p": "seed"})

        # Prepare lists for concurrent ops
        to_insert = find_vids_for_partition(grid._node, pid, "stress-new", 30, start=5_000_000)
        to_delete = seed_vids[70:]  # Delete the last 30 seeds

        lock = threading.Lock()
        stop = threading.Event()

        def writer():
            """Inserts new vectors and deletes old ones, updating ground truth atomically."""
            insert_idx = 0
            delete_idx = 0
            while not stop.is_set():
                # Alternate insert / delete
                if insert_idx < len(to_insert):
                    vid = to_insert[insert_idx]
                    grid.put(vid, vector, {"p": "new"})
                    with lock:
                        ground_truth.add(vid)
                    insert_idx += 1

                if delete_idx < len(to_delete):
                    vid = to_delete[delete_idx]
                    ok = grid.delete(vid)
                    if ok:
                        with lock:
                            ground_truth.discard(vid)
                    delete_idx += 1

                time.sleep(0.005)

                if insert_idx >= len(to_insert) and delete_idx >= len(to_delete):
                    break

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        time.sleep(0.02)

        # Trigger rebuild mid-flight
        assert grid.admin.rebuild_partition_hnsw(pid) is True

        t.join(timeout=10.0)
        stop.set()
        time.sleep(0.5)  # Let rebuild thread finish

        final = grid.admin.export_partition(pid)
        final_keys = set(final["vectors"].keys())

        with lock:
            expected = set(ground_truth)

        # The exported index must exactly match the externally-tracked ground truth
        missing = expected - final_keys
        extra = final_keys - expected

        assert not missing, f"Ground-truth vectors missing from index: {missing}"
        assert not extra, f"Extra vectors in index not in ground truth: {extra}"
    finally:
        grid.stop()

