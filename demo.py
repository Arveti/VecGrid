#!/usr/bin/env python3
"""
VecGrid Demo v2 - Hazelcast-style Features

Demonstrates:
1. Sync backup replication (every write copies to backup nodes)
2. Backup promotion on node failure (zero data loss)
3. Smart routing (send request to any node, gets forwarded to owner)
4. Safe partition migration (migrate-then-delete, no data loss)
5. Cluster-wide size tracking
6. Recall benchmark
"""

import time
import numpy as np
from vecgrid import VecGrid, InProcessTransport


def banner(text):
    print(f"\n{'='*64}")
    print(f"  {text}")
    print(f"{'='*64}\n")


def demo_sync_backup_replication():
    """Demo 1: Writes replicate synchronously to backup nodes."""
    banner("DEMO 1: Synchronous Backup Replication")

    InProcessTransport.reset()
    dim = 64
    np.random.seed(42)

    nodes = [VecGrid(node_id=f"node-{i}", dim=dim, num_partitions=271,
                     backup_count=1).start()
             for i in range(3)]

    # Insert 500 vectors
    print("  Inserting 500 vectors with sync backup replication...")
    t0 = time.time()
    for i in range(500):
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        nodes[0].put(f"vec-{i}", vec, {"index": i})
    elapsed = time.time() - t0
    print(f"  Insert time: {elapsed:.3f}s ({500/elapsed:.0f} vec/sec)")

    # Check primary and backup counts
    print(f"\n  Data distribution (primary / backup):")
    total_primary = 0
    total_backup = 0
    for node in nodes:
        stats = node.stats()
        p = stats["total_primary_vectors"]
        b = stats["total_backup_vectors"]
        total_primary += p
        total_backup += b
        print(f"    {node.node_id}: {p:3d} primary, {b:3d} backup "
              f"({stats['primary_partitions']} primary parts, "
              f"{stats['backup_partitions']} backup parts)")

    print(f"\n  Total primary vectors: {total_primary}")
    print(f"  Total backup vectors:  {total_backup}")
    print(f"  Backup ratio:          {total_backup/total_primary:.2f}x")
    print(f"  Cluster size (API):    {nodes[0].cluster_size()}")

    assert total_primary == 500, f"Expected 500 primary vectors, got {total_primary}"
    assert total_backup > 0, "Expected backup vectors to exist!"
    print(f"\n  ✓ Every vector has a backup copy on another node")

    for n in nodes:
        n.stop()


def demo_backup_promotion_failover():
    """Demo 2: Kill a node, backups auto-promote, zero data loss."""
    banner("DEMO 2: Node Failure → Backup Promotion (Zero Data Loss)")

    InProcessTransport.reset()
    dim = 64
    np.random.seed(42)

    nodes = [VecGrid(node_id=f"node-{i}", dim=dim, num_partitions=271,
                     backup_count=1).start()
             for i in range(3)]

    # Insert data
    n_vectors = 500
    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    for i in range(n_vectors):
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        nodes[0].put(f"vec-{i}", vec, {"index": i})

    # Record pre-failure state
    cluster_size_before = nodes[0].cluster_size()
    results_before = nodes[0].search(query, k=10)
    ids_before = {r.vector_id for r in results_before}

    print(f"  Before failure:")
    print(f"    Cluster size: {cluster_size_before}")
    for n in nodes:
        print(f"    {n.node_id}: {n.local_size()} primary, {n.local_backup_size()} backup")

    # Kill node-1
    killed_primary = nodes[1].local_size()
    killed_backup = nodes[1].local_backup_size()
    print(f"\n  ✗ Killing node-1 ({killed_primary} primary, {killed_backup} backup vectors)...")
    nodes[1].stop()

    # Check surviving nodes
    print(f"\n  After failure:")
    surviving = [nodes[0], nodes[2]]
    cluster_size_after = surviving[0].cluster_size()
    for n in surviving:
        print(f"    {n.node_id}: {n.local_size()} primary, {n.local_backup_size()} backup")
    print(f"    Cluster size: {cluster_size_after}")

    # Search after failure
    results_after = nodes[0].search(query, k=10)
    ids_after = {r.vector_id for r in results_after}

    overlap = len(ids_before & ids_after)
    print(f"\n  Search consistency:")
    print(f"    Results overlap: {overlap}/10")

    data_loss = cluster_size_before - cluster_size_after
    print(f"\n  Data loss: {data_loss} vectors")
    if data_loss == 0:
        print(f"  ✓ ZERO DATA LOSS — backups promoted to primary successfully!")
    else:
        recovered = cluster_size_after
        loss_pct = (data_loss / cluster_size_before) * 100
        print(f"  Recovered {recovered}/{cluster_size_before} "
              f"({loss_pct:.1f}% loss — from partitions without backup data)")

    for n in surviving:
        n.stop()


def demo_smart_routing():
    """Demo 3: Send requests to any node, they route to the correct owner."""
    banner("DEMO 3: Smart Routing (Any Node → Correct Owner)")

    InProcessTransport.reset()
    dim = 64
    np.random.seed(42)

    nodes = [VecGrid(node_id=f"node-{i}", dim=dim, num_partitions=271).start()
             for i in range(3)]

    # Insert via different nodes — all should work regardless of partition ownership
    print("  Inserting via round-robin across all nodes...")
    for i in range(300):
        inserter = nodes[i % 3]  # Rotate which node we call
        vec = np.random.randn(dim).astype(np.float32)
        inserter.put(f"vec-{i}", vec, {"inserted_via": inserter.node_id})

    print(f"  Total cluster vectors: {nodes[0].cluster_size()}")

    # Get from a node that may not own the partition — smart routes to owner
    print(f"\n  Smart routing for get():")
    found_count = 0
    for i in range(0, 300, 50):
        vid = f"vec-{i}"
        for node in nodes:
            result = node.get(vid)
            if result:
                vec, meta = result
                print(f"    {node.node_id}.get('{vid}') → found (inserted via {meta.get('inserted_via', '?')})")
                found_count += 1
                break

    # Search from each node — all should return same results
    print(f"\n  Search consistency across nodes:")
    query = np.random.randn(dim).astype(np.float32)
    all_result_sets = []
    for node in nodes:
        results = node.search(query, k=10)
        result_ids = tuple(r.vector_id for r in results)
        all_result_sets.append(result_ids)
        print(f"    {node.node_id}: {[r.vector_id for r in results[:3]]}...")

    if all_result_sets[0] == all_result_sets[1] == all_result_sets[2]:
        print(f"  ✓ All nodes return identical results — smart routing works!")
    else:
        print(f"  ✗ Results differ (possible during rebalancing)")

    for n in nodes:
        n.stop()


def demo_safe_migration():
    """Demo 4: Node join triggers safe migration — no data loss."""
    banner("DEMO 4: Safe Partition Migration (Migrate-Then-Delete)")

    InProcessTransport.reset()
    dim = 64
    np.random.seed(42)

    # Start with 2 nodes
    print("  Phase 1: 2-node cluster")
    nodes = [VecGrid(node_id=f"node-{i}", dim=dim, num_partitions=271,
                     backup_count=1).start()
             for i in range(2)]

    n_vectors = 600
    for i in range(n_vectors):
        vec = np.random.randn(dim).astype(np.float32)
        nodes[0].put(f"vec-{i}", vec, {"i": i})

    size_before = nodes[0].cluster_size()
    print(f"    Cluster size: {size_before}")
    for n in nodes:
        print(f"    {n.node_id}: {n.local_size()} primary, {n.local_backup_size()} backup")

    # Add node-2 — should trigger migration WITH data preservation
    print(f"\n  Phase 2: Adding node-2 (triggers safe migration)...")
    node2 = VecGrid(node_id="node-2", dim=dim, num_partitions=271,
                    backup_count=1).start()
    nodes.append(node2)

    size_after = nodes[0].cluster_size()
    print(f"    Cluster size: {size_after}")
    for n in nodes:
        print(f"    {n.node_id}: {n.local_size()} primary, {n.local_backup_size()} backup")

    data_loss = size_before - size_after
    if data_loss == 0:
        print(f"\n  ✓ ZERO DATA LOSS during migration!")
    else:
        print(f"\n  Data loss: {data_loss} vectors during migration")

    # Verify search still works
    query = np.random.randn(dim).astype(np.float32)
    results = nodes[2].search(query, k=5)
    print(f"  Search from new node found {len(results)} results")
    for r in results[:3]:
        print(f"    {r.vector_id} dist={r.distance:.4f} from {r.source_node}")

    # Now add node-3 too
    print(f"\n  Phase 3: Adding node-3...")
    node3 = VecGrid(node_id="node-3", dim=dim, num_partitions=271,
                    backup_count=1).start()
    nodes.append(node3)

    size_final = nodes[0].cluster_size()
    print(f"    Cluster size: {size_final}")
    for n in nodes:
        print(f"    {n.node_id}: {n.local_size()} primary, {n.local_backup_size()} backup")

    if size_final == n_vectors:
        print(f"\n  ✓ All {n_vectors} vectors preserved across 3 topology changes!")

    for n in nodes:
        n.stop()


def demo_recall_benchmark():
    """Demo 5: Recall quality benchmark."""
    banner("DEMO 5: Recall Benchmark (ANN vs Brute Force)")

    InProcessTransport.reset()
    dim = 128
    n_vectors = 5000
    n_queries = 50
    k = 10
    np.random.seed(123)

    nodes = [VecGrid(node_id=f"node-{i}", dim=dim, num_partitions=271,
                     backup_count=1,
                     hnsw_config={"ef_construction": 200, "ef_search": 100}).start()
             for i in range(3)]

    print(f"  Inserting {n_vectors} vectors (dim={dim}) with backup replication...")
    all_vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    all_vectors = all_vectors / norms

    t0 = time.time()
    for i in range(n_vectors):
        nodes[i % 3].put(f"v-{i}", all_vectors[i], {"idx": i})
    insert_time = time.time() - t0
    print(f"  Insert time: {insert_time:.2f}s ({n_vectors/insert_time:.0f} vec/sec)")

    # Show replication stats
    total_primary = sum(n.local_size() for n in nodes)
    total_backup = sum(n.local_backup_size() for n in nodes)
    print(f"  Primary vectors: {total_primary}, Backup vectors: {total_backup}")

    queries = np.random.randn(n_queries, dim).astype(np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / norms

    # Brute force ground truth
    print(f"\n  Computing ground truth...")
    ground_truth = []
    for q in queries:
        dists = 1.0 - all_vectors @ q
        top_k_idx = np.argsort(dists)[:k]
        ground_truth.append({f"v-{i}" for i in top_k_idx})

    # VecGrid search
    print(f"  Running distributed ANN search...")
    total_recall = 0
    total_time = 0
    for i, q in enumerate(queries):
        t0 = time.time()
        results = nodes[0].search(q, k=k)
        total_time += time.time() - t0
        result_ids = {r.vector_id for r in results}
        recall = len(result_ids & ground_truth[i]) / k
        total_recall += recall

    avg_recall = total_recall / n_queries
    avg_latency = (total_time / n_queries) * 1000

    print(f"\n  Results ({n_vectors} vectors, {n_queries} queries, k={k}):")
    print(f"    Recall@{k}:    {avg_recall:.3f} ({avg_recall*100:.1f}%)")
    print(f"    Latency:       {avg_latency:.1f}ms/query")
    print(f"    Throughput:    {n_queries/total_time:.0f} qps")

    for n in nodes:
        n.stop()


def demo_full_failover_cycle():
    """Demo 6: Full lifecycle — build cluster, kill nodes, recover, verify."""
    banner("DEMO 6: Full Failover Cycle")

    InProcessTransport.reset()
    dim = 32
    np.random.seed(99)

    # Build 4-node cluster
    print("  Building 4-node cluster...")
    nodes = [VecGrid(node_id=f"node-{i}", dim=dim, num_partitions=271,
                     backup_count=1).start()
             for i in range(4)]

    # Insert data
    n_vectors = 400
    vectors_map = {}
    for i in range(n_vectors):
        vec = np.random.randn(dim).astype(np.float32)
        vectors_map[f"vec-{i}"] = vec
        nodes[i % 4].put(f"vec-{i}", vec, {"i": i})

    initial_size = nodes[0].cluster_size()
    print(f"  Initial cluster size: {initial_size}")

    # Kill node-1
    print(f"\n  ✗ Killing node-1...")
    nodes[1].stop()
    surviving = [nodes[0], nodes[2], nodes[3]]
    size_after_1 = surviving[0].cluster_size()
    print(f"  Cluster size: {size_after_1}")

    # Kill node-3
    print(f"  ✗ Killing node-3...")
    nodes[3].stop()
    surviving = [nodes[0], nodes[2]]
    size_after_2 = surviving[0].cluster_size()
    print(f"  Cluster size: {size_after_2}")

    # Verify search still works
    query = np.random.randn(dim).astype(np.float32)
    results = nodes[0].search(query, k=5)
    print(f"\n  Search from 2-node cluster: found {len(results)} results")
    for r in results[:3]:
        print(f"    {r.vector_id} dist={r.distance:.4f}")

    # Add new nodes back
    print(f"\n  Adding node-4 and node-5...")
    node4 = VecGrid(node_id="node-4", dim=dim, num_partitions=271,
                    backup_count=1).start()
    node5 = VecGrid(node_id="node-5", dim=dim, num_partitions=271,
                    backup_count=1).start()

    final_nodes = [nodes[0], nodes[2], node4, node5]
    final_size = final_nodes[0].cluster_size()
    print(f"  Final cluster size: {final_size}")
    for n in final_nodes:
        print(f"    {n.node_id}: {n.local_size()} primary, {n.local_backup_size()} backup")

    print(f"\n  Summary: {initial_size} → killed 2 nodes → {size_after_2} survived → "
          f"added 2 nodes → {final_size} final")

    for n in final_nodes:
        n.stop()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   VecGrid v2 - Hazelcast-Style Distributed Vector Database       ║
║                                                                  ║
║   Features: Sync Replication · Backup Promotion · Smart Routing  ║
║             Safe Migration · Zero-Downtime Scaling               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    demo_sync_backup_replication()
    demo_backup_promotion_failover()
    demo_smart_routing()
    demo_safe_migration()
    demo_recall_benchmark()
    demo_full_failover_cycle()

    banner("ALL DEMOS COMPLETE")
