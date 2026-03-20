#!/usr/bin/env python3
"""VecGrid Benchmark Suite"""
import time, numpy as np, sys
from vecgrid import VecGrid, InProcessTransport
from vecgrid.hnsw import get_backend_name

print(f"Backend: {get_backend_name()}\n")

def bench(n_vec, dim, n_nodes, n_q, k, ef):
    InProcessTransport.reset()
    np.random.seed(42)
    label = f"{n_vec:,} vectors | dim={dim} | {n_nodes} nodes | k={k} | ef={ef}"
    print(f"--- {label} ---")

    nodes = [VecGrid(f"n-{i}", dim=dim, num_partitions=271, backup_count=1,
                     hnsw_config={"ef_construction": 200, "ef_search": ef}).start()
             for i in range(n_nodes)]

    vecs = np.random.randn(n_vec, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    t0 = time.time()
    for i in range(n_vec):
        nodes[i % n_nodes].put(f"v-{i}", vecs[i])
    ins_t = time.time() - t0
    ins_rate = n_vec / ins_t

    total_pri = sum(n.local_size() for n in nodes)
    total_bak = sum(n.local_backup_size() for n in nodes)
    print(f"  Insert: {ins_t:.2f}s ({ins_rate:,.0f} vec/s)")
    print(f"  Primary: {total_pri:,} | Backup: {total_bak:,}")

    qs = np.random.randn(n_q, dim).astype(np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    print(f"  Computing ground truth...")
    gt = []
    for q in qs:
        d = 1.0 - vecs @ q
        gt.append({f"v-{i}" for i in np.argsort(d)[:k]})

    print(f"  Searching...")
    recall_sum = 0.0
    lats = []
    for i, q in enumerate(qs):
        t0 = time.time()
        res = nodes[0].search(q, k=k)
        lats.append((time.time() - t0) * 1000)
        result_ids = {r.vector_id for r in res}
        recall_sum += len(result_ids & gt[i]) / k

    avg_recall = recall_sum / n_q
    avg_lat = np.mean(lats)
    p50_lat = np.median(lats)
    p95_lat = np.percentile(lats, 95)
    qps = n_q / (sum(lats) / 1000)

    print(f"  Recall@{k}: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"  Latency: avg={avg_lat:.1f}ms | p50={p50_lat:.1f}ms | p95={p95_lat:.1f}ms")
    print(f"  QPS: {qps:.1f}")

    # Failover
    killed_pri = nodes[1].local_size()
    nodes[1].stop()
    after = nodes[0].cluster_size()
    loss = total_pri - after
    print(f"  Failover: killed n-1 ({killed_pri} vectors) -> {after:,} remain | loss={loss}")

    for n in [nodes[0]] + nodes[2:]:
        n.stop()
    print()
    return (n_vec, dim, avg_recall, avg_lat, p95_lat, qps, ins_rate, loss)

results = []
results.append(bench(1000, 128, 3, 100, 10, 50))
results.append(bench(5000, 128, 3, 100, 10, 100))
results.append(bench(10000, 128, 3, 50, 10, 100))

print("=" * 78)
print("BENCHMARK SUMMARY")
print("=" * 78)
header = f"{'Vectors':>8} {'Dim':>5} {'Recall':>8} {'Avg ms':>8} {'P95 ms':>8} {'QPS':>7} {'Ins/s':>9} {'Loss':>5}"
print(header)
print("-" * 78)
for v, d, r, a, p, q, ins, loss in results:
    print(f"{v:>8,} {d:>5} {r:>8.3f} {a:>8.1f} {p:>8.1f} {q:>7.1f} {ins:>9,.0f} {loss:>5}")

print(f"\nBackend: {get_backend_name()}")
print("Tip: pip install hnswlib for 50-100x faster search")
