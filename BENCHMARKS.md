# VecGrid Benchmarks

All benchmarks run on the numpy (pure Python) backend. Install `hnswlib` for 50-100x faster performance on the hot path.

## Setup

- 3-node embedded cluster (InProcessTransport)
- 271 partitions, backup_count=1 (sync replication)
- Cosine distance, ef_construction=200
- Random normalized vectors, random queries
- Ground truth computed via brute-force (numpy matmul)
- Recall@10 = fraction of true top-10 found by VecGrid

## Results (numpy backend)

```
 Vectors   Dim   Recall@10   Avg ms   P95 ms     QPS     Ins/s  Data Loss
---------------------------------------------------------------------------
   1,000   128      1.000     10.1     58.1    99.2     9,770         0
   5,000   128      1.000     41.4     77.3    24.2     4,189         0
  10,000   128      1.000     93.8    114.4    10.7       623         0
```

### Key observations

**Recall is 100% across all tested sizes.** This is because 10,000 vectors across 271 partitions = ~37 vectors per partition. HNSW graphs of that size are effectively brute-force — the graph is dense enough that the algorithm visits every vector. This is expected behavior and will decrease at larger scale (see projections below).

**Insert throughput degrades at 10K** because HNSW graph construction is O(N log N) and each insert requires traversing the growing graph in pure Python. With hnswlib (C++), expect 50,000-100,000 inserts/sec at this scale.

**Search latency scales linearly with vector count** in the numpy backend because distance computation dominates. With hnswlib, expect sub-5ms at 10K vectors.

**Zero data loss on every failover test.** Backup replication works correctly — killing any node preserves all data through backup promotion.

## Expected results with hnswlib backend

Based on published hnswlib benchmarks and the fact that VecGrid's overhead is only the scatter-gather coordination (not the index itself):

```
 Vectors   Dim   Est. Recall   Est. Avg ms   Est. QPS   Est. Ins/s
--------------------------------------------------------------------
  10,000   128     1.000           1-3        300-500     50,000+
 100,000   128     0.97-0.99       3-8        100-300     40,000+
 500,000   128     0.95-0.98      10-30        30-100     30,000+
1,000,000  128     0.93-0.97      20-60        15-50      20,000+
```

These estimates assume ef_search=100 and M=16. Higher ef_search improves recall at the cost of latency.

## Recall vs partition count

Fewer partitions → more vectors per partition → larger HNSW graphs → potentially lower recall (but also fewer scatter-gather round-trips):

```
 Partitions   Vectors/part (at 100K)   Expected Recall@10
-----------------------------------------------------------
       271                      369             0.98-1.00
       100                    1,000             0.96-0.99
        50                    2,000             0.95-0.98
        13                    7,692             0.93-0.97
```

The default 271 partitions (prime, Hazelcast convention) provides the best recall at moderate scale due to dense per-partition graphs.

## Failover performance

Every benchmark test kills one node and verifies:
- Cluster size unchanged (backup promotion successful)
- Data loss = 0 in all cases
- Search continues working from surviving nodes

## Running benchmarks yourself

```bash
# Basic benchmark (1K, 5K, 10K)
PYTHONPATH=. python3 benchmark.py

# For larger scale (requires hnswlib):
# pip install hnswlib
# Edit benchmark.py to add larger vector counts
```

## How to benchmark at 100K+ vectors

The numpy backend is too slow for 100K+ vector benchmarks. Install hnswlib first:

```bash
pip install hnswlib
```

Then modify `benchmark.py` to add:

```python
results.append(bench(100000, 128, 3, 100, 10, 100))
```

VecGrid auto-detects hnswlib and uses the C++ backend. Expected runtime: ~2-3 minutes for 100K insert + search.
