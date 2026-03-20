# VecGrid Manual Test Guide

Step-by-step validation of every feature. Run these in order.
Each test is a self-contained Python script you can copy-paste into a terminal.

**Prerequisites:**
- Python 3.10+
- numpy (`pip install numpy`)
- Clone or copy the vecgrid folder
- Run all scripts from the vecgrid project root with `PYTHONPATH=.`

---

## Test 1: Basic Insert and Search (Single Node)

**What it tests:** HNSW index, put, search, get, delete — the fundamentals.

```bash
cd vecgrid
PYTHONPATH=. python3 -c "
import numpy as np
from vecgrid import VecGrid, InProcessTransport

InProcessTransport.reset()
grid = VecGrid('solo', dim=64)
grid.start()

# Insert 500 vectors
np.random.seed(42)
for i in range(500):
    vec = np.random.randn(64).astype(np.float32)
    grid.put(f'doc-{i}', vec, {'title': f'Document {i}', 'category': i % 5})

print(f'Inserted: {grid.local_size()} vectors')
assert grid.local_size() == 500, 'FAIL: wrong count'

# Search
query = np.random.randn(64).astype(np.float32)
results = grid.search(query, k=10)
print(f'Search returned: {len(results)} results')
assert len(results) == 10, 'FAIL: wrong result count'

# Distances should be sorted ascending
dists = [r.distance for r in results]
assert dists == sorted(dists), 'FAIL: results not sorted'
print(f'Distances: {[f\"{d:.4f}\" for d in dists[:5]]}...')

# Get by ID
vec, meta = grid.get('doc-42')
assert meta['title'] == 'Document 42', 'FAIL: wrong metadata'
print(f'Get doc-42: meta={meta}')

# Delete
assert grid.delete('doc-42') == True
assert grid.get('doc-42') is None
assert grid.local_size() == 499
print(f'After delete: {grid.local_size()} vectors')

grid.stop()
print()
print('✓ TEST 1 PASSED: Basic insert, search, get, delete work')
"
```

**Expected:** 500 inserted, 10 search results sorted by distance, get returns correct metadata, delete works.

---

## Test 2: Multi-Node Cluster (Embedded Mode)

**What it tests:** 3-node cluster formation, consistent hashing, scatter-gather search consistency.

```bash
PYTHONPATH=. python3 -c "
import numpy as np
from vecgrid import VecGrid, InProcessTransport

InProcessTransport.reset()

nodes = [VecGrid(f'node-{i}', dim=64, num_partitions=271).start() for i in range(3)]

np.random.seed(42)
for i in range(600):
    vec = np.random.randn(64).astype(np.float32)
    nodes[i % 3].put(f'vec-{i}', vec)  # Round-robin insert

# Check distribution
for n in nodes:
    print(f'{n.node_id}: {n.local_size()} primary vectors')

total = sum(n.local_size() for n in nodes)
assert total == 600, f'FAIL: expected 600, got {total}'

# Search from each node — results must be identical
query = np.random.randn(64).astype(np.float32)
r0 = [r.vector_id for r in nodes[0].search(query, k=10)]
r1 = [r.vector_id for r in nodes[1].search(query, k=10)]
r2 = [r.vector_id for r in nodes[2].search(query, k=10)]

print(f'node-0 results: {r0[:5]}...')
print(f'node-1 results: {r1[:5]}...')
print(f'node-2 results: {r2[:5]}...')

assert r0 == r1 == r2, 'FAIL: inconsistent search results across nodes'

for n in nodes:
    n.stop()

print()
print('✓ TEST 2 PASSED: Multi-node cluster with consistent search results')
"
```

**Expected:** ~200 vectors per node, all 3 nodes return identical top-10.

---

## Test 3: Sync Backup Replication

**What it tests:** Every write creates a backup copy on another node.

```bash
PYTHONPATH=. python3 -c "
import numpy as np
from vecgrid import VecGrid, InProcessTransport

InProcessTransport.reset()
nodes = [VecGrid(f'node-{i}', dim=32, backup_count=1).start() for i in range(3)]

np.random.seed(42)
for i in range(300):
    vec = np.random.randn(32).astype(np.float32)
    nodes[0].put(f'vec-{i}', vec)

total_primary = sum(n.local_size() for n in nodes)
total_backup = sum(n.local_backup_size() for n in nodes)

for n in nodes:
    s = n.stats()
    print(f'{n.node_id}: {s[\"total_primary_vectors\"]} primary, {s[\"total_backup_vectors\"]} backup')

print(f'Total primary: {total_primary}')
print(f'Total backup:  {total_backup}')
print(f'Backup ratio:  {total_backup/total_primary:.2f}x')

assert total_primary == 300, f'FAIL: expected 300 primary, got {total_primary}'
assert total_backup == 300, f'FAIL: expected 300 backup, got {total_backup}'

for n in nodes:
    n.stop()

print()
print('✓ TEST 3 PASSED: 1:1 backup ratio — every vector has a backup')
"
```

**Expected:** 300 primary, 300 backup. Backup ratio 1.00x.

---

## Test 4: Node Failure + Backup Promotion (Zero Data Loss)

**What it tests:** Kill a node, backups promote to primary, no data lost.

```bash
PYTHONPATH=. python3 -c "
import numpy as np
from vecgrid import VecGrid, InProcessTransport

InProcessTransport.reset()
nodes = [VecGrid(f'node-{i}', dim=32, backup_count=1).start() for i in range(3)]

np.random.seed(42)
for i in range(400):
    vec = np.random.randn(32).astype(np.float32)
    nodes[0].put(f'vec-{i}', vec)

size_before = nodes[0].cluster_size()
print(f'Before failure: {size_before} vectors')

# Record search results before failure
query = np.random.randn(32).astype(np.float32)
results_before = [r.vector_id for r in nodes[0].search(query, k=10)]

# KILL node-1
killed_primary = nodes[1].local_size()
print(f'Killing node-1 ({killed_primary} primary vectors)...')
nodes[1].stop()

# Check survivors
size_after = nodes[0].cluster_size()
print(f'After failure: {size_after} vectors')

results_after = [r.vector_id for r in nodes[0].search(query, k=10)]
print(f'Search match: {results_before == results_after}')

assert size_after == size_before, f'FAIL: lost {size_before - size_after} vectors'
assert results_before == results_after, 'FAIL: search results changed after failure'

nodes[0].stop()
nodes[2].stop()

print()
print('✓ TEST 4 PASSED: Zero data loss on node failure — backups promoted')
"
```

**Expected:** Cluster size stays at 400 after killing node-1. Search results unchanged.

---

## Test 5: Smart Routing

**What it tests:** Insert/get from any node — requests auto-route to the correct partition owner.

```bash
PYTHONPATH=. python3 -c "
import numpy as np
from vecgrid import VecGrid, InProcessTransport

InProcessTransport.reset()
nodes = [VecGrid(f'node-{i}', dim=32).start() for i in range(3)]

np.random.seed(42)
vec = np.random.randn(32).astype(np.float32)

# Insert via node-0
nodes[0].put('target', vec, {'source': 'node-0'})

# Get from every node — should all find it
for n in nodes:
    result = n.get('target')
    assert result is not None, f'FAIL: {n.node_id} cannot find target'
    assert np.allclose(result[0], vec), f'FAIL: vector mismatch on {n.node_id}'
    print(f'{n.node_id}.get(\"target\") = found, meta={result[1]}')

# Delete from node-2 (may not own the partition)
assert nodes[2].delete('target') == True
assert nodes[0].get('target') is None
print(f'Deleted via node-2, confirmed gone via node-0')

for n in nodes:
    n.stop()

print()
print('✓ TEST 5 PASSED: Smart routing works for get, put, and delete')
"
```

**Expected:** All three nodes can get/delete the vector regardless of which node owns the partition.

---

## Test 6: Safe Partition Migration (Node Join)

**What it tests:** Add a node to a running cluster — data migrates without loss.

```bash
PYTHONPATH=. python3 -c "
import numpy as np
from vecgrid import VecGrid, InProcessTransport

InProcessTransport.reset()
nodes = [VecGrid(f'node-{i}', dim=32, backup_count=1).start() for i in range(2)]

np.random.seed(42)
for i in range(500):
    vec = np.random.randn(32).astype(np.float32)
    nodes[0].put(f'vec-{i}', vec)

size_before = nodes[0].cluster_size()
print(f'2-node cluster: {size_before} vectors')

# Add node-2
print('Adding node-2...')
n2 = VecGrid('node-2', dim=32, backup_count=1).start()
nodes.append(n2)

size_after = nodes[0].cluster_size()
print(f'3-node cluster: {size_after} vectors')
for n in nodes:
    print(f'  {n.node_id}: {n.local_size()} primary, {n.local_backup_size()} backup')

assert size_after == size_before, f'FAIL: lost {size_before - size_after} vectors during migration'

# Add node-3
print('Adding node-3...')
n3 = VecGrid('node-3', dim=32, backup_count=1).start()
nodes.append(n3)

size_final = nodes[0].cluster_size()
print(f'4-node cluster: {size_final} vectors')

assert size_final == 500, f'FAIL: expected 500, got {size_final}'

for n in nodes:
    n.stop()

print()
print('✓ TEST 6 PASSED: Zero data loss across 2 topology changes')
"
```

**Expected:** 500 vectors preserved as cluster grows from 2 → 3 → 4 nodes.

---

## Test 7: Persistence — Survive Restart

**What it tests:** Insert data, shutdown, restart, verify all data recovered from disk.

```bash
PYTHONPATH=. python3 -c "
import numpy as np, shutil
from vecgrid import VecGrid, InProcessTransport

DATA_DIR = '/tmp/vecgrid_test7'
shutil.rmtree(DATA_DIR, ignore_errors=True)

InProcessTransport.reset()
dim = 32
np.random.seed(42)

# Phase 1: Insert and shutdown
print('Phase 1: Insert 300 vectors, then shutdown...')
grid = VecGrid('node-0', dim=dim, data_dir=f'{DATA_DIR}/node-0', snapshot_interval=50)
grid.start()

vectors = {}
for i in range(300):
    vec = np.random.randn(dim).astype(np.float32)
    vectors[f'vec-{i}'] = vec
    grid.put(f'vec-{i}', vec, {'i': i})

query = np.random.randn(dim).astype(np.float32)
results_before = [r.vector_id for r in grid.search(query, k=10)]
print(f'  Size: {grid.local_size()}')
print(f'  Search: {results_before[:5]}...')
grid.stop()
print('  Shutdown complete.')

# Phase 2: Restart and verify
print()
print('Phase 2: Restart and verify recovery...')
InProcessTransport.reset()
grid2 = VecGrid('node-0', dim=dim, data_dir=f'{DATA_DIR}/node-0', snapshot_interval=50)
grid2.start()

print(f'  Recovered: {grid2.local_size()} vectors')
results_after = [r.vector_id for r in grid2.search(query, k=10)]
print(f'  Search: {results_after[:5]}...')

assert grid2.local_size() == 300, f'FAIL: recovered {grid2.local_size()}, expected 300'
assert results_before == results_after, 'FAIL: search results changed after restart'

# Spot check: verify actual vector bytes match
for vid in ['vec-0', 'vec-100', 'vec-299']:
    recovered, meta = grid2.get(vid)
    assert np.allclose(recovered, vectors[vid], atol=1e-6), f'FAIL: vector {vid} corrupted'
    print(f'  Spot check {vid}: OK')

grid2.stop()
shutil.rmtree(DATA_DIR, ignore_errors=True)

print()
print('✓ TEST 7 PASSED: All data recovered after restart')
"
```

**Expected:** 300 vectors recovered, search results identical, vectors bit-for-bit correct.

---

## Test 8: WAL Replay After Crash

**What it tests:** Crash (no graceful shutdown) — WAL replays un-snapshotted writes.

```bash
PYTHONPATH=. python3 -c "
import numpy as np, shutil
from vecgrid import VecGrid, InProcessTransport

DATA_DIR = '/tmp/vecgrid_test8'
shutil.rmtree(DATA_DIR, ignore_errors=True)

InProcessTransport.reset()
dim = 16
np.random.seed(99)

grid = VecGrid('node-0', dim=dim, data_dir=f'{DATA_DIR}/node-0', snapshot_interval=50)
grid.start()

# Insert 80 (triggers snapshot at 50)
for i in range(80):
    grid.put(f'vec-{i}', np.random.randn(dim).astype(np.float32))

# Insert 20 more (WAL only, NOT snapshotted)
extra = {}
for i in range(80, 100):
    vec = np.random.randn(dim).astype(np.float32)
    extra[f'vec-{i}'] = vec
    grid.put(f'vec-{i}', vec)

print(f'Total: {grid.local_size()} vectors (20 in WAL only)')

# SIMULATE CRASH — close persistence without clean shutdown
grid._node._persistence.close()
InProcessTransport.reset()

# Recover
print('Simulated crash. Recovering...')
grid2 = VecGrid('node-0', dim=dim, data_dir=f'{DATA_DIR}/node-0', snapshot_interval=50)
grid2.start()

print(f'Recovered: {grid2.local_size()} vectors')
assert grid2.local_size() == 100, f'FAIL: recovered {grid2.local_size()}, expected 100'

# Verify WAL-only vectors are intact
for vid, original in extra.items():
    result = grid2.get(vid)
    assert result is not None, f'FAIL: {vid} not recovered'
    assert np.allclose(result[0], original, atol=1e-6), f'FAIL: {vid} corrupted'

print(f'All 20 WAL-only vectors verified')

grid2.stop()
shutil.rmtree(DATA_DIR, ignore_errors=True)

print()
print('✓ TEST 8 PASSED: WAL replay recovered un-snapshotted writes after crash')
"
```

**Expected:** All 100 vectors recovered including the 20 that were only in the WAL.

---

## Test 9: TCP Transport + Seed Node Discovery

**What it tests:** Real TCP sockets, seed-based auto-discovery, cross-node operations.

```bash
PYTHONPATH=. python3 -c "
import time, numpy as np, shutil
from vecgrid import VecGrid

DATA_DIR = '/tmp/vecgrid_test9'
shutil.rmtree(DATA_DIR, ignore_errors=True)

dim = 32
np.random.seed(42)

# Start 3 nodes on different ports, node-2 and node-3 discover via node-1
print('Starting 3-node TCP cluster with seed discovery...')

n1 = VecGrid('node-1', dim=dim, transport='tcp', port=6701,
             discovery='seed', seeds=[], backup_count=1,
             data_dir=f'{DATA_DIR}/node-1')
n1.start()

n2 = VecGrid('node-2', dim=dim, transport='tcp', port=6702,
             discovery='seed', seeds=['127.0.0.1:6701'], backup_count=1,
             data_dir=f'{DATA_DIR}/node-2')
n2.start()
time.sleep(0.5)

n3 = VecGrid('node-3', dim=dim, transport='tcp', port=6703,
             discovery='seed', seeds=['127.0.0.1:6701'], backup_count=1,
             data_dir=f'{DATA_DIR}/node-3')
n3.start()
time.sleep(0.5)

# Verify discovery
nodes_seen = sorted(n1._node._cluster_nodes)
print(f'Cluster members: {nodes_seen}')
assert len(nodes_seen) == 3, f'FAIL: expected 3 nodes, got {len(nodes_seen)}'

# Insert via node-1
print()
print('Inserting 200 vectors via node-1...')
for i in range(200):
    vec = np.random.randn(dim).astype(np.float32)
    n1.put(f'vec-{i}', vec, {'i': i})

print(f'  node-1: {n1.local_size()} primary, {n1.local_backup_size()} backup')
print(f'  node-2: {n2.local_size()} primary, {n2.local_backup_size()} backup')
print(f'  node-3: {n3.local_size()} primary, {n3.local_backup_size()} backup')

total = n1.cluster_size()
print(f'  Cluster total: {total}')
assert total == 200, f'FAIL: expected 200, got {total}'

# Search from node-3
query = np.random.randn(dim).astype(np.float32)
results = n3.search(query, k=5)
print(f'\nSearch from node-3: {len(results)} results')
for r in results:
    print(f'  {r.vector_id} dist={r.distance:.4f} from {r.source_node}')
assert len(results) == 5, 'FAIL: wrong result count'

# Smart routing: get from node-2
result = n2.get('vec-0')
assert result is not None, 'FAIL: smart routing failed'
print(f'\nSmart route: node-2.get(\"vec-0\") = found')

# Kill node-2, verify no data loss
print(f'\nKilling node-2...')
n2.stop()
time.sleep(0.5)

remaining = n1.cluster_size()
print(f'Remaining cluster size: {remaining}')
assert remaining == 200, f'FAIL: lost {200 - remaining} vectors'

n1.stop()
n3.stop()
shutil.rmtree(DATA_DIR, ignore_errors=True)

print()
print('✓ TEST 9 PASSED: TCP transport + seed discovery + failover all work')
"
```

**Expected:** 3 nodes discover each other, 200 vectors distributed, search works across TCP, zero data loss on node failure.

---

## Test 10: Multi-Machine TCP Discovery (2 Terminals)

**What it tests:** Real distributed setup across 2 separate Python processes.

**Terminal 1 (Machine A or Terminal A):**

```bash
PYTHONPATH=. python3 -c "
import time, numpy as np
from vecgrid import VecGrid

print('=== Node 1 (seed node) ===')
grid = VecGrid('node-1', dim=32, transport='tcp', port=5701,
               discovery='multicast', backup_count=1,
               data_dir='/tmp/vecgrid_multi/node-1')
grid.start()
print(f'Listening on port 5701')
print(f'Cluster: {sorted(grid._node._cluster_nodes)}')

# Insert some data
np.random.seed(42)
for i in range(100):
    grid.put(f'vec-{i}', np.random.randn(32).astype(np.float32), {'i': i})
print(f'Inserted 100 vectors')

# Wait for other node to join
print('Waiting for node-2 to join (start Terminal 2 now)...')
for _ in range(60):
    time.sleep(1)
    members = sorted(grid._node._cluster_nodes)
    if len(members) > 1:
        print(f'Node-2 joined! Cluster: {members}')
        break
else:
    print('Timeout waiting for node-2')

# Show final state
time.sleep(2)
print(f'Cluster size: {grid.cluster_size()}')
print(f'Local: {grid.local_size()} primary, {grid.local_backup_size()} backup')

# Search
query = np.random.randn(32).astype(np.float32)
results = grid.search(query, k=5)
print(f'Search: {[r.vector_id for r in results]}')

input('Press Enter to shutdown...')
grid.stop()
"
```

**Terminal 2 (Machine B or Terminal B):**

```bash
PYTHONPATH=. python3 -c "
import time, numpy as np
from vecgrid import VecGrid

print('=== Node 2 (joins via discovery) ===')
grid = VecGrid('node-2', dim=32, transport='tcp', port=5702,
               discovery='multicast', backup_count=1,
               data_dir='/tmp/vecgrid_multi/node-2')
grid.start()
print(f'Listening on port 5702')

# Wait for discovery
time.sleep(3)
print(f'Cluster: {sorted(grid._node._cluster_nodes)}')

# Insert more data
np.random.seed(99)
for i in range(100, 200):
    grid.put(f'vec-{i}', np.random.randn(32).astype(np.float32), {'i': i})
print(f'Inserted 100 more vectors')

time.sleep(1)
print(f'Cluster size: {grid.cluster_size()}')
print(f'Local: {grid.local_size()} primary, {grid.local_backup_size()} backup')

# Cross-node search
query = np.random.randn(32).astype(np.float32)
results = grid.search(query, k=5)
print(f'Search: {[(r.vector_id, r.source_node) for r in results]}')

input('Press Enter to shutdown...')
grid.stop()
"
```

**Expected:** Both terminals show 2-node cluster, cluster size = 200, search returns results from both nodes.

**Note:** If multicast doesn't work on your network, replace `discovery='multicast'` with `discovery='seed', seeds=['<IP_OF_NODE_1>:5701']` in Terminal 2.

---

## Test 11: Automated Test Suite

**What it tests:** All 18 unit tests in one go.

```bash
PYTHONPATH=. python3 tests/test_vecgrid.py
```

**Expected:**

```
18/18 tests passed
✓ ALL TESTS PASS
```

---

## Quick Reference: What Each Feature Does

| Feature | Test | How to verify |
|---------|------|---------------|
| HNSW search | Test 1 | Results sorted by distance |
| Multi-node partitioning | Test 2 | All nodes return identical results |
| Sync backup replication | Test 3 | Backup count = primary count |
| Backup promotion | Test 4 | Cluster size unchanged after kill |
| Smart routing | Test 5 | get/delete from any node works |
| Safe migration | Test 6 | No data loss on node join |
| Persistence (clean) | Test 7 | All data survives restart |
| WAL crash recovery | Test 8 | Un-snapshotted writes recovered |
| TCP + discovery | Test 9 | 3 nodes auto-discover over TCP |
| Multi-process | Test 10 | 2 separate terminals form cluster |
| Full test suite | Test 11 | 18/18 pass |
| Benchmark (10K) | Test 12 | Recall, latency, QPS numbers |
| Benchmark (100K) | Test 13 | Requires hnswlib installed |

---

## Test 12: Benchmark (1K / 5K / 10K vectors)

**What it tests:** Recall@10, search latency, insert throughput, failover correctness — real numbers.

```bash
PYTHONPATH=. python3 benchmark.py
```

**Expected (numpy backend):**

```
 Vectors   Dim   Recall   Avg ms   P95 ms     QPS     Ins/s  Loss
----------------------------------------------------------------------
   1,000   128    1.000   ~10ms    ~60ms     ~100     ~10K      0
   5,000   128    1.000   ~40ms    ~80ms      ~24     ~4K       0
  10,000   128    1.000   ~90ms   ~115ms      ~11     ~600      0
```

All recall should be 1.000, all data loss should be 0.

---

## Test 13: Large Scale Benchmark (100K vectors — requires hnswlib)

**What it tests:** Recall at real scale where HNSW approximation actually matters.

**Prerequisites:** `pip install hnswlib`

```bash
PYTHONPATH=. python3 -c "
import time, numpy as np
from vecgrid import VecGrid, InProcessTransport
from vecgrid.hnsw import get_backend_name

assert get_backend_name() == 'hnswlib', 'Install hnswlib first: pip install hnswlib'

InProcessTransport.reset()
np.random.seed(42)
n_vec, dim, k = 100000, 128, 10

print(f'Backend: {get_backend_name()}')
print(f'Inserting {n_vec:,} vectors...')

nodes = [VecGrid(f'n-{i}', dim=dim, num_partitions=271, backup_count=1,
                 hnsw_config={'ef_construction': 200, 'ef_search': 100}).start()
         for i in range(3)]

vecs = np.random.randn(n_vec, dim).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

t0 = time.time()
for i in range(n_vec):
    nodes[i % 3].put(f'v-{i}', vecs[i])
print(f'Insert: {time.time()-t0:.1f}s ({n_vec/(time.time()-t0):,.0f} vec/s)')
print(f'Primary: {sum(n.local_size() for n in nodes):,}')

# Ground truth
n_q = 100
qs = np.random.randn(n_q, dim).astype(np.float32)
qs /= np.linalg.norm(qs, axis=1, keepdims=True)

print(f'Computing ground truth ({n_q} queries)...')
gt = []
for q in qs:
    d = 1.0 - vecs @ q
    gt.append({f'v-{i}' for i in np.argsort(d)[:k]})

print(f'Searching...')
recall_sum = 0
lats = []
for i, q in enumerate(qs):
    t0 = time.time()
    res = nodes[0].search(q, k=k)
    lats.append((time.time()-t0)*1000)
    recall_sum += len({r.vector_id for r in res} & gt[i]) / k

r = recall_sum / n_q
print(f'Recall@{k}: {r:.4f} ({r*100:.1f}%)')
print(f'Latency: avg={np.mean(lats):.1f}ms p50={np.median(lats):.1f}ms p95={np.percentile(lats,95):.1f}ms')
print(f'QPS: {n_q/sum(l/1000 for l in lats):.1f}')

# Failover
nodes[1].stop()
print(f'After failover: {nodes[0].cluster_size():,} vectors (should be {n_vec:,})')

for n in [nodes[0], nodes[2]]: n.stop()
print('Done')
"
```

**Expected (hnswlib backend):**

- Recall@10: 0.95-0.99 (this is where approximation genuinely matters)
- Latency: avg 5-15ms
- QPS: 70-200
- Insert: 30,000-50,000 vec/s
- Failover: 100,000 vectors preserved
