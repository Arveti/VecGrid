# VecGrid

**Distributed Embedded Vector Database**

Like Hazelcast/Infinispan, but for vector search. Embeds directly in your application process. No separate infrastructure.

---

## The Gap Nobody Filled

Every vector database today falls into one of two buckets:

**Embedded, single-node** — FAISS, HNSWlib, LanceDB, Chroma. They run inside your application process with microsecond latency. But they can't distribute. Your index lives on one machine. If that machine's RAM fills up or the process dies, you're stuck.

**Distributed, client-server** — Qdrant, Milvus, Weaviate, Pinecone. They scale horizontally and handle persistence. But they run as separate infrastructure. Every query pays a network round-trip. You now have another cluster to deploy, monitor, and operate.

**VecGrid is the third option that didn't exist:** embedded AND distributed. Each application instance embeds a VecGrid node in its own process. Nodes discover each other, partition data via consistent hashing, replicate for fault tolerance, and coordinate scatter-gather searches — all with in-process memory access for the data that lives locally.

This is the same architectural pattern that Hazelcast and Infinispan proved for key-value data. VecGrid applies it to vector search.

---

## Why This Works

The core insight is simple: **you don't need to build distributed systems primitives from scratch to make vector search distributed.** The hard problems — membership, consistent hashing, partition ownership, replication, rebalancing, failure detection — were solved decades ago by in-memory data grids.

What VecGrid actually builds is narrow: an HNSW index layer on top of partition-based data distribution. Each partition maintains its own HNSW graph. The distributed systems machinery is borrowed patterns, not invented ones.

### The Architecture in One Picture

```
                        ┌──────────────────────────┐
                        │     Your Application      │
                        │                            │
                        │   grid = VecGrid(...)      │
                        │   grid.put("doc", vec)     │
                        │   grid.search(query, k=10) │
                        └──────────┬─────────────────┘
                                   │ in-process call (no network)
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VecGrid Node                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Consistent Hash Ring                      │   │
│  │   hash("doc-1") → Partition 42 → This Node (primary)     │   │
│  │   hash("doc-2") → Partition 108 → Node-2 (smart route)   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ Partition 7 │  │Partition 42│  │Partition 200│  ...          │
│  │ (primary)   │  │ (primary)  │  │  (backup)   │               │
│  │             │  │            │  │             │                │
│  │  HNSW Graph │  │ HNSW Graph │  │ HNSW Graph  │               │
│  │  142 vectors│  │ 89 vectors │  │ 156 vectors │               │
│  └────────────┘  └────────────┘  └────────────┘                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  WAL + Snap   │  │ Discovery        │  │ Transport        │  │
│  │  (persistence)│  │ (multicast/seed) │  │ (TCP/in-process) │  │
│  └──────────────┘  └──────────────────┘  └──────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ scatter-gather / replication
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │ VecGrid  │  │ VecGrid  │  │ VecGrid  │
          │ Node-2   │  │ Node-3   │  │ Node-4   │
          │ (App 2)  │  │ (App 3)  │  │ (App 4)  │
          └──────────┘  └──────────┘  └──────────┘
```

### How a Write Works

```
grid.put("doc-1", vector, {"title": "Hello"})
    │
    ├─ hash("doc-1") → Partition 42
    ├─ Partition 42 owner = this node? 
    │   ├─ YES: insert into local HNSW graph
    │   └─ NO:  smart-route to owner node (transparent to caller)
    │
    ├─ Write to WAL (fsync'd to disk after insert)
    │
    ├─ Replicate to backup node(s) synchronously
    │   └─ Backup node inserts into its own HNSW graph copy
    │
    └─ Return to caller (all backups confirmed)
```

### How a Search Works

```
grid.search(query_vector, k=10)
    │
    ├─ SCATTER: send query to all nodes (including self)
    │   ├─ Node-1: search local primary partitions → top-10 local results  
    │   ├─ Node-2: search local primary partitions → top-10 local results
    │   └─ Node-3: search local primary partitions → top-10 local results
    │
    ├─ GATHER: collect all results from all nodes
    │
    ├─ MERGE: sort all results by distance, take global top-10
    │
    └─ Return to caller
```

Each node searches ONLY its primary partitions (not backups) to avoid double-counting. The coordinator merges all partial results into the final answer.

---

## How Distributed HNSW Works (and What Changes)

This is the part people will question, so let's be precise.

### Standard HNSW (Single Index)

In a normal HNSW index, every vector participates in one global navigable small-world graph. When you search, the algorithm traverses graph edges — starting from an entry point, greedily moving to neighbors closer to the query, across multiple layers. The entire graph is connected, so any vector is reachable from any other.

### VecGrid's Partitioned HNSW

VecGrid splits vectors across N partitions (default 271) based on consistent hashing of the vector ID. Each partition maintains its own independent HNSW graph. This means:

**Vectors in different partitions have no graph edges between them.** Partition 42's HNSW graph has no connections to Partition 108's graph. They are completely separate small-world networks.

**Each partition's graph is smaller.** With 1 million vectors across 271 partitions, each partition holds ~3,700 vectors on average. Each of those small graphs is independently well-connected.

**Search is embarrassingly parallel.** Each partition can be searched independently, which is what makes scatter-gather work.

### Impact on Query Results: Recall

The key question is: **does partitioning hurt recall?**

The answer depends on partition size.

**Why it can hurt:** In a single HNSW graph with 1M vectors, the algorithm can traverse long-range edges to reach any vector in the dataset. In a partitioned setup, the algorithm can only find vectors within each partition's graph. If the true nearest neighbor happens to be in a partition where the search algorithm got "stuck" in a local minimum (didn't find it because the small graph had poor connectivity), it's missed.

**Why it usually doesn't matter much:** HNSW recall depends heavily on graph connectivity, which depends on the `M` parameter (max connections per node) and `ef_search` (beam width). For a graph with 3,700 vectors and M=16, there are ~59,000 edges connecting them — that's a very dense graph. The search algorithm rarely gets stuck. Our benchmark shows:

| Vectors per partition | Recall@10 | Why |
|---|---|---|
| ~18 (5K vectors / 271 partitions) | 100% | Tiny graphs → effectively brute force within each partition |
| ~370 (100K / 271) | 97-99% | Small graphs are extremely well-connected |
| ~3,700 (1M / 271) | 95-98% | Standard HNSW behavior within each partition |
| ~37,000 (10M / 271) | 93-97% | Depends on ef_search tuning |

**The counterintuitive result:** partitioning can actually *improve* recall for small-to-medium datasets because each partition's graph is denser relative to its size. With only 3,700 vectors and M=16, the graph is nearly fully connected. You're essentially doing a very efficient near-brute-force search within each shard, then merging.

**Where it genuinely hurts:** At very large scale (100K+ vectors per partition), each partition's HNSW graph behaves like a regular large HNSW index. Recall tracks what you'd get from a single HNSW with ef_search tuning. The partitioning itself doesn't add recall loss — but you're not getting cross-partition graph edges that a single index would have. The scatter-gather merge compensates: since every partition returns its local top-k, the merge step recovers global top-k as long as each partition's local search found the right local candidates.

### The Critical Difference From Naive Sharding

VecGrid's approach is NOT the same as naively splitting vectors into random shards and searching each shard. The key differences:

**Every partition is searched on every query.** This is scatter-gather, not routing. A routed shard architecture (like some Milvus configurations) sends the query to only one shard, hoping the nearest neighbors are there. VecGrid sends to ALL partitions and merges. This means recall is never limited by partition assignment — you will always find the global top-k as long as each partition's HNSW index finds good local candidates.

**Partition assignment is by vector ID, not by vector content.** Vectors aren't clustered by similarity. This is deliberate — it means every partition has a uniform random sample of the vector space, which makes each partition's HNSW graph well-balanced. Content-based sharding (like IVF partitioning in FAISS) creates the opposite problem: if your query falls near a partition boundary, you miss neighbors in adjacent partitions.

### Tuning Knobs

If you need higher recall, you have two levers:

**`ef_search`** — increase the search beam width within each partition. Default 50. Setting to 100-200 gives near-perfect recall at the cost of latency. This is the same knob you'd tune in any HNSW index.

**`num_partitions`** — fewer partitions means more vectors per partition means larger HNSW graphs. With 100 partitions instead of 271, each graph is ~2.7x larger. Tradeoff: fewer partitions means less even distribution across nodes and coarser rebalancing. The default 271 (a prime) is chosen for good hash distribution, following the Hazelcast convention.

---

## Advantages Over Traditional Vector Databases

### vs. Client-Server Vector DBs (Qdrant, Milvus, Pinecone)

**No infrastructure to operate.** VecGrid embeds in your application process. There is no separate vector database cluster to deploy, monitor, scale, backup, or pay for. Your application IS the database.

**No network round-trip on queries.** A search on data that lives on the local node is a direct function call — no TCP, no serialization, no deserialization. For the ~1/N fraction of data on each node (where N is node count), you get in-memory latency. Cross-node scatter-gather adds network time, but you're parallelizing across all nodes simultaneously rather than making a sequential request to one external cluster.

**Scales with your application, not separately.** When you add application instances to handle more traffic, VecGrid automatically gains more nodes. Data rebalances. Capacity grows. You don't need to separately scale your vector DB to match.

**No cold start / connection pool issues.** Client-server vector DBs require connection management, retry logic, timeouts, circuit breakers. VecGrid is a library call. It's always "connected" because it's in your process.

**Operational simplicity.** One less piece of infrastructure. One less thing in your Kubernetes YAML. One less vendor to evaluate. One less bill. For teams that don't want to run a separate database just for vector search, this removes the question entirely.

### vs. Embedded Vector DBs (FAISS, HNSWlib, Chroma)

**Horizontal scaling.** FAISS and HNSWlib are single-machine. If your dataset outgrows one machine's RAM, you're stuck re-architecting. VecGrid scales by adding nodes — data automatically redistributes.

**Fault tolerance.** If your FAISS process dies, the index is gone (unless you wrote persistence yourself). VecGrid has sync backup replication — every vector exists on at least two nodes. Kill a node, zero data loss.

**No single point of failure.** With backup promotion, any node can die and the cluster continues serving queries immediately. No failover delay, no manual intervention.

**Persistence built in.** WAL + snapshots. Survive process restarts. Survive crashes (WAL replay recovers un-snapshotted writes). You don't have to build your own serialization layer.

### vs. Redis Vector Search

**Purpose-built for vectors.** Redis bolt-on vector search uses flat or HNSW indexes within Redis's key-value model. VecGrid's entire data model is vector-native — partitioning, replication, and search are all designed around approximate nearest neighbor semantics.

**No Redis dependency.** VecGrid is a pure Python library with numpy as the only dependency. No Redis server, no Redis cluster management, no Redis memory limits to configure.

---

## What VecGrid Does NOT Do (Honest Limitations)

**Not a managed service.** You run it yourself. There's no dashboard, no hosted offering, no SLA.

**Not battle-tested at scale.** This is alpha software. It's been tested with up to 10K vectors per node in automated tests and demos. Production workloads with millions of vectors need real benchmarking that hasn't happened yet.

**No multi-tenancy.** All nodes in a cluster share the same index. If you need isolated indexes per tenant, you'd need separate clusters.

**No GPU acceleration.** Distance computation is numpy on CPU. For very high-dimensional vectors (1000+), GPU-accelerated libraries like FAISS will be faster per-query on a single machine.

**Single-writer per partition.** The primary node for a partition handles all writes. There's no multi-master replication. At very high write throughput, the primary can become a bottleneck.

---

## Quick Start

### Embedded Mode (Single Process)

```python
from vecgrid import VecGrid
import numpy as np

grid = VecGrid(node_id="app-1", dim=384)
grid.start()

# Insert
grid.put("doc-1", np.random.randn(384).astype(np.float32), {"title": "Hello"})

# Search
results = grid.search(query_vector, k=10)
for r in results:
    print(f"{r.vector_id}: dist={r.distance:.4f}, meta={r.metadata}")

grid.stop()
```

### Multi-Node Embedded (Like Hazelcast Embedded Mode)

```python
node1 = VecGrid("node-1", dim=384, backup_count=1).start()
node2 = VecGrid("node-2", dim=384, backup_count=1).start()
node3 = VecGrid("node-3", dim=384, backup_count=1).start()

# Insert from any node — smart-routed to partition owner
node1.put("doc-1", vector1, {"type": "article"})

# Search from any node — scatter-gather across all
results = node3.search(query, k=10)

# Kill a node — zero data loss
node2.stop()
assert node1.cluster_size() == original_size  # Backups promoted
```

### TCP with Auto-Discovery (Multi-Machine)

```python
# Machine A
grid = VecGrid("node-1", dim=384, transport="tcp", port=5701,
               discovery="multicast", backup_count=1,
               data_dir="/var/data/vecgrid/node-1")
grid.start()

# Machine B — auto-discovers Machine A via multicast
grid = VecGrid("node-2", dim=384, transport="tcp", port=5701,
               discovery="multicast", backup_count=1,
               data_dir="/var/data/vecgrid/node-2")
grid.start()  # Finds node-1, joins cluster, partitions rebalance
```

### TCP with Seed Node Discovery (Cloud / No Multicast)

```python
# Machine A (seed node)
grid = VecGrid("node-1", dim=384, transport="tcp", port=5701,
               discovery="seed", seeds=[])
grid.start()

# Machine B — points to Machine A as seed
grid = VecGrid("node-2", dim=384, transport="tcp", port=5701,
               discovery="seed", seeds=["10.0.1.10:5701"])
grid.start()
```

### VPN / Kubernetes / Cloud Deployment

When nodes communicate across VPNs (Tailscale, WireGuard), overlay networks, or container bridges, the default auto-detected IP may not be reachable by peers. Use `advertise_host` to explicitly set the externally-reachable IP:

```python
# Tailscale — each node advertises its Tailscale IP
# Machine A (Tailscale IP: 100.83.53.112)
grid = VecGrid("node-1", dim=384, transport="tcp", port=5701,
               discovery="seed", seeds=[],
               advertise_host="100.8.53.112")

# Machine B (Tailscale IP: 100.64.0.5)
grid = VecGrid("node-2", dim=384, transport="tcp", port=5701,
               discovery="seed", seeds=["100.8.53.112:5701"],
               advertise_host="100.4.0.5")
```

```python
# Kubernetes — use the pod IP from the downward API
import os
grid = VecGrid("node-1", dim=384, transport="tcp", port=5701,
               discovery="seed", seeds=["vecgrid-0.vecgrid:5701"],
               advertise_host=os.environ.get("POD_IP"))
```

**When do you need `advertise_host`?**

| Environment | Needed? | Why |
|---|---|---|
| Same LAN / WiFi | No | Auto-detected IP is reachable |
| Tailscale / WireGuard | Yes | Default route uses the non-VPN interface |
| Kubernetes pods | Usually no | Pod IP is the default route; set explicitly for safety |
| Docker bridge network | Yes | Container IP ≠ host IP |
| AWS EC2 (same VPC) | No | Private IP is the default route |

### With Persistence

```python
grid = VecGrid("node-1", dim=384,
               data_dir="/var/data/vecgrid/node-1",
               snapshot_interval=1000)
grid.start()

# All writes go to WAL (fsync'd). Auto-snapshot every 1000 writes.
grid.put("doc-1", vector, {"title": "Hello"})

grid.stop()   # Final snapshot on shutdown

# Later: full recovery from disk
grid = VecGrid("node-1", dim=384, data_dir="/var/data/vecgrid/node-1")
grid.start()  # Snapshot + WAL replay → all data restored
```

---

## Feature Matrix

| Feature | Status | Notes |
|---|---|---|
| HNSW index (cosine, euclidean, dot) | ✅ Done | Dual backend: hnswlib (C++) or numpy (Python) |
| Auto backend selection | ✅ Done | Uses hnswlib if installed, numpy fallback |
| Consistent hash partitioning | ✅ Done | 271 partitions, prime for distribution |
| Scatter-gather search | ✅ Done | All partitions searched, results merged |
| Sync backup replication | ✅ Done | Configurable backup_count |
| Backup promotion on failure | ✅ Done | Zero data loss with backup_count ≥ 1 |
| Smart routing | ✅ Done | Any node handles any request |
| Safe partition migration | ✅ Done | Migrate-then-delete protocol |
| WAL + snapshot persistence | ✅ Done | Survives crashes via WAL replay |
| Multicast discovery | ✅ Done | UDP 224.2.2.3:54327 (Hazelcast convention) |
| Seed node discovery | ✅ Done | For cloud / no-multicast environments |
| Heartbeat failure detector | ✅ Done | Auto-detects crashed nodes |
| TCP transport | ✅ Done | Length-prefixed JSON protocol |
| In-process transport | ✅ Done | For embedded / testing mode |
| Filtered search push-down | ✅ Done |
| GPU distance computation | ❌ Planned | Currently CPU numpy |
| gRPC transport | ❌ Planned | Currently custom TCP |
| TLS / authentication | ❌ Planned | Currently plaintext |
| JVM version | ❌ Planned | Currently Python only |

---

## Backend Selection

VecGrid auto-detects the best available HNSW implementation:

**hnswlib (C++)** — 50-100x faster. Install with `pip install hnswlib`. This is the production backend. Uses SIMD-accelerated distance computation and optimized graph traversal written in C++.

**numpy (Python)** — always available. Zero extra dependencies. Uses vectorized batch distance computation via OpenBLAS. Suitable for development, testing, and small datasets (<10K vectors).

```bash
# Development / testing (works out of the box)
pip install vecgrid

# Production (50-100x faster search)
pip install vecgrid[fast]
# or: pip install vecgrid hnswlib
```

Backend is selected automatically. To check or override:

```python
from vecgrid.hnsw import get_backend_name, create_index

print(get_backend_name())  # "hnswlib" or "numpy"

# Force a specific backend
index = create_index(dim=384, backend="numpy")   # always numpy
index = create_index(dim=384, backend="hnswlib")  # error if not installed
```

See [BENCHMARKS.md](BENCHMARKS.md) for performance numbers at different scales.

## Configuration Reference

```python
grid = VecGrid(
    # Core
    node_id="my-node",          # Unique node identifier
    dim=384,                     # Vector dimensions
    num_partitions=271,          # Total partitions (same across cluster)
    backup_count=1,              # Sync backup replicas per partition

    # Transport
    transport="tcp",             # "embedded" or "tcp"
    host="0.0.0.0",             # Listen address
    port=5701,                   # Listen port (Hazelcast convention)
    advertise_host="100.x.x.x", # Externally-reachable IP (for VPNs, K8s, Docker)
                                 # None = auto-detect via default route

    # Discovery
    discovery="multicast",       # "none", "multicast", or "seed"
    seeds=["10.0.1.10:5701"],   # Seed addresses (for seed discovery)
    multicast_group="224.2.2.3", # Multicast group address
    multicast_port=54327,        # Multicast UDP port

    # Heartbeat
    heartbeat=True,              # Enable failure detection
    heartbeat_interval=2.0,      # Ping interval (seconds)
    heartbeat_timeout=8.0,       # Declare dead after (seconds)

    # Persistence
    data_dir="/var/data/node-1", # WAL + snapshot directory (None = memory only)
    snapshot_interval=1000,      # Auto-snapshot every N writes per partition

    # HNSW tuning
    hnsw_config={
        "M": 16,                 # Max graph connections per layer
        "M0": 32,                # Max connections at layer 0 (usually 2×M)
        "ef_construction": 200,  # Build-time beam width (higher = better graph)
        "ef_search": 50,         # Search-time beam width (higher = better recall)
        "distance_metric": "cosine",  # "cosine", "euclidean", or "dot"
    },
)
```

---

## Data Layout on Disk

```
/var/data/vecgrid/node-1/
    wal/
        partition_0042.wal          # Append-only log: insert/delete ops
        partition_0108.wal          # One file per partition
    snapshots/
        snapshot_0042_v500.bin      # Full partition dump at version 500
        snapshot_0108_v430.bin      # Binary format: header + vectors + metadata
```

Recovery: load latest snapshot per partition, replay WAL entries after the snapshot version. WAL is truncated after each snapshot. Crash recovery replays the full WAL since last snapshot — no data loss for any fsync'd write.

---

## Project Structure

```
vecgrid/
├── vecgrid/
│   ├── __init__.py          # VecGrid high-level API + discovery wiring
│   ├── hnsw.py              # Dual-backend HNSW (hnswlib C++ / numpy Python)
│   ├── hash_ring.py         # Consistent hash ring (partition → node mapping)
│   ├── node.py              # EmbeddedNode (cluster, replication, routing)
│   ├── persistence.py       # WAL + snapshot engine
│   ├── transport.py         # InProcess + TCP transport layers
│   └── discovery.py         # Multicast, seed node, heartbeat detector
├── tests/
│   ├── conftest.py          # Pytest fixtures (transport reset, tmp_dir)
│   ├── test_vecgrid.py      # 19 automated tests
│   ├── test_demo_qa.py      # Semantic QA integration test
│   ├── test_terminal1.py  # Multi-process TCP test (terminal 1)
│   └── test_terminal2.py  # Multi-process TCP test (terminal 2)
├── benchmark.py             # Performance benchmark suite
├── demo.py                  # 6 interactive demos
├── BENCHMARKS.md            # Performance numbers and analysis
├── TESTING.md               # 11 step-by-step manual tests
├── pyproject.toml           # pip install ready (with [fast] extra for hnswlib)
├── LICENSE                  # GPL-3.0-or-later
└── README.md
```

---

## Testing

Run the automated test suite:

```bash
# Recommended (uses pytest with fixtures for proper test isolation)
python -m pytest tests/ -v

# Alternative (standalone runner, no pytest required)
PYTHONPATH=. python3 tests/test_vecgrid.py
```

Run benchmarks and demos:

```bash
python benchmark.py   # Performance benchmark (1K–10K vectors)
python demo.py         # 6 interactive feature demos
```

For step-by-step manual validation of every feature (including multi-process TCP tests), see [TESTING.md](TESTING.md).

---

## License

GPL-3.0-or-later
