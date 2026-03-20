# VecGrid Concepts Guide

> Everything you need to understand VecGrid, explained from scratch.
> No prior distributed systems or ML knowledge required.

---

## Table of Contents

1. [What is VecGrid?](#1-what-is-vecgrid)
2. [How Vectors Work](#2-how-vectors-work)
3. [HNSW — The Search Algorithm](#3-hnsw--the-search-algorithm)
4. [Partitioning — Splitting Data Across Nodes](#4-partitioning--splitting-data-across-nodes)
5. [Cluster — Multiple Nodes Working Together](#5-cluster--multiple-nodes-working-together)
6. [Replication — Keeping Copies Safe](#6-replication--keeping-copies-safe)
7. [Failure Handling — When Things Go Wrong](#7-failure-handling--when-things-go-wrong)
8. [Smart Routing — Ask Anyone, Get the Right Answer](#8-smart-routing--ask-anyone-get-the-right-answer)
9. [Safe Migration — Moving Data Without Losing It](#9-safe-migration--moving-data-without-losing-it)
10. [Scatter-Gather Search — Searching Everywhere](#10-scatter-gather-search--searching-everywhere)
11. [Persistence — Surviving Crashes](#11-persistence--surviving-crashes)
12. [Transport — How Nodes Talk](#12-transport--how-nodes-talk)
13. [Discovery — Finding Other Nodes](#13-discovery--finding-other-nodes)
14. [Putting It All Together](#14-putting-it-all-together)
15. [Glossary](#15-glossary)

---

## 1. What is VecGrid?

### What is a vector?

A **vector** is just a list of numbers. That's it.

```
[0.2, 0.8, 0.1, 0.5]
```

These numbers describe *something*. Think of it like GPS coordinates — two numbers (latitude, longitude) describe a location on Earth. A vector is the same idea, but with more numbers, describing more complex things.

In AI and machine learning, vectors describe the *meaning* of text, images, audio, or anything else. A sentence like "The cat sat on the mat" gets converted into a vector of hundreds of numbers that capture what that sentence *means*.

### What is a vector database?

A **vector database** stores these vectors and lets you search for *similar* ones.

Imagine you have 1 million product descriptions, each converted into a vector. A customer types "comfortable running shoes." You convert that query into a vector too, then ask: *"Which stored vectors are closest to this one?"* The database returns the most similar products.

This is called **similarity search** or **nearest neighbor search**.

### What is "distributed"?

**Distributed** means the work is split across multiple computers (called **nodes**).

Why? Because one computer has limits — limited memory, limited CPU, limited reliability. If that one computer crashes, everything is gone. By spreading data across multiple nodes:

- You can store more data (each node holds a piece)
- You can handle more queries (nodes work in parallel)
- You survive failures (if one node dies, others have copies)

### What is "embedded"?

Most databases run as a separate program — you start a database server, then your application connects to it over the network. **Embedded** means the database lives *inside* your application. No separate server to install, configure, or maintain.

```
Traditional:
  Your App  ──network──>  Database Server

Embedded (VecGrid):
  ┌─────────────────────────┐
  │  Your App               │
  │    └── VecGrid (inside) │
  └─────────────────────────┘
```

Your app and the database share the same process. This means:
- Zero network latency for local data
- No separate infrastructure to manage
- Just `pip install vecgrid` and go

### The Hazelcast inspiration

VecGrid borrows its architecture from **Hazelcast**, a well-known distributed data grid. Hazelcast solved the hard problems of distributed computing — how to split data, keep copies, handle failures, and rebalance when nodes join or leave. VecGrid applies these same battle-tested ideas to vector search.

---

## 2. How Vectors Work

### Numbers that represent meaning

Modern AI models (like sentence-transformers) convert real-world data into vectors:

```
"I love pizza"     → [0.82, 0.14, 0.91, 0.33, ...]   (384 numbers)
"Pizza is great"   → [0.79, 0.18, 0.88, 0.31, ...]   (384 numbers)
"The stock market"  → [0.12, 0.67, 0.05, 0.89, ...]   (384 numbers)
```

Notice: "I love pizza" and "Pizza is great" have similar numbers because they have similar meanings. "The stock market" has very different numbers because it means something different.

The number of values in each vector is called the **dimension** (`dim`). Common dimensions are 128, 384, 768, or 1536, depending on the AI model used.

### Distance = similarity

To find similar vectors, we measure the **distance** between them. Smaller distance = more similar.

VecGrid supports three distance metrics:

**Cosine distance** (default) — Measures the *angle* between two vectors. Ignores magnitude (length), focuses only on direction.

```
Analogy: Two arrows pointing in nearly the same direction are
"similar," even if one arrow is longer than the other.

     ↗  "pizza is great"
    ↗   "I love pizza"        ← small angle = similar
   →    "the stock market"    ← large angle = different
```

**Euclidean distance** — Measures the straight-line distance between two points.

```
Analogy: How far apart are two pins on a map?
Closer pins = more similar.
```

**Dot product** — A mathematical operation that combines both direction and magnitude.

```
Analogy: How much do two vectors "agree" with each other?
Higher dot product = more agreement = more similar.
```

For most text and image applications, **cosine** is the best default.

### Why brute force is slow

The simplest way to find similar vectors: compare your query against *every* stored vector, compute the distance to each one, and return the closest.

This works perfectly for 100 vectors. For 1 million vectors, each with 384 dimensions? You'd compute 1,000,000 distance calculations per query. That's too slow for real-time applications.

We need a smarter approach — that's where HNSW comes in.

---

## 3. HNSW — The Search Algorithm

### The core idea

**HNSW** stands for **Hierarchical Navigable Small World**. It's an algorithm that finds *approximate* nearest neighbors very quickly by building a clever graph structure.

Think of it like a transportation system:

```
Layer 3 (express):    A ─────────────────── D

Layer 2 (fast):       A ──────── C ──────── D

Layer 1 (local):      A ── B ── C ── D ── E

Layer 0 (all stops):  A B C D E F G H I J K L M N O P
```

- **Layer 0** contains *every* vector, connected to nearby neighbors
- **Higher layers** contain fewer vectors but with longer-range connections
- To search: start at the top layer, take big jumps to get close, then drop down for precision

This is like navigating a city:
1. Take the highway to the right neighborhood (top layers)
2. Take local streets to the right block (middle layers)
3. Walk to the exact address (bottom layer)

Instead of checking all 1,000,000 vectors, HNSW typically checks only a few hundred — and still finds the right answer 95-100% of the time.

### Building the graph

When you insert a vector, HNSW:

1. **Assigns a random layer** — Most vectors go to layer 0 only. A few lucky ones get promoted to higher layers (like skip list randomization).
2. **Finds neighbors** — Starting from the top, navigates down to find the closest existing vectors.
3. **Creates connections** — Links the new vector to its nearest neighbors at each layer.

The result is a graph where similar vectors are connected, and you can navigate from any vector to any other through a short chain of hops.

### Key parameters

**M** (max connections per node) — How many neighbors each vector connects to.
- Higher M = better recall but more memory and slower inserts
- Default: 16
- Think: each person in a social network has M friends

**M0** (connections at layer 0) — Usually 2 × M. Layer 0 gets extra connections because it's where the fine-grained search happens.

**ef_construction** (build-time beam width) — How hard to search when inserting a new vector.
- Higher = better graph quality but slower inserts
- Default: 200
- Think: how many candidates to consider when choosing friends

**ef_search** (search-time beam width) — How hard to search when answering queries.
- Higher = better recall but slower search
- Default: 50
- Think: how many paths to explore when looking for someone

### Recall vs speed

**Recall** measures accuracy: if the true 10 nearest neighbors are {A, B, C, D, E, F, G, H, I, J}, and HNSW returns {A, B, C, D, E, F, G, H, I, K}, that's 90% recall (got 9 out of 10 right).

```
                  Recall
  100% |          ___________
       |        /
       |      /
   90% |    /
       |  /
   80% |/
       └──────────────────── Speed (queries/sec)
       slow                  fast

Higher ef_search = more recall, less speed
Lower ef_search  = less recall, more speed
```

At small scales (under 10K vectors), VecGrid achieves 100% recall. At larger scales, you trade a tiny bit of accuracy for massive speed gains.

### Dual backends

VecGrid has two HNSW implementations:

**hnswlib (C++ backend)** — A battle-tested C++ library wrapped for Python. Uses SIMD CPU instructions for blazing-fast distance calculations. 50-100x faster than Python.

```bash
pip install hnswlib    # Enable the fast backend
```

**NumpyHNSWIndex (Pure Python)** — Built from scratch using only NumPy. Always available, no extra install needed. Slower, but great for learning, testing, and small datasets.

VecGrid automatically uses hnswlib if installed, otherwise falls back to the numpy backend. You don't need to change any code.

---

## 4. Partitioning — Splitting Data Across Nodes

### Why partition?

If you have 1 million vectors and 3 nodes, you can't put everything on one node — that defeats the purpose of distributing. **Partitioning** is how you decide which data goes where.

### The hash ring

VecGrid uses **consistent hashing** to assign vectors to partitions. Here's how it works:

Imagine a clock face, but instead of 12 hours, it has 271 positions (0 to 270):

```
              0
          270   1
        269       2
       .           .
       .     271     .
       .  positions  .
        .           .
         135     136
            ...
```

Every vector ID gets **hashed** (converted to a number) and placed on this ring:

```python
partition = hash("my-document-id") % 271
# Result: partition 42
```

This is deterministic — the same ID always maps to the same partition, on any node, without communication.

### Partitions assigned to nodes

Each partition is assigned to a node using simple round-robin:

```
3 nodes, 271 partitions:

  Node 0 gets: partition 0, 3, 6, 9, 12, ...   (~ 90 partitions)
  Node 1 gets: partition 1, 4, 7, 10, 13, ...   (~ 90 partitions)
  Node 2 gets: partition 2, 5, 8, 11, 14, ...   (~ 91 partitions)
```

Every node knows the full mapping, so any node can instantly determine: *"vector X belongs to partition P, which is owned by node N."*

### Why 271?

271 is a **prime number**. Prime numbers distribute data more evenly across nodes because they avoid patterns that could cluster partitions on certain nodes. This is a convention borrowed from Hazelcast.

### Each partition = its own index

Each partition has its own independent HNSW graph. When you insert a vector into partition 42, it goes into partition 42's graph — completely separate from partition 43's graph.

```
Node 0:
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ Partition 0  │  │ Partition 3  │  │ Partition 6  │  ...
  │ HNSW graph   │  │ HNSW graph   │  │ HNSW graph   │
  │ 37 vectors   │  │ 41 vectors   │  │ 35 vectors   │
  └─────────────┘  └─────────────┘  └─────────────┘
```

---

## 5. Cluster — Multiple Nodes Working Together

### What is a node?

A **node** is one running instance of VecGrid. It could be:
- A separate process on the same machine
- A separate machine in a data center
- A container in Kubernetes

Each node has a unique ID (like `"node-1"`, `"app-server-east-2"`) and knows about all other nodes in the cluster.

### What is a cluster?

A **cluster** is a group of nodes that work together as one logical database. From the outside, you interact with any single node, and it handles routing your request to wherever the data actually lives.

```
┌─────────────────────────────────────┐
│            VecGrid Cluster          │
│                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  │ Node 0  │  │ Node 1  │  │ Node 2  │
│  │ ~90 pts │  │ ~90 pts │  │ ~91 pts │
│  └─────────┘  └─────────┘  └─────────┘
│                                     │
└─────────────────────────────────────┘
         ↑
    Your app talks to
    any node — it just works
```

### How nodes discover each other

When a new node starts, it needs to find the existing cluster. VecGrid supports two discovery methods (covered in detail in [Section 13](#13-discovery--finding-other-nodes)):

- **Multicast**: The node shouts "I'm here!" on the local network
- **Seed nodes**: The node contacts known addresses to ask "who's in the cluster?"

### Rebalancing when nodes join or leave

When a new node joins:
1. The hash ring recalculates partition assignments
2. Some partitions move from existing nodes to the new node
3. Data migrates safely (see [Section 9](#9-safe-migration--moving-data-without-losing-it))

When a node leaves:
1. Its partitions get reassigned to remaining nodes
2. Backup copies are promoted (see [Section 7](#7-failure-handling--when-things-go-wrong))

---

## 6. Replication — Keeping Copies Safe

### Primary and backup

Every partition has one **primary** owner and one or more **backup** copies on other nodes.

```
Partition 42:
  Primary → Node 0  (the "real" copy — handles reads and writes)
  Backup  → Node 1  (the safety copy — ready to take over)
```

Think of it like important documents:
- The **primary** is the original in your filing cabinet
- The **backup** is a photocopy stored at a friend's house
- If your house burns down, your friend still has the copy

### Synchronous replication

When you insert a vector, here's what happens:

```
You:    "Insert vector X"
          │
          ▼
Node 0 (primary):
  1. Write to local index    ✓
  2. Write to WAL (disk)     ✓
  3. Send copy to Node 1     ──→  Node 1 (backup):
                                    1. Write to local index  ✓
                                    2. Send "OK" back        ──→
  4. Receive "OK" from backup ✓
  5. Tell you: "Done!"
          │
          ▼
You:    "Got it, thanks!"
```

The key word is **synchronous** — the primary *waits* for the backup to confirm before telling you "done." This guarantees that when you get a success response, the data exists in at least two places.

### Why synchronous matters

The alternative is **asynchronous** replication: the primary says "done!" immediately and sends the backup copy in the background. This is faster, but dangerous:

```
Async (risky):
  1. Primary writes vector, says "Done!"
  2. Primary starts sending to backup...
  3. Primary crashes before backup receives it
  4. Data is LOST — backup never got it

Sync (safe — what VecGrid does):
  1. Primary writes vector
  2. Primary sends to backup, waits...
  3. Backup confirms
  4. THEN primary says "Done!"
  5. If primary crashes now, backup has the data
```

### backup_count

The `backup_count` setting controls how many backup copies to maintain:

- `backup_count=0`: No backups. Fast, but data is lost if a node dies.
- `backup_count=1` (default): One backup per partition. Survives any single node failure.
- `backup_count=2`: Two backups. Survives two simultaneous node failures.

Higher backup counts use more memory (each backup is a full copy) and slow down writes (must wait for more confirmations).

---

## 7. Failure Handling — When Things Go Wrong

### Heartbeat — "Are you alive?"

Every node periodically sends a small "ping" message to every other node. If a node doesn't respond after several pings, it's declared dead.

```
Node 0 ──ping──→ Node 1    "Are you alive?"
Node 0 ←──pong── Node 1    "Yes!"

Node 0 ──ping──→ Node 2    "Are you alive?"
Node 0 ──ping──→ Node 2    "Are you alive?"    (no response)
Node 0 ──ping──→ Node 2    "Are you alive?"    (no response)
Node 0 ──ping──→ Node 2    "Are you alive?"    (no response)

Node 0: "Node 2 missed 4 pings — declaring it dead."
```

Default settings:
- Ping every **2 seconds**
- Declare dead after **8 seconds** (4 missed pings)

### What happens when a node dies

Let's say Node 1 dies. Here's the sequence:

**Step 1: Detection**
Other nodes notice Node 1 stopped responding to heartbeats.

**Step 2: Remove from cluster**
Node 1 is removed from the hash ring. All nodes agree on the new topology.

**Step 3: Backup promotion**
For every partition where Node 1 was the primary, the backup automatically becomes the new primary.

```
Before failure:
  Partition 42:  Primary=Node 1, Backup=Node 2

After Node 1 dies:
  Partition 42:  Primary=Node 2 (promoted!)
```

Node 2 already has all the data (it was the backup), so this promotion is instant — no data transfer needed.

**Step 4: Create new backups**
The promoted partitions now have no backup. VecGrid selects another node and syncs the data to create a fresh backup.

```
After promotion:
  Partition 42:  Primary=Node 2, Backup=??? (need a new one)

After re-replication:
  Partition 42:  Primary=Node 2, Backup=Node 0 (new backup created)
```

### Zero data loss

As long as `backup_count >= 1`, no data is lost when a single node fails. The backups contain complete copies of every vector, and they're promoted to primary instantly.

```
4-node cluster with backup_count=1:

  Initial:  Node 0, Node 1, Node 2, Node 3  — 1000 vectors

  Kill Node 1:
    Backups promote, re-replicate
    → 1000 vectors (zero loss)

  Kill Node 3:
    Backups promote, re-replicate
    → 1000 vectors (zero loss)

  Down to 2 nodes, all data intact!
```

---

## 8. Smart Routing — Ask Anyone, Get the Right Answer

### The problem

If vector "doc-42" lives on Node 2, but you're connected to Node 0, what happens when you ask Node 0 for it?

### The solution: automatic forwarding

Any node can handle any request. If the data isn't local, the node automatically forwards your request to the correct owner.

```
You → Node 0: "get('doc-42')"
       │
       Node 0 thinks: "doc-42 hashes to partition 87,
                        partition 87 is owned by Node 2"
       │
       Node 0 → Node 2: "get('doc-42')"
       Node 0 ← Node 2: {vector data}
       │
You ← Node 0: {vector data}
```

This is **transparent** — you don't need to know which node owns the data. You just talk to any node in the cluster, and VecGrid handles the routing.

This works for all operations: `put`, `get`, `delete`, and `search`.

### Why this matters

Without smart routing, you'd need to:
1. Know which node owns each piece of data
2. Maintain connections to all nodes
3. Handle re-routing yourself when nodes join or leave

With smart routing, your application code is simple:

```python
# Connect to ANY node — it just works
grid = VecGrid("app-1", dim=384)
grid.start()
grid.put("doc-42", vector)         # Routed automatically
result = grid.get("doc-42")        # Routed automatically
results = grid.search(query, k=10) # Scatter-gather automatically
```

---

## 9. Safe Migration — Moving Data Without Losing It

### When does data need to move?

When the cluster topology changes — nodes join or leave — the hash ring reassigns partitions. Some partitions end up assigned to different nodes than before. The data for those partitions needs to physically move.

```
Before (2 nodes):
  Partition 42 → Node 0
  Partition 43 → Node 1
  Partition 44 → Node 0

After adding Node 2 (3 nodes):
  Partition 42 → Node 0  (no change)
  Partition 43 → Node 1  (no change)
  Partition 44 → Node 2  (MOVED from Node 0)
```

### The danger of "delete first"

A naive approach: delete from the old node, then insert on the new node.

```
DANGEROUS (don't do this):
  1. Delete partition 44 from Node 0
  2. Crash happens here!
  3. Insert partition 44 on Node 2  ← never happens

  Result: Data LOST forever
```

### Migrate-then-delete (what VecGrid does)

VecGrid uses a safe protocol:

```
SAFE (migrate-then-delete):
  1. Node 2 requests partition 44's data from Node 0
  2. Node 0 sends all vectors to Node 2
  3. Node 2 inserts everything into its local index
  4. Node 2 confirms: "I have all the data"
  5. ONLY THEN: Node 0 deletes its local copy

  If a crash happens at step 2 or 3:
    → Node 0 still has the data, nothing is lost
    → Migration can be retried
```

The key principle: **never delete the old copy until the new copy is confirmed**. This ensures zero data loss during any topology change.

---

## 10. Scatter-Gather Search — Searching Everywhere

### The challenge of distributed search

Your vectors are spread across multiple nodes. When you search for the 10 nearest neighbors, the answer might include vectors from *any* node. You can't just search one node — you'd miss results on other nodes.

### How scatter-gather works

```
You: "Find 10 nearest to query Q"
       │
       ▼
  ┌─ SCATTER ──────────────────────────────────┐
  │                                             │
  │  Node 0: search local partitions → top 10  │
  │  Node 1: search local partitions → top 10  │
  │  Node 2: search local partitions → top 10  │
  │                                             │
  └─────────────────────────────────────────────┘
       │
       ▼
  ┌─ GATHER ───────────────────────────────────┐
  │                                             │
  │  Collect all 30 results                    │
  │  Sort by distance                          │
  │  Return global top 10                      │
  │                                             │
  └─────────────────────────────────────────────┘
       │
       ▼
You: [10 best results from across the entire cluster]
```

**Scatter**: Send the query to all nodes. Each node searches only its **primary** partitions (not backups — to avoid counting the same vector twice).

**Gather**: Collect results from all nodes, merge them, sort by distance, and return the overall top-k.

### Why this gives high recall

Because every partition is searched, no data is skipped. The only source of imperfection is HNSW's approximate nature within each partition — it might miss a neighbor within a single partition's graph. But since partitions are small (a few thousand vectors each), HNSW's accuracy is very high.

At 10K total vectors across 271 partitions, each partition has ~37 vectors — HNSW is essentially doing brute force at that scale, giving 100% recall.

---

## 11. Persistence — Surviving Crashes

### The problem

HNSW indexes live in memory (RAM). If the process crashes or the machine reboots, everything in memory is gone. Persistence ensures data survives restarts.

### WAL — The Write-Ahead Log

A **WAL** is like a journal or diary. Before making any change, you write it down in the journal first.

```
Analogy: Cooking with a recipe journal

  Before you cook each dish:
    1. Write in your journal: "Making pasta, ingredients: ..."
    2. Then actually cook the pasta

  If the kitchen catches fire mid-cooking:
    1. You lost the half-cooked pasta
    2. But your journal survived!
    3. You can read the journal and remake everything
```

In VecGrid:

```
Insert vector "doc-1":
  1. Append to WAL file on disk: {op: "insert", id: "doc-1", vector: [...]}
  2. fsync (force write to disk — see below)
  3. Insert into in-memory HNSW index
  4. Return success

If crash happens after step 2 but before step 3:
  → WAL has the record on disk
  → On restart, replay the WAL to rebuild the index
```

Each partition has its own WAL file:

```
data/node-1/wal/
  partition_0000.wal
  partition_0001.wal
  partition_0002.wal
  ...
```

### Snapshots — Full photographs

The WAL grows over time. Replaying thousands of entries on restart would be slow. A **snapshot** is a complete dump of the current state — like taking a photograph instead of replaying every moment.

```
Analogy: Bank account

  WAL approach: "Record every transaction since the account opened"
    +$100, -$20, +$50, -$10, +$200, -$30, ...
    (To know the balance, replay ALL transactions)

  Snapshot approach: "The balance is $290 as of Tuesday"
    (Just load the snapshot — instant)
```

In VecGrid:

```
data/node-1/snapshots/
  snapshot_0042_v500.bin    ← Partition 42 at version 500
```

The snapshot contains: every vector, every metadata entry, and the version number.

### Recovery: snapshot + WAL replay

On restart, VecGrid combines both:

```
Recovery for partition 42:
  1. Load latest snapshot: version 500 (contains vectors 1-500)
  2. Read WAL entries after version 500: entries 501, 502, 503
  3. Replay those 3 entries
  4. Index is fully restored at version 503!
```

This gives you the speed of snapshots (don't replay everything) with the completeness of WAL (don't miss recent writes).

### fsync — Making sure it's really on disk

When you write a file, the operating system often keeps it in a memory buffer first and writes to disk later (for performance). If the machine loses power before the buffer is flushed, the data is lost.

**fsync** forces the operating system to write the buffer to physical disk immediately. It's slower, but guarantees the data is durable.

```
Without fsync:
  write("data") → OS buffer (in memory) ─── power loss ──→ DATA LOST

With fsync:
  write("data") → OS buffer → fsync → physical disk ──→ power loss ──→ data safe!
```

VecGrid calls fsync after every WAL append, ensuring crash safety.

---

## 12. Transport — How Nodes Talk

### The message system

Nodes communicate by sending **messages** — small structured packets of data.

```python
Message:
  type:    "insert"                    # What kind of operation
  sender:  "node-0"                    # Who sent it
  payload: {id: "doc-1", vector: [...]} # The actual data
```

VecGrid supports two transport layers:

### InProcessTransport — Same machine

When all nodes run in the same process (for testing or single-machine use), messages are passed directly through memory. No network, no serialization, no overhead.

```
Node 0 ──direct function call──→ Node 1
         (same process, same memory)
```

This is like passing a note to someone sitting next to you — instant.

### TCPTransport — Across machines

For real distributed deployments, nodes communicate over **TCP** (the same protocol your web browser uses).

```
Node 0 (Machine A) ──TCP socket──→ Node 1 (Machine B)
                    ← JSON data ──
```

Messages are serialized to **JSON** (a text format), sent over the network, and deserialized on the other end. Each message is prefixed with its length so the receiver knows where one message ends and the next begins.

```
Wire format:
  [4 bytes: message length][JSON message bytes]
  [4 bytes: message length][JSON message bytes]
  ...
```

### When to use which

| Transport | Use case |
|-----------|----------|
| `"embedded"` (InProcess) | Testing, single-machine, demos |
| `"tcp"` | Production, multi-machine deployments |

---

## 13. Discovery — Finding Other Nodes

When a node starts, it needs to find the rest of the cluster. VecGrid offers two approaches:

### Multicast — Shouting in a room

**Multicast** is like walking into a room and shouting "Is anyone here?" Everyone in the room hears you and shouts back.

```
New node:  "Hello! I'm node-3, at 10.0.1.4:5701!"
              ↓ (UDP multicast to 224.2.2.3:54327)
Node 0:    "I heard node-3! Adding to my cluster."
Node 1:    "I heard node-3! Adding to my cluster."
Node 2:    "I heard node-3! Adding to my cluster."
```

Nodes continuously broadcast their presence every 2 seconds. New nodes listen for these broadcasts to discover the cluster.

**Pros**: Zero configuration — nodes find each other automatically.
**Cons**: Only works on local networks. Most cloud environments (AWS, GCP) block multicast.

### Seed nodes — Asking a known friend

**Seed discovery** is like knowing one person's phone number. You call them and ask "Who else is in the group?"

```
Config: seeds = ["10.0.1.1:5701"]

New node → 10.0.1.1:5701: "Who's in the cluster?"
           10.0.1.1:5701 → New node: "Me, 10.0.1.2:5701, and 10.0.1.3:5701"

New node now knows the full cluster!
```

You only need to know *one* existing node's address. That node shares its knowledge of the full cluster.

**Pros**: Works across any network, including cloud environments.
**Cons**: Requires configuring at least one seed address.

### When to use which

| Discovery | Use case |
|-----------|----------|
| `"none"` | Single-node or InProcess testing |
| `"multicast"` | Local network, development |
| `"seed"` | Cloud, production, cross-network |

---

## 14. Putting It All Together

Let's trace three complete operations through the entire system.

### Writing a vector (end-to-end)

```
grid.put("doc-42", [0.1, 0.8, 0.3, ...], {"title": "Hello World"})

Step 1: HASH
  SHA256("doc-42") % 271 = partition 155

Step 2: ROUTE
  Hash ring says: partition 155 → primary on Node 1
  We're on Node 0, so forward to Node 1

Step 3: PRIMARY WRITE (on Node 1)
  a. Increment version: 99 → 100
  b. Append to WAL:  wal/partition_0155.wal ← {insert, "doc-42", v100}
  c. fsync WAL to disk
  d. Insert into partition 155's HNSW graph

Step 4: REPLICATE
  Hash ring says: backup for partition 155 → Node 2
  Send to Node 2: "backup_insert doc-42 [0.1, 0.8, 0.3, ...]"
  Node 2 inserts into its local copy of partition 155
  Node 2 replies: "OK"

Step 5: ACKNOWLEDGE
  Node 1 → Node 0 → You: "Success!"
  Data now exists on Node 1 (primary) and Node 2 (backup)
```

### Searching (end-to-end)

```
results = grid.search([0.5, 0.2, 0.9, ...], k=5)

Step 1: LOCAL SEARCH
  Node 0 searches its ~90 primary partitions
  Each partition's HNSW graph returns its best matches
  Node 0 collects local results: [(id="doc-7", dist=0.12), ...]

Step 2: SCATTER
  Node 0 sends search request to Node 1 and Node 2
  Node 1 searches its ~90 primary partitions → results
  Node 2 searches its ~91 primary partitions → results

Step 3: GATHER
  Node 0 collects results from all nodes:
    From self:   [(doc-7, 0.12), (doc-99, 0.18), ...]
    From Node 1: [(doc-42, 0.08), (doc-3, 0.15), ...]
    From Node 2: [(doc-200, 0.11), (doc-55, 0.19), ...]

Step 4: MERGE
  Sort all results by distance:
    1. doc-42   dist=0.08  (from Node 1)
    2. doc-200  dist=0.11  (from Node 2)
    3. doc-7    dist=0.12  (from Node 0)
    4. doc-3    dist=0.15  (from Node 1)
    5. doc-99   dist=0.18  (from Node 0)

  Return top 5 to you
```

### Handling a failure (end-to-end)

```
Node 1 crashes unexpectedly!

Step 1: DETECTION (after ~8 seconds)
  Node 0: "Node 1 missed 4 heartbeats — declaring dead"
  Node 2: "Node 1 missed 4 heartbeats — declaring dead"

Step 2: REMOVE FROM RING
  All nodes remove Node 1 from the hash ring
  Partition ownership recalculated

Step 3: BACKUP PROMOTION
  Partition 155 was: Primary=Node 1, Backup=Node 2
  Node 2 promotes its backup of partition 155 to primary
  (Node 2 already has all the data — instant!)

Step 4: RE-REPLICATION
  Partition 155 now: Primary=Node 2, Backup=???
  Node 2 selects Node 0 as the new backup
  Node 2 sends all of partition 155's data to Node 0
  Node 0 creates a backup copy

Step 5: BACK TO NORMAL
  Partition 155: Primary=Node 2, Backup=Node 0
  All data intact, zero loss, cluster continues serving queries
```

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **ANN** | Approximate Nearest Neighbor — finding "close enough" matches quickly instead of scanning everything |
| **Backup** | A copy of a partition stored on a different node for safety |
| **Backup count** | How many backup copies to maintain per partition (default: 1) |
| **Backup promotion** | When a primary node dies, its backup automatically becomes the new primary |
| **Brute force** | Comparing a query against every single stored vector — accurate but slow |
| **Cluster** | A group of nodes working together as one database |
| **Consistent hashing** | A method of mapping data to nodes that minimizes data movement when nodes join/leave |
| **Cosine distance** | A distance metric measuring the angle between two vectors (ignores magnitude) |
| **Dimension (dim)** | The number of values in each vector (e.g., 384) |
| **Discovery** | The process by which nodes find each other to form a cluster |
| **Dot product** | A distance metric combining direction and magnitude |
| **Embedded** | A database that runs inside your application process (no separate server) |
| **Euclidean distance** | Straight-line distance between two points in space |
| **ef_construction** | HNSW parameter: how thoroughly to search when building the graph |
| **ef_search** | HNSW parameter: how thoroughly to search when answering queries |
| **fsync** | Operating system call that forces data from memory buffers to physical disk |
| **Hash ring** | A circular structure mapping vector IDs to partition numbers |
| **Heartbeat** | Periodic "are you alive?" messages between nodes |
| **HNSW** | Hierarchical Navigable Small World — the core search algorithm |
| **hnswlib** | C++ implementation of HNSW, 50-100x faster than pure Python |
| **M** | HNSW parameter: maximum connections per node in the graph |
| **Metadata** | Extra information stored alongside a vector (e.g., title, category) |
| **Migrate-then-delete** | Safety protocol: only delete old data after new location confirms receipt |
| **Multicast** | Network broadcast for automatic node discovery on local networks |
| **Node** | One running instance of VecGrid |
| **Partition** | A logical subdivision of the data (VecGrid uses 271 by default) |
| **Primary** | The main owner of a partition — handles reads and writes |
| **Recall** | Accuracy metric: what percentage of true nearest neighbors were found |
| **Rebalancing** | Redistributing partitions when nodes join or leave the cluster |
| **Replication** | Copying data to multiple nodes for safety |
| **Scatter-gather** | Search pattern: send query to all nodes, merge results |
| **Seed node** | A known node address used to discover the rest of the cluster |
| **Smart routing** | Automatic forwarding of requests to the correct node |
| **Snapshot** | A complete dump of a partition's state at a point in time |
| **Synchronous replication** | Waiting for backup confirmation before acknowledging a write |
| **Transport** | The communication layer between nodes (InProcess or TCP) |
| **Vector** | A list of numbers representing the meaning of some data |
| **WAL** | Write-Ahead Log — a journal of all changes, written to disk before applied |

---

*This document covers VecGrid v0.0.1. See the [README](README.md) for quick-start guides and API reference.*
