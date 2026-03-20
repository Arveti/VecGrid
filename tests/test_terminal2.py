import sys, os, time
import numpy as np
# Ensure we import the local vecgrid, not a globally installed outdated one
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vecgrid import VecGrid

print('=== Node 2 (joins via discovery) ===')
grid = VecGrid('node-2', dim=32, transport='tcp', port=5702,
               discovery='seed', seeds=['127.0.0.1:5701'], backup_count=1,
               data_dir='/tmp/vecgrid_multi/node-2')
grid.start()
print(f'Listening on port 5702')

# Wait for discovery
time.sleep(3)
print(f'Cluster: {sorted(grid._node._cluster_nodes)}')

# Insert more data
np.random.seed(99)
categories = ["Internal", "Public"]
for i in range(100, 200):
    grid.put(f'vec-{i}', np.random.randn(32).astype(np.float32), {'i': i, 'access': categories[i % 2]})
print(f'Inserted 100 more vectors')

time.sleep(1)
print(f'Cluster size: {grid.cluster_size()}')
print(f'Local: {grid.local_size()} primary, {grid.local_backup_size()} backup')

# Cross-node search
query = np.random.randn(32).astype(np.float32)
print("\\n--- Simple Cross-node Search ---")
results = grid.search(query, k=5)
print(f'Unfiltered cross-node: {[(r.vector_id, r.source_node, r.metadata.get("access")) for r in results]}')

print("\\n--- Pushdown Filtered Cross-node Search ('access' == 'Internal') ---")
f_spec = {"field": "access", "op": "eq", "value": "Internal"}
results_filtered = grid.search(query, k=5, filter=f_spec)
print(f'Filtered cross-node: {[(r.vector_id, r.source_node, r.metadata.get("access")) for r in results_filtered]}')

time.sleep(5) # Wait a bit before shutting down

grid.stop()
print('Node 2 shutdown.')
