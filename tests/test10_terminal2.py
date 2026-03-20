import time, numpy as np
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

time.sleep(5) # Wait a bit before shutting down

grid.stop()
print('Node 2 shutdown.')
