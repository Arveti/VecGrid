import time, numpy as np
from vecgrid import VecGrid

print('=== Node 1 (seed node) ===')
grid = VecGrid('node-1', dim=32, transport='tcp', port=5701,
               discovery='seed', seeds=[], backup_count=1,
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
print('Waiting for node-2 to join...')
for _ in range(20): # a bit shorter wait
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

time.sleep(10) # Wait for terminal 2 to finish

grid.stop()
print('Node 1 shutdown.')
