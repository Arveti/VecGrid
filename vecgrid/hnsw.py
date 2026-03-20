"""
HNSW Index — dual backend.

Backend selection (automatic):
    1. hnswlib (C++ with Python bindings) — if installed, 50-100x faster
    2. NumpyHNSW (pure Python + vectorized numpy) — always available

Both backends expose the same HNSWIndex interface so the rest of
VecGrid doesn't care which is running.

Install hnswlib for production:
    pip install hnswlib
"""

import math
import random
import heapq
import logging
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger("vecgrid.hnsw")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HNSWConfig:
    """Configuration for HNSW index."""
    M: int = 16                  # Max connections per node per layer
    M0: int = 0                  # Max connections at layer 0 (0 = auto = 2*M)
    ef_construction: int = 200   # Beam width during construction
    ef_search: int = 50          # Beam width during search
    distance_metric: str = "cosine"  # cosine, euclidean, dot

    def __post_init__(self):
        if self.M0 == 0:
            self.M0 = 2 * self.M


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class HNSWIndex:
    """
    HNSW index interface.

    Implementations: HNSWLibIndex (C++), NumpyHNSWIndex (pure Python).
    Use create_index() factory to auto-select the best available backend.
    """

    def insert(self, vector_id: str, vector: np.ndarray,
               metadata: Optional[dict] = None):
        raise NotImplementedError

    def search(self, query: np.ndarray, k: int = 10,
               ef: Optional[int] = None,
               filter_fn: Optional[Callable] = None
               ) -> list[tuple[float, str, dict]]:
        """Returns list of (distance, vector_id, metadata)."""
        raise NotImplementedError

    def delete(self, vector_id: str) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, vector_id: str) -> bool:
        raise NotImplementedError

    @property
    def vectors(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @property
    def metadata(self) -> dict[str, dict]:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Backend 1: hnswlib (C++)
# ---------------------------------------------------------------------------

def _try_import_hnswlib():
    try:
        import hnswlib
        return hnswlib
    except ImportError:
        return None


class HNSWLibIndex(HNSWIndex):
    """
    hnswlib-backed HNSW index.

    Uses the C++ hnswlib library for 50-100x faster distance computation
    and graph traversal. This is the production backend.
    """

    def __init__(self, dim: int, config: Optional[HNSWConfig] = None,
                 max_elements: int = 50000):
        hnswlib = _try_import_hnswlib()
        if hnswlib is None:
            raise ImportError("hnswlib not installed. pip install hnswlib")

        self.dim = dim
        self.config = config or HNSWConfig()
        self._max_elements = max_elements

        space_map = {"cosine": "cosine", "euclidean": "l2", "dot": "ip"}
        space = space_map.get(self.config.distance_metric, "cosine")

        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            M=self.config.M,
            ef_construction=self.config.ef_construction,
        )
        self._index.set_ef(self.config.ef_search)

        # ID mapping: hnswlib uses integer labels
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}
        self._next_id: int = 0
        self._deleted: set[str] = set()

    def _ensure_capacity(self):
        if self._next_id >= self._max_elements:
            new_max = self._max_elements * 2
            self._index.resize_index(new_max)
            self._max_elements = new_max

    def insert(self, vector_id: str, vector: np.ndarray,
               metadata: Optional[dict] = None):
        vec = vector.astype(np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of dim {self.dim}, got shape {vector.shape}"
            )

        if vector_id in self._deleted:
            # Un-delete: unmark in hnswlib so it can be re-added
            int_id = self._str_to_int.get(vector_id)
            if int_id is not None:
                try:
                    self._index.unmark_deleted(int_id)
                except (RuntimeError, AttributeError):
                    pass  # older hnswlib versions may not support unmark
            self._deleted.discard(vector_id)

        self._ensure_capacity()

        if vector_id in self._str_to_int:
            int_id = self._str_to_int[vector_id]
        else:
            int_id = self._next_id
            self._next_id += 1
            self._str_to_int[vector_id] = int_id
            self._int_to_str[int_id] = vector_id

        self._index.add_items(
            vec.reshape(1, -1), np.array([int_id])
        )
        self._vectors[vector_id] = vec.copy()
        self._metadata[vector_id] = metadata if metadata is not None else self._metadata.get(vector_id, {})

    def search(self, query: np.ndarray, k: int = 10,
               ef: Optional[int] = None,
               filter_fn: Optional[Callable] = None
               ) -> list[tuple[float, str, dict]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if len(self) == 0:
            return []

        if ef:
            self._index.set_ef(ef)

        # Clamp fetch_k to actual index element count to avoid hnswlib
        # "Cannot return the results in a contiguous 2D array" error.
        index_count = self._index.get_current_count()
        if index_count == 0:
            return []
        active_count = index_count - len(self._deleted)
        if active_count <= 0:
            return []
        fetch_k = min(k * 3 + len(self._deleted), index_count)
        fetch_k = min(fetch_k, active_count)
        fetch_k = max(1, fetch_k)

        # hnswlib requires ef >= k for the query
        prev_ef = self.config.ef_search
        if fetch_k > (ef or prev_ef):
            self._index.set_ef(fetch_k)

        try:
            labels, distances = self._index.knn_query(
                query.reshape(1, -1).astype(np.float32), k=fetch_k
            )
        except RuntimeError:
            # Fallback: try with exactly active_count
            fetch_k = max(1, active_count)
            self._index.set_ef(max(fetch_k, ef or prev_ef))
            labels, distances = self._index.knn_query(
                query.reshape(1, -1).astype(np.float32), k=fetch_k
            )

        results = []
        for dist, int_id in zip(distances[0], labels[0]):
            if int_id < 0:
                continue
            str_id = self._int_to_str.get(int(int_id))
            if str_id is None or str_id in self._deleted:
                continue
            meta = self._metadata.get(str_id, {})
            if filter_fn and not filter_fn(meta):
                continue
            results.append((float(dist), str_id, meta))
            if len(results) >= k:
                break

        # Always restore ef to default
        self._index.set_ef(self.config.ef_search)

        return results

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._str_to_int or vector_id in self._deleted:
            return False
        int_id = self._str_to_int[vector_id]
        try:
            self._index.mark_deleted(int_id)
        except RuntimeError as e:
            logger.warning(f"hnswlib mark_deleted failed for {vector_id}: {e}")
        self._deleted.add(vector_id)
        self._vectors.pop(vector_id, None)
        self._metadata.pop(vector_id, None)
        return True

    def __len__(self) -> int:
        return len(self._str_to_int) - len(self._deleted)

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self._str_to_int and vector_id not in self._deleted

    @property
    def vectors(self) -> dict[str, np.ndarray]:
        return {k: v for k, v in self._vectors.items() if k not in self._deleted}

    @property
    def metadata(self) -> dict[str, dict]:
        return {k: v for k, v in self._metadata.items() if k not in self._deleted}

    def stats(self) -> dict:
        return {
            "backend": "hnswlib",
            "total_vectors": len(self),
            "dimensions": self.dim,
            "max_elements": self._max_elements,
            "deleted": len(self._deleted),
        }


# ---------------------------------------------------------------------------
# Backend 2: Optimized pure numpy HNSW
# ---------------------------------------------------------------------------

class NumpyHNSWIndex(HNSWIndex):
    """
    Pure Python HNSW with vectorized numpy distance computation.

    Key optimizations over naive implementations:
    - Vectors stored in contiguous numpy arrays for batch distance ops
    - Pre-normalized vectors for cosine (distance = 1 - dot product)
    - Integer IDs internally, string mapping at the boundary
    - Batch distance computation: one np.dot call for many candidates
    """

    def __init__(self, dim: int, config: Optional[HNSWConfig] = None):
        self.dim = dim
        self.config = config or HNSWConfig()
        self._ml = 1.0 / math.log(self.config.M)

        # Storage — contiguous arrays for vectorized ops
        self._vectors_dict: dict[str, np.ndarray] = {}    # possibly normalized (for graph)
        self._original_vectors: dict[str, np.ndarray] = {}  # always original (for retrieval)
        self._metadata_dict: dict[str, dict] = {}

        # Graph layers: layer -> {node_id -> set of neighbor_ids}
        self._graphs: list[dict[str, set]] = [{}]
        self._node_levels: dict[str, int] = {}
        self._entry_point: Optional[str] = None
        self._max_level: int = 0

        # Distance function
        self._needs_normalize = self.config.distance_metric == "cosine"
        self._dist_fn = self._get_batch_dist_fn()
        self._pair_dist_fn = self._get_pair_dist_fn()

    def _get_batch_dist_fn(self):
        """Return a function: (query_vec, candidate_vecs_matrix) -> distances_array"""
        metric = self.config.distance_metric
        if metric == "cosine":
            def fn(q, M):
                # q and rows of M are pre-normalized
                dots = M @ q
                return 1.0 - dots
            return fn
        elif metric == "euclidean":
            def fn(q, M):
                diff = M - q
                return np.linalg.norm(diff, axis=1)
            return fn
        elif metric == "dot":
            def fn(q, M):
                return -(M @ q)
            return fn
        raise ValueError(f"Unknown metric: {metric}")

    def _get_pair_dist_fn(self):
        metric = self.config.distance_metric
        if metric == "cosine":
            return lambda a, b: 1.0 - float(np.dot(a, b))
        elif metric == "euclidean":
            return lambda a, b: float(np.linalg.norm(a - b))
        elif metric == "dot":
            return lambda a, b: -float(np.dot(a, b))
        raise ValueError(f"Unknown metric: {metric}")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def _random_level(self) -> int:
        level = 0
        while random.random() < math.exp(-1.0 / self._ml) and level < 32:
            level += 1
        return level

    def _ensure_layers(self, level: int):
        while len(self._graphs) <= level:
            self._graphs.append({})

    def _dist_to_query(self, query: np.ndarray, node_id: str) -> float:
        return self._pair_dist_fn(query, self._vectors_dict[node_id])

    def _dist_pair(self, a: str, b: str) -> float:
        return self._pair_dist_fn(self._vectors_dict[a], self._vectors_dict[b])

    def _search_layer(self, query: np.ndarray, entry_points: list[str],
                      ef: int, layer: int) -> list[tuple[float, str]]:
        """Beam search on a single layer. Returns sorted (distance, id) pairs."""
        visited = set(entry_points)
        candidates = []  # min-heap
        results = []     # max-heap (negated)

        for ep in entry_points:
            dist = self._dist_to_query(query, ep)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            f_dist = -results[0][0]
            if c_dist > f_dist:
                break

            neighbors = self._graphs[layer].get(c_id, set())
            # Batch distance computation for unvisited neighbors
            unvisited = [n for n in neighbors if n not in visited]
            if not unvisited:
                continue
            visited.update(unvisited)

            if len(unvisited) > 4:
                # Vectorized batch distance
                vecs = np.array([self._vectors_dict[n] for n in unvisited])
                dists = self._dist_fn(query, vecs)
                for n_id, n_dist in zip(unvisited, dists):
                    n_dist = float(n_dist)
                    f_dist = -results[0][0]
                    if n_dist < f_dist or len(results) < ef:
                        heapq.heappush(candidates, (n_dist, n_id))
                        heapq.heappush(results, (-n_dist, n_id))
                        if len(results) > ef:
                            heapq.heappop(results)
            else:
                for n_id in unvisited:
                    n_dist = self._dist_to_query(query, n_id)
                    f_dist = -results[0][0]
                    if n_dist < f_dist or len(results) < ef:
                        heapq.heappush(candidates, (n_dist, n_id))
                        heapq.heappush(results, (-n_dist, n_id))
                        if len(results) > ef:
                            heapq.heappop(results)

        return sorted([(-d, nid) for d, nid in results])

    def _select_neighbors(self, query_id: str,
                          candidates: list[tuple[float, str]],
                          M: int) -> list[str]:
        """Heuristic neighbor selection (diversity-aware)."""
        if len(candidates) <= M:
            return [cid for _, cid in candidates]

        candidates_sorted = sorted(candidates)
        selected = []

        for dist, cid in candidates_sorted:
            if len(selected) >= M:
                break
            good = True
            for sel_id in selected:
                if self._dist_pair(cid, sel_id) < dist:
                    good = False
                    break
            if good:
                selected.append(cid)

        # Fill remaining with closest
        if len(selected) < M:
            selected_set = set(selected)
            for _, cid in candidates_sorted:
                if cid not in selected_set:
                    selected.append(cid)
                    if len(selected) >= M:
                        break

        return selected

    def _connect(self, node_id: str, neighbors: list[str], layer: int):
        """Bidirectional connection with pruning."""
        M_max = self.config.M0 if layer == 0 else self.config.M

        if node_id not in self._graphs[layer]:
            self._graphs[layer][node_id] = set()

        for neighbor in neighbors:
            self._graphs[layer][node_id].add(neighbor)
            if neighbor not in self._graphs[layer]:
                self._graphs[layer][neighbor] = set()
            self._graphs[layer][neighbor].add(node_id)

            if len(self._graphs[layer][neighbor]) > M_max:
                cands = [(self._dist_pair(neighbor, n), n)
                         for n in self._graphs[layer][neighbor]]
                new_neighbors = self._select_neighbors(neighbor, cands, M_max)
                removed = self._graphs[layer][neighbor] - set(new_neighbors)
                self._graphs[layer][neighbor] = set(new_neighbors)
                for r in removed:
                    if r in self._graphs[layer]:
                        self._graphs[layer][r].discard(neighbor)

    def insert(self, vector_id: str, vector: np.ndarray,
               metadata: Optional[dict] = None):
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected dim {self.dim}, got {vector.shape}")

        original = vector.astype(np.float32).copy()
        self._original_vectors[vector_id] = original

        vec = original.copy()
        if self._needs_normalize:
            vec = self._normalize(vec)

        self._vectors_dict[vector_id] = vec
        if metadata:
            self._metadata_dict[vector_id] = metadata

        node_level = self._random_level()
        self._node_levels[vector_id] = node_level
        self._ensure_layers(node_level)

        if self._entry_point is None:
            self._entry_point = vector_id
            self._max_level = node_level
            for l in range(node_level + 1):
                self._graphs[l][vector_id] = set()
            return

        ep = self._entry_point

        # Greedy descent from top to node_level + 1
        for level in range(self._max_level, node_level, -1):
            if ep in self._graphs[level]:
                results = self._search_layer(vec, [ep], ef=1, layer=level)
                if results:
                    ep = results[0][1]

        # Insert at each layer from node_level down to 0
        for level in range(min(node_level, self._max_level), -1, -1):
            if ep not in self._graphs[level]:
                self._graphs[level][vector_id] = set()
                continue

            results = self._search_layer(
                vec, [ep], ef=self.config.ef_construction, layer=level
            )
            M = self.config.M0 if level == 0 else self.config.M
            neighbors = self._select_neighbors(vector_id, results, M)
            self._connect(vector_id, neighbors, level)
            if results:
                ep = results[0][1]

        if node_level > self._max_level:
            self._entry_point = vector_id
            self._max_level = node_level

    def search(self, query: np.ndarray, k: int = 10,
               ef: Optional[int] = None,
               filter_fn: Optional[Callable] = None
               ) -> list[tuple[float, str, dict]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if self._entry_point is None:
            return []
        if query.shape != (self.dim,):
            raise ValueError(f"Expected dim {self.dim}, got {query.shape}")

        q = query.astype(np.float32)
        if self._needs_normalize:
            q = self._normalize(q)

        ef = ef or self.config.ef_search
        ef = max(ef, k)
        ep = self._entry_point

        for level in range(self._max_level, 0, -1):
            if ep in self._graphs[level]:
                results = self._search_layer(q, [ep], ef=1, layer=level)
                if results:
                    ep = results[0][1]

        results = self._search_layer(q, [ep], ef=ef, layer=0)

        final = []
        for dist, nid in results:
            meta = self._metadata_dict.get(nid, {})
            if filter_fn and not filter_fn(meta):
                continue
            final.append((dist, nid, meta))
            if len(final) >= k:
                break

        return final

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._vectors_dict:
            return False

        level = self._node_levels[vector_id]
        for l in range(level + 1):
            if vector_id in self._graphs[l]:
                neighbors = self._graphs[l].pop(vector_id)
                for neighbor in neighbors:
                    if neighbor in self._graphs[l]:
                        self._graphs[l][neighbor].discard(vector_id)

        del self._vectors_dict[vector_id]
        del self._node_levels[vector_id]
        self._original_vectors.pop(vector_id, None)
        self._metadata_dict.pop(vector_id, None)

        if self._entry_point == vector_id:
            if self._vectors_dict:
                self._entry_point = next(iter(self._vectors_dict))
                self._max_level = self._node_levels[self._entry_point]
            else:
                self._entry_point = None
                self._max_level = 0

        return True

    def __len__(self) -> int:
        return len(self._vectors_dict)

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self._vectors_dict

    @property
    def vectors(self) -> dict[str, np.ndarray]:
        return self._original_vectors

    @property
    def metadata(self) -> dict[str, dict]:
        return self._metadata_dict

    def stats(self) -> dict:
        layer_sizes = {l: len(g) for l, g in enumerate(self._graphs) if g}
        avg_conn = {}
        for l, g in enumerate(self._graphs):
            if g:
                conns = [len(nb) for nb in g.values()]
                avg_conn[l] = sum(conns) / len(conns) if conns else 0
        return {
            "backend": "numpy",
            "total_vectors": len(self),
            "dimensions": self.dim,
            "max_level": self._max_level,
            "layer_sizes": layer_sizes,
            "avg_connections": avg_conn,
            "entry_point": self._entry_point,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_HNSWLIB_AVAILABLE = _try_import_hnswlib() is not None


def create_index(dim: int, config: Optional[HNSWConfig] = None,
                 backend: str = "auto",
                 max_elements: int = 50000) -> HNSWIndex:
    """
    Create an HNSW index with the best available backend.

    Args:
        dim: Vector dimensions
        config: HNSW configuration
        backend: "auto" (hnswlib if available, else numpy), "hnswlib", or "numpy"
        max_elements: Initial capacity for hnswlib backend (auto-resizes)

    Returns:
        HNSWIndex instance
    """
    if backend == "auto":
        if _HNSWLIB_AVAILABLE:
            logger.info("Using hnswlib (C++) backend")
            return HNSWLibIndex(dim, config, max_elements)
        else:
            logger.info("Using numpy (Python) backend — install hnswlib for 50-100x speedup")
            return NumpyHNSWIndex(dim, config)
    elif backend == "hnswlib":
        return HNSWLibIndex(dim, config, max_elements)
    elif backend == "numpy":
        return NumpyHNSWIndex(dim, config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_backend_name() -> str:
    """Return which backend will be used by default."""
    return "hnswlib" if _HNSWLIB_AVAILABLE else "numpy"
