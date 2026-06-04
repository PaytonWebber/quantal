"""Ergonomic Index over the quantajump C ABI.

    import numpy as np
    from quantajump import Index

    with Index(dim=384) as index:
        ids = index.add(vectors)                 # auto-assigns ids 0..n-1
        hits = index.search(query, k=10)         # -> [(id, score), ...]
        index.remove(ids[0])
        index.save("docs.tq")
    index = Index.load("docs.tq")
"""

import os

import numpy as np

from . import _native

_u64p = _native._u64p
_f32p = _native._f32p
_usizep = _native._usizep


class Index:
    def __init__(self, dim=None, *, lib_path=None, ef_construction=200, seed=42,
                 _handle=None, _lib=None):
        self._lib = _lib or _native.load(dim, lib_path)[0]
        self.dim = int(self._lib.qj_dim())
        self.routing_bits = int(self._lib.qj_routing_bits())
        if _handle is not None:
            self._handle = _handle
        else:
            self._handle = self._lib.qj_index_create(ef_construction, seed)
            if not self._handle:
                raise MemoryError("qj_index_create failed")
        self._next_id = 0
        self._contexts = {}
        self._context_len = -1

    # --- ingest ---

    def add(self, vectors, ids=None, threads=0):
        """Adds vectors (shape (n, dim) or (dim,)). When `ids` is omitted,
        contiguous ids are assigned automatically. Returns the ids used."""
        vectors = self._as_matrix(vectors, "vectors")
        n = vectors.shape[0]
        if ids is None:
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.uint64)
        else:
            ids = np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)
            if ids.shape[0] != n:
                raise ValueError("len(ids) must match number of vectors")
        threads = threads or os.cpu_count() or 1
        rc = self._lib.qj_index_add_batch(
            self._handle, ids.ctypes.data_as(_u64p), vectors.ctypes.data_as(_f32p), n, threads
        )
        if rc != 0:
            raise RuntimeError("add failed (duplicate id, or out of memory)")
        self._next_id = max(self._next_id, int(ids.max()) + 1) if n else self._next_id
        return ids

    def remove(self, id_):
        """Removes a vector by id; returns True if it was present."""
        return self._lib.qj_index_remove(self._handle, int(id_)) == 0

    def __len__(self):
        return int(self._lib.qj_index_len(self._handle))

    # --- query ---

    def search(self, query, k=10, m=128):
        """Single-query search -> list of (id, score), best first."""
        scores, ids, counts = self.search_batch(self._as_matrix(query, "query"), k=k, m=m, threads=1)
        c = counts[0]
        return list(zip(ids[0, :c].tolist(), scores[0, :c].tolist()))

    def search_batch(self, queries, k=10, m=128, threads=0):
        """Batch search -> (scores, ids, counts) numpy arrays; row i is valid
        up to counts[i]. threads=0 uses all cores."""
        queries = self._as_matrix(queries, "queries")
        n = queries.shape[0]
        threads = threads or os.cpu_count() or 1
        ids = np.zeros((n, k), dtype=np.uint64)
        scores = np.zeros((n, k), dtype=np.float32)
        counts = np.zeros(n, dtype=np.uintp)
        rc = self._lib.qj_search_batch(
            self._handle, queries.ctypes.data_as(_f32p), n, k, m, threads,
            ids.ctypes.data_as(_u64p), scores.ctypes.data_as(_f32p), counts.ctypes.data_as(_usizep),
        )
        if rc != 0:
            raise RuntimeError("search failed")
        return scores, ids, counts.astype(np.int64)

    def search_filtered(self, query, allowlist, k=10, m=128):
        """Search restricted to an id allowlist -> list of (id, score)."""
        query = self._as_matrix(query, "query")
        if query.shape[0] != 1:
            raise ValueError("search_filtered takes a single query")
        allowlist = np.ascontiguousarray(allowlist, dtype=np.uint64).reshape(-1)
        ctx = self._context(m)
        ids = np.zeros(k, dtype=np.uint64)
        scores = np.zeros(k, dtype=np.float32)
        count = self._lib.qj_search_filtered(
            self._handle, ctx, query.ctypes.data_as(_f32p),
            allowlist.ctypes.data_as(_u64p), allowlist.shape[0], k,
            ids.ctypes.data_as(_u64p), scores.ctypes.data_as(_f32p),
        )
        return list(zip(ids[:count].tolist(), scores[:count].tolist()))

    # --- persistence ---

    def save(self, path):
        if self._lib.qj_index_save(self._handle, str(path).encode()) != 0:
            raise OSError(f"failed to save index to {path}")

    @classmethod
    def load(cls, path, *, lib_path=None):
        # Read the dim from the file header so the right library is loaded.
        dim = _read_tq_dim(path)
        lib = _native.load(dim, lib_path)[0]
        handle = lib.qj_index_load(str(path).encode())
        if not handle:
            raise OSError(f"failed to load index from {path}")
        self = cls(_handle=handle, _lib=lib)
        self._next_id = len(self)
        return self

    # --- lifecycle ---

    def _as_matrix(self, x, name):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"{name} must have shape (n, {self.dim})")
        return x

    def _context(self, m):
        if self._context_len != len(self):
            for ctx in self._contexts.values():
                self._lib.qj_context_destroy(ctx)
            self._contexts.clear()
            self._context_len = len(self)
        if m not in self._contexts:
            ctx = self._lib.qj_context_create(self._handle, m)
            if not ctx:
                raise MemoryError("qj_context_create failed")
            self._contexts[m] = ctx
        return self._contexts[m]

    def close(self):
        if getattr(self, "_handle", None):
            for ctx in self._contexts.values():
                self._lib.qj_context_destroy(ctx)
            self._contexts.clear()
            self._lib.qj_index_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


def _read_tq_dim(path):
    # .tq header: magic(4) u32 dim ...  (little-endian)
    with open(path, "rb") as f:
        head = f.read(8)
    if len(head) < 8 or head[:3] != b"TQX":
        raise OSError(f"{path} is not a quantajump index")
    return int.from_bytes(head[4:8], "little")
