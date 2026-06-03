"""ctypes bindings for the quantajump vector index.

The shared library is compiled for a fixed dimension:

    zig build -Doptimize=ReleaseFast -Dc-dim=1536

Usage:

    import numpy as np
    from quantajump import Index

    index = Index("zig-out/lib/libquantajump.so")
    index.add(ids, vectors)                      # uint64 ids, float32 (n, dim)
    scores, ids, counts = index.search(queries, k=10, m=128, threads=12)
    scores, ids = index.search_filtered(query, allowlist, k=10)
    index.remove(1002)
    index.save("my.tq")
    loaded = Index.load("my.tq", "zig-out/lib/libquantajump.so")
"""

import ctypes

import numpy as np

_u64p = ctypes.POINTER(ctypes.c_uint64)
_f32p = ctypes.POINTER(ctypes.c_float)
_usizep = ctypes.POINTER(ctypes.c_size_t)


def _load_lib(path):
    lib = ctypes.CDLL(path)
    sigs = {
        "qj_dim": ([], ctypes.c_size_t),
        "qj_index_create": ([ctypes.c_size_t, ctypes.c_uint64], ctypes.c_void_p),
        "qj_index_destroy": ([ctypes.c_void_p], None),
        "qj_index_add": ([ctypes.c_void_p, ctypes.c_uint64, _f32p], ctypes.c_int32),
        "qj_index_add_batch": (
            [ctypes.c_void_p, _u64p, _f32p, ctypes.c_size_t, ctypes.c_size_t],
            ctypes.c_int32,
        ),
        "qj_index_remove": ([ctypes.c_void_p, ctypes.c_uint64], ctypes.c_int32),
        "qj_index_len": ([ctypes.c_void_p], ctypes.c_size_t),
        "qj_index_save": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int32),
        "qj_index_load": ([ctypes.c_char_p], ctypes.c_void_p),
        "qj_context_create": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_void_p),
        "qj_context_destroy": ([ctypes.c_void_p], None),
        "qj_search": (
            [ctypes.c_void_p, ctypes.c_void_p, _f32p, ctypes.c_size_t, _u64p, _f32p],
            ctypes.c_size_t,
        ),
        "qj_search_filtered": (
            [ctypes.c_void_p, ctypes.c_void_p, _f32p, _u64p, ctypes.c_size_t,
             ctypes.c_size_t, _u64p, _f32p],
            ctypes.c_size_t,
        ),
        "qj_search_batch": (
            [ctypes.c_void_p, _f32p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
             ctypes.c_size_t, _u64p, _f32p, _usizep],
            ctypes.c_int32,
        ),
    }
    for name, (argtypes, restype) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def _as_f32(array, dim, name):
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"{name} must have shape (n, {dim})")
    return array


class Index:
    """A two-stage cascading vector index (1-bit graph + 3-bit payloads +
    exact rerank). The library dimension is fixed at build time."""

    def __init__(self, lib_path="zig-out/lib/libquantajump.so", ef_construction=200,
                 seed=42, _handle=None, _lib=None):
        self._lib = _lib or _load_lib(lib_path)
        self.dim = int(self._lib.qj_dim())
        if _handle is not None:
            self._handle = _handle
        else:
            self._handle = self._lib.qj_index_create(ef_construction, seed)
            if not self._handle:
                raise MemoryError("qj_index_create failed")
        self._contexts = {}  # m -> context handle (recreated when index grows)
        self._context_len = 0

    @classmethod
    def load(cls, path, lib_path="zig-out/lib/libquantajump.so"):
        lib = _load_lib(lib_path)
        handle = lib.qj_index_load(path.encode())
        if not handle:
            raise OSError(f"failed to load index from {path}")
        return cls(_handle=handle, _lib=lib)

    def __len__(self):
        return int(self._lib.qj_index_len(self._handle))

    def add(self, ids, vectors, threads=0):
        import os
        vectors = _as_f32(vectors, self.dim, "vectors")
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        if ids.shape != (vectors.shape[0],):
            raise ValueError("ids must be a 1-D uint64 array matching vectors")
        threads = threads or os.cpu_count() or 1
        rc = self._lib.qj_index_add_batch(
            self._handle,
            ids.ctypes.data_as(_u64p),
            vectors.ctypes.data_as(_f32p),
            vectors.shape[0],
            threads,
        )
        if rc != 0:
            raise RuntimeError("qj_index_add_batch failed (duplicate id?)")

    def remove(self, id_):
        return self._lib.qj_index_remove(self._handle, int(id_)) == 0

    def search(self, queries, k=10, m=128, threads=1):
        """Returns (scores, ids, counts); rows are padded past counts[i]."""
        queries = _as_f32(queries, self.dim, "queries")
        n = queries.shape[0]
        ids = np.zeros((n, k), dtype=np.uint64)
        scores = np.zeros((n, k), dtype=np.float32)
        counts = np.zeros(n, dtype=np.uintp)
        rc = self._lib.qj_search_batch(
            self._handle,
            queries.ctypes.data_as(_f32p),
            n, k, m, threads,
            ids.ctypes.data_as(_u64p),
            scores.ctypes.data_as(_f32p),
            counts.ctypes.data_as(_usizep),
        )
        if rc != 0:
            raise RuntimeError("qj_search_batch failed")
        return scores, ids, counts.astype(np.int64)

    def search_filtered(self, query, allowlist, k=10, m=128):
        """Restricts results to the allowlist (exact scoring over it)."""
        query = _as_f32(query, self.dim, "query")
        if query.shape[0] != 1:
            raise ValueError("search_filtered takes a single query")
        allowlist = np.ascontiguousarray(allowlist, dtype=np.uint64)
        ctx = self._context(m)
        ids = np.zeros(k, dtype=np.uint64)
        scores = np.zeros(k, dtype=np.float32)
        count = self._lib.qj_search_filtered(
            self._handle, ctx,
            query.ctypes.data_as(_f32p),
            allowlist.ctypes.data_as(_u64p), allowlist.shape[0],
            k,
            ids.ctypes.data_as(_u64p),
            scores.ctypes.data_as(_f32p),
        )
        return scores[:count], ids[:count]

    def save(self, path):
        if self._lib.qj_index_save(self._handle, path.encode()) != 0:
            raise OSError(f"failed to save index to {path}")

    def _context(self, m):
        # Contexts are sized for the index at creation; recreate after growth.
        if self._context_len != len(self) or m not in self._contexts:
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

    def __del__(self):
        self.close()
