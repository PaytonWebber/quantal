"""ann-benchmarks adapter for quantal.

Drop this directory into `ann_benchmarks/algorithms/quantal/` in a clone of
github.com/erikbern/ann-benchmarks (alongside the Dockerfile and config.yml
here), then:

    python install.py --algorithm quantal
    python run.py --algorithm quantal --dataset glove-100-angular

The Dockerfile builds the shared library for the dataset's dimension; the
wrapper loads it through the same ctypes layer shipped in python/quantal.py.
"""

import os

import numpy as np

try:
    from ann_benchmarks.algorithms.base.module import BaseANN
except ImportError:  # allow standalone import (local harness / tests)
    class BaseANN:
        pass

# python/quantal.py is copied next to this file by the Dockerfile.
from quantal import Index


class QuantaJump(BaseANN):
    def __init__(self, metric, dim, ef_construction=200):
        if metric not in ("angular", "dot"):
            raise NotImplementedError(f"quantal: unsupported metric {metric}")
        self._metric = metric
        self._dim = dim
        self._ef = ef_construction
        self._m = 128
        # The Dockerfile builds one shared library per supported dimension
        # (quantal fixes dim at compile time); pick the matching one.
        lib_dir = os.environ.get("QJ_LIB_DIR", "/home/app/lib")
        self._lib = os.path.join(lib_dir, f"libquantal-{dim}.so")
        self._index = None

    def _prep(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self._metric == "angular":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.maximum(norms, 1e-30)
        return X

    def fit(self, X):
        X = self._prep(X)
        self._index = Index(lib_path=self._lib, ef_construction=self._ef)
        self._index.add(X)  # auto-assigns ids 0..n-1, uses all cores

    def set_query_arguments(self, m):
        self._m = int(m)

    def query(self, v, n):
        v = self._prep(v.reshape(1, -1))
        _, ids, counts = self._index.search_batch(v, k=n, m=self._m, threads=1)
        return ids[0, : counts[0]]

    def batch_query(self, X, n):
        X = self._prep(X)
        scores, ids, counts = self._index.search_batch(X, k=n, m=self._m, threads=os.cpu_count())
        self._batch = (ids, counts)

    def get_batch_results(self):
        ids, counts = self._batch
        return [row[:c] for row, c in zip(ids, counts)]

    def __str__(self):
        return f"QuantaJump(ef={self._ef}, m={self._m})"
