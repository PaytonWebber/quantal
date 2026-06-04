"""Same-machine turbovec comparison on identical fvecs data.

Environment overrides (defaults reproduce the 100k run in RESULTS.md):
    QJ_BASE=data/dbpedia1536_base.fvecs   base vectors
    QJ_QUERY=data/dbpedia1536_query.fvecs query vectors
    QJ_N=100000                            base vector count
    QJ_NQ=1000                             query count
    QJ_BITS=2,4                            turbovec bit widths to run
    RAYON_NUM_THREADS=1                    single-threaded turbovec

The 1M run (see RUN_1M.md):
    QJ_BASE=data/dbpedia1536_1m_base.fvecs QJ_QUERY=data/dbpedia1536_1m_query.fvecs \
    QJ_N=999000 python3 benchmarks/turbovec_compare.py
"""

import os
import time

import numpy as np


def load_fvecs(path, max_n):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    raw = raw.reshape(-1, dim + 1)[:max_n]
    assert (raw[:, 0] == dim).all()
    return raw[:, 1:].view(np.float32).copy()


base_path = os.environ.get("QJ_BASE", "data/dbpedia1536_base.fvecs")
query_path = os.environ.get("QJ_QUERY", "data/dbpedia1536_query.fvecs")
n_base = int(os.environ.get("QJ_N", "100000"))
n_queries = int(os.environ.get("QJ_NQ", "1000"))
bit_widths = [int(b) for b in os.environ.get("QJ_BITS", "2,4").split(",")]

base = load_fvecs(base_path, n_base)
queries = load_fvecs(query_path, n_queries)
base /= np.linalg.norm(base, axis=1, keepdims=True)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
print(f"base {base.shape} queries {queries.shape}", flush=True)

# Exact ground truth top-1, blocked over queries to bound the matmul size.
gt = np.empty(len(queries), dtype=np.int64)
t0 = time.perf_counter()
for i in range(0, len(queries), 100):
    gt[i:i + 100] = (queries[i:i + 100] @ base.T).argmax(axis=1)
print(f"ground truth: {time.perf_counter() - t0:.1f}s", flush=True)

from turbovec import TurboQuantIndex

for bits in bit_widths:
    idx = TurboQuantIndex(dim=base.shape[1], bit_width=bits)
    t0 = time.perf_counter()
    idx.add(base)
    t_build = time.perf_counter() - t0

    # batch search (turbovec's preferred path; honours RAYON_NUM_THREADS)
    t0 = time.perf_counter()
    scores, ids = idx.search(queries, 64)
    t_batch = time.perf_counter() - t0

    # per-query loop latency
    t0 = time.perf_counter()
    for q in queries[:200]:
        idx.search(q.reshape(1, -1), 64)
    t_loop = (time.perf_counter() - t0) / 200

    recalls = {}
    for k in (1, 2, 4, 8, 16, 32, 64):
        recalls[k] = float((ids[:, :k] == gt[:, None]).any(axis=1).mean())
    print(f"bits={bits} build={t_build:.2f}s batch={t_batch * 1e3 / len(queries):.3f}ms/q "
          f"loop={t_loop * 1e3:.3f}ms/q")
    print("  recall1@k:", {k: round(v, 4) for k, v in recalls.items()}, flush=True)
