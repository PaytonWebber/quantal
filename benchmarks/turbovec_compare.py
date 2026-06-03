import numpy as np, time, sys, os

def load_fvecs(path, max_n):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    raw = raw.reshape(-1, dim + 1)[:max_n]
    assert (raw[:, 0] == dim).all()
    return raw[:, 1:].view(np.float32).copy()

base = load_fvecs("data/dbpedia1536_base.fvecs", 100_000)
queries = load_fvecs("data/dbpedia1536_query.fvecs", 1_000)
base /= np.linalg.norm(base, axis=1, keepdims=True)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
print(f"base {base.shape} queries {queries.shape}", flush=True)

# exact ground truth top-1 (blocked)
gt = np.empty(len(queries), dtype=np.int64)
for i in range(0, len(queries), 100):
    gt[i:i+100] = (queries[i:i+100] @ base.T).argmax(axis=1)

from turbovec import TurboQuantIndex
for bits in (2, 4):
    idx = TurboQuantIndex(dim=1536, bit_width=bits)
    t0 = time.perf_counter()
    idx.add(base)
    t_build = time.perf_counter() - t0

    # batch search
    t0 = time.perf_counter()
    scores, ids = idx.search(queries, 64)
    t_batch = time.perf_counter() - t0

    # per-query loop
    t0 = time.perf_counter()
    for q in queries[:200]:
        idx.search(q.reshape(1, -1), 64)
    t_loop = (time.perf_counter() - t0) / 200

    recalls = {}
    for k in (1, 2, 4, 8, 16, 32, 64):
        recalls[k] = float((ids[:, :k] == gt[:, None]).any(axis=1).mean())
    print(f"bits={bits} build={t_build:.2f}s batch={t_batch*1e3/len(queries):.3f}ms/q loop={t_loop*1e3:.3f}ms/q")
    print("  recall1@k:", {k: round(v, 4) for k, v in recalls.items()}, flush=True)
