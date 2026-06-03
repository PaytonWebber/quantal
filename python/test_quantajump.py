import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from quantajump import Index

rng = np.random.default_rng(42)
dim = 64
n = 5000

vectors = rng.standard_normal((n, dim), dtype=np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
ids = np.arange(1000, 1000 + n, dtype=np.uint64)

index = Index("zig-out/lib/libquantajump.so")
assert index.dim == dim
index.add(ids, vectors, threads=8)
assert len(index) == n
print(f"indexed {len(index)} vectors")

# batch search: self-queries must return themselves first
scores, found, counts = index.search(vectors[:100], k=5, m=64, threads=8)
self_hits = (found[:, 0] == ids[:100]).mean()
print(f"self-recall@1: {self_hits:.2f}, top score: {scores[0, 0]:.4f}")
assert self_hits >= 0.95, self_hits
assert abs(scores[0, 0] - 1.0) < 0.01

# exact-vs-numpy agreement on one query
q = rng.standard_normal(dim).astype(np.float32)
q /= np.linalg.norm(q)
exact_top = ids[np.argmax(vectors @ q)]
s, f, c = index.search(q, k=10, m=512)
print(f"true NN {exact_top} -> found rank", list(f[0]).index(exact_top) if exact_top in f[0] else "miss")
assert exact_top in f[0]

# filtered search
allow = np.array([1003, 1017, 1042, 1099, 999999], dtype=np.uint64)
fs, fi = index.search_filtered(q, allow, k=10)
print("filtered ids:", fi.tolist())
assert set(fi) <= {1003, 1017, 1042, 1099} and len(fi) == 4
assert list(fs) == sorted(fs, reverse=True)

# remove
assert index.remove(1003)
assert not index.remove(1003)
assert len(index) == n - 1
fs2, fi2 = index.search_filtered(q, allow, k=10)
assert 1003 not in fi2 and len(fi2) == 3
print("remove + filtered-after-remove ok")

# save / load roundtrip
index.save("/tmp/qj_py_test.tq")
loaded = Index.load("/tmp/qj_py_test.tq")
assert len(loaded) == n - 1
s1, f1, _ = index.search(vectors[:20], k=5, m=64)
s2, f2, _ = loaded.search(vectors[:20], k=5, m=64)
assert (f1 == f2).all() and np.allclose(s1, s2)
print("save/load roundtrip ok")
os.remove("/tmp/qj_py_test.tq")

print("ALL PYTHON BINDING TESTS PASSED")
