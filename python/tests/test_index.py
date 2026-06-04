import sys, tempfile, os
import numpy as np

# --- low-level Index, build-on-demand for a fresh dim (exercises zig build) ---
from quantajump import Index
rng = np.random.default_rng(0)
idx = Index(dim=32)                      # triggers build+cache for dim=32
print("Index dim", idx.dim, "routing_bits", idx.routing_bits)
vecs = rng.standard_normal((500, 32)).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
ids = idx.add(vecs)                      # auto-ids
assert len(idx) == 500 and ids[0] == 0
hits = idx.search(vecs[7], k=5)          # list of (id, score)
assert hits[0][0] == 7 and abs(hits[0][1] - 1.0) < 1e-2, hits[0]
assert idx.remove(7)
assert len(idx) == 499
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "x.tq")
    idx.save(p)
    re = Index.load(p)
    assert len(re) == 499
    # self-query of a surviving vector still returns it
    assert re.search(vecs[8], k=3)[0][0] == 8
print("low-level Index: OK")

# --- context manager ---
with Index(dim=32) as i2:
    i2.add(vecs[:10])
    assert len(i2) == 10
print("context manager: OK")
print("ALL LOW-LEVEL TESTS PASSED")
