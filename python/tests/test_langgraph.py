import numpy as np
from quantajump.langgraph_store import QuantajumpStore

def embed(texts, dim=32):
    out = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        out.append(rng.standard_normal(dim).astype(np.float32).tolist())
    return out

store = QuantajumpStore(index={"dims": 32, "embed": embed, "fields": ["text"]})

# put across namespaces
store.put(("mem", "alice"), "m1", {"text": "prefers dark mode", "kind": "pref"})
store.put(("mem", "alice"), "m2", {"text": "lives in Ottawa", "kind": "fact"})
store.put(("mem", "bob"),   "m1", {"text": "prefers dark mode", "kind": "pref"})

# get
it = store.get(("mem", "alice"), "m1")
assert it.value["text"] == "prefers dark mode" and it.namespace == ("mem","alice")
print("get:", it.value)

# semantic search scoped to alice — bob's identical memory must NOT appear
hits = store.search(("mem", "alice"), query="prefers dark mode", limit=5)
assert all(h.namespace == ("mem","alice") for h in hits), [h.namespace for h in hits]
assert hits[0].key == "m1" and hits[0].score > 0.99, (hits[0].key, hits[0].score)
print("scoped semantic search top:", hits[0].key, round(hits[0].score, 4), "| n_hits", len(hits))

# filter on a value field
hits_f = store.search(("mem", "alice"), query="lives in Ottawa", filter={"kind": "fact"}, limit=5)
assert all(h.value["kind"] == "fact" for h in hits_f)
print("filtered search keys:", [h.key for h in hits_f])

# update re-indexes
store.put(("mem","alice"), "m1", {"text": "switched to light mode", "kind": "pref"})
h2 = store.search(("mem","alice"), query="light mode", limit=1)
assert h2[0].key == "m1" and h2[0].value["text"] == "switched to light mode"
print("update re-index: OK")

# delete
store.delete(("mem","bob"), "m1")
assert store.get(("mem","bob"), "m1") is None
print("delete: OK")

# list_namespaces
ns = store.list_namespaces()
assert ("mem","alice") in ns and ("mem","bob") not in [n for n in ns if store.get(n,"m1")]
print("list_namespaces:", ns)

# no-query search (recency) + non-indexed plain KV mode
plain = QuantajumpStore()  # no index config -> pure KV
plain.put(("cfg",), "k", {"v": 1})
assert plain.get(("cfg",), "k").value == {"v": 1}
print("plain KV mode: OK")
print("ALL LANGGRAPH TESTS PASSED")
