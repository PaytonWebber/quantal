import tempfile, os
import numpy as np
from langchain_core.embeddings import Embeddings
from quantal.langchain import QuantalVectorStore

# Deterministic 32-d embedding: hash each text to a seed -> fixed vector.
class FakeEmb(Embeddings):
    dim = 32
    def _vec(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self.dim).astype(np.float32).tolist()
    def embed_documents(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)

emb = FakeEmb()
texts = ["the cat sat", "a dog barked", "quantum entanglement", "vector search is fast", "ottawa is in canada"]
metas = [{"src": f"doc{i}"} for i in range(len(texts))]

vs = QuantalVectorStore.from_texts(texts, emb, metadatas=metas)
print("built store, len", len(vs._docs))

# Querying with an exact stored text must return that doc first.
docs = vs.similarity_search("quantum entanglement", k=3)
assert docs[0].page_content == "quantum entanglement", docs[0].page_content
assert docs[0].metadata["src"] == "doc2"
print("similarity_search top:", repr(docs[0].page_content), docs[0].metadata)

# with score
ds = vs.similarity_search_with_score("a dog barked", k=2)
assert ds[0][0].page_content == "a dog barked"
assert ds[0][1] > 0.99, ds[0][1]   # cosine self-similarity ~1
print("with_score top:", round(ds[0][1], 4))

# delete by id (the metadata carries the engine id)
target_id = ds[0][0].metadata["id"]
assert vs.delete([target_id])
after = vs.similarity_search("a dog barked", k=5)
assert all(d.page_content != "a dog barked" for d in after)
print("delete: OK")

# add more, then persistence round-trip
vs.add_texts(["new fact one", "new fact two"], metadatas=[{"src":"n1"},{"src":"n2"}])
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "store.qj")
    vs.save(p)
    assert os.path.exists(p + ".tq") and os.path.exists(p + ".json")
    re = QuantalVectorStore.load(p, emb)
    r = re.similarity_search("quantum entanglement", k=1)
    assert r[0].page_content == "quantum entanglement"
    assert r[0].metadata["src"] == "doc2"
print("save/load round-trip: OK")

# retriever interface (what RAG chains actually call)
retr = vs.as_retriever(search_kwargs={"k": 2})
got = retr.invoke("vector search is fast")
assert got[0].page_content == "vector search is fast"
print("as_retriever().invoke: OK")
print("ALL LANGCHAIN TESTS PASSED")
