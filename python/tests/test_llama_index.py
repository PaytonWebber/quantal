import numpy as np
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import VectorStoreQuery
from quantal.llama_index import QuantalVectorStore

def emb(text, dim=32):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim).astype(np.float32).tolist()

texts = ["the cat sat", "a dog barked", "quantum entanglement", "ottawa is canada's capital"]
nodes = []
for i, t in enumerate(texts):
    n = TextNode(text=t, id_=f"n{i}", metadata={"src": f"doc{i}"})
    n.embedding = emb(t)
    n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"ref{i}")
    nodes.append(n)

store = QuantalVectorStore()
ids = store.add(nodes)
assert ids == ["n0","n1","n2","n3"], ids
print("added", ids, "| index dim", store.client.dim, "rb", store.client.routing_bits)

# query with an exact embedding -> that node first, cosine ~1
res = store.query(VectorStoreQuery(query_embedding=emb("quantum entanglement"), similarity_top_k=3))
assert res.nodes[0].get_content() == "quantum entanglement", res.nodes[0].get_content()
assert res.nodes[0].metadata["src"] == "doc2"
assert res.similarities[0] > 0.99, res.similarities[0]
assert res.ids[0] == "n2"
print("query top:", repr(res.nodes[0].get_content()), round(res.similarities[0],4), res.ids[0])

# node_id-restricted query (exact allowlist path)
res2 = store.query(VectorStoreQuery(query_embedding=emb("quantum entanglement"),
                                    similarity_top_k=3, node_ids=["n0","n1"]))
assert set(res2.ids) <= {"n0","n1"} and "n2" not in res2.ids
print("restricted query ids:", res2.ids)

# delete by ref_doc_id
store.delete("ref2")
res3 = store.query(VectorStoreQuery(query_embedding=emb("quantum entanglement"), similarity_top_k=5))
assert "n2" not in res3.ids
print("delete by ref_doc_id: OK")
print("ALL LLAMAINDEX TESTS PASSED")
