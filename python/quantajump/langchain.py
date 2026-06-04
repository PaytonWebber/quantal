"""LangChain VectorStore backed by quantajump — a one-line swap for the
in-memory / FAISS store:

    from quantajump.langchain import QuantajumpVectorStore
    vs = QuantajumpVectorStore.from_texts(texts, embedding=my_embeddings)
    docs = vs.similarity_search("query", k=5)
    vs.save("store.qj")               # vectors (.qj.tq) + docs (.qj.json)
    vs = QuantajumpVectorStore.load("store.qj", embedding=my_embeddings)

quantajump stores vectors keyed by integer id; the documents and metadata
live in a Python sidecar persisted alongside the .tq file. Distances use
cosine similarity (vectors are L2-normalized on the way in and out), so the
returned score is cosine similarity in [-1, 1], higher = closer.
"""

import json

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "QuantajumpVectorStore needs langchain-core: pip install quantajump[langchain]"
    ) from e

from .index import Index


def _normalize(m):
    m = np.ascontiguousarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-30)


class QuantajumpVectorStore(VectorStore):
    """A drop-in LangChain VectorStore over the embedded quantajump engine."""

    def __init__(self, embedding, *, dim=None, index=None, m=128, lib_path=None):
        self._embedding = embedding
        self._m = m
        self._index = index if index is not None else Index(dim=dim, lib_path=lib_path)
        # id -> (text, metadata); the engine only stores vectors.
        self._docs = {}

    @property
    def embeddings(self):
        return self._embedding

    # --- ingest ---

    def add_texts(self, texts, metadatas=None, **kwargs):
        texts = list(texts)
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        vectors = _normalize(self._embedding.embed_documents(texts))
        ids = self._index.add(vectors)
        out = []
        for i, text, meta in zip(ids.tolist(), texts, metadatas):
            self._docs[i] = (text, dict(meta))
            out.append(str(i))
        return out

    def delete(self, ids=None, **kwargs):
        if ids is None:
            return False
        ok = True
        for sid in ids:
            i = int(sid)
            ok = self._index.remove(i) and ok
            self._docs.pop(i, None)
        return ok

    # --- query ---

    def similarity_search(self, query, k=4, **kwargs):
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def similarity_search_with_score(self, query, k=4, **kwargs):
        vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(vector, k=k, **kwargs)

    def similarity_search_by_vector(self, embedding, k=4, **kwargs):
        return [doc for doc, _ in self.similarity_search_by_vector_with_score(embedding, k=k, **kwargs)]

    def similarity_search_by_vector_with_score(self, embedding, k=4, *, allowlist=None, **kwargs):
        q = _normalize(embedding)
        if allowlist is not None:
            hits = self._index.search_filtered(q, [int(x) for x in allowlist], k=k, m=self._m)
        else:
            hits = self._index.search(q, k=k, m=self._m)
        results = []
        for i, score in hits:
            text, meta = self._docs.get(i, ("", {}))
            results.append((Document(page_content=text, metadata={**meta, "id": i}), float(score)))
        return results

    # --- construction / persistence ---

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, *, dim=None, m=128, lib_path=None, **kwargs):
        texts = list(texts)
        if dim is None:
            # Infer dimension from the embedding model itself.
            dim = len(embedding.embed_query(texts[0] if texts else ""))
        store = cls(embedding, dim=dim, m=m, lib_path=lib_path)
        store.add_texts(texts, metadatas=metadatas)
        return store

    def save(self, path):
        """Persists vectors to <path>.tq and documents to <path>.json."""
        self._index.save(f"{path}.tq")
        with open(f"{path}.json", "w") as f:
            json.dump({"m": self._m, "docs": {str(i): d for i, d in self._docs.items()}}, f)

    @classmethod
    def load(cls, path, embedding, *, lib_path=None):
        index = Index.load(f"{path}.tq", lib_path=lib_path)
        store = cls(embedding, index=index)
        with open(f"{path}.json") as f:
            blob = json.load(f)
        store._m = blob.get("m", 128)
        store._docs = {int(i): (t, meta) for i, (t, meta) in blob["docs"].items()}
        return store
