"""LlamaIndex vector store backed by quantal — a one-line swap for the
in-memory store:

    from quantal.llama_index import QuantalVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    store = QuantalVectorStore()
    ctx = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex(nodes, storage_context=ctx, embed_model=embed)
    results = index.as_retriever(similarity_top_k=5).retrieve("query")

LlamaIndex populates each node's `.embedding` upstream, so this store only
handles vectors + node payloads. Vectors are L2-normalized, so similarities
are cosine in [-1, 1]. The dimension is inferred from the first node.
"""

from typing import Any, List, Sequence

import numpy as np

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import BaseNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.vector_stores.utils import (
        metadata_dict_to_node,
        node_to_metadata_dict,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "QuantalVectorStore needs llama-index-core: pip install llama-index-core"
    ) from e

from .index import Index


def _normalize(vec):
    a = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / max(n, 1e-30)


class QuantalVectorStore(BasePydanticVectorStore):
    stores_text: bool = True

    _m: int = PrivateAttr()
    _lib_path: Any = PrivateAttr()
    _index: Any = PrivateAttr(default=None)
    _by_node: dict = PrivateAttr(default_factory=dict)   # node_id -> internal id
    _payload: dict = PrivateAttr(default_factory=dict)   # internal id -> metadata dict
    _by_ref: dict = PrivateAttr(default_factory=dict)    # ref_doc_id -> set(node_id)

    def __init__(self, m: int = 128, lib_path: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._m = m
        self._lib_path = lib_path
        self._index = None
        self._by_node = {}
        self._payload = {}
        self._by_ref = {}

    @property
    def client(self) -> Any:
        return self._index

    def _ensure_index(self, dim: int):
        if self._index is None:
            self._index = Index(dim=dim, lib_path=self._lib_path)
        return self._index

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        if not nodes:
            return []
        vecs = np.stack([_normalize(n.get_embedding()) for n in nodes])
        index = self._ensure_index(vecs.shape[1])
        ids = index.add(vecs)
        out = []
        for node, internal in zip(nodes, ids.tolist()):
            self._by_node[node.node_id] = internal
            self._payload[internal] = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
            if node.ref_doc_id is not None:
                self._by_ref.setdefault(node.ref_doc_id, set()).add(node.node_id)
            out.append(node.node_id)
        return out

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        for node_id in self._by_ref.pop(ref_doc_id, set()):
            internal = self._by_node.pop(node_id, None)
            if internal is not None:
                self._index.remove(internal)
                self._payload.pop(internal, None)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if self._index is None or query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        k = query.similarity_top_k or 10
        q = _normalize(query.query_embedding)

        # Honor an explicit node_id restriction via exact allowlist search.
        if query.node_ids:
            allow = [self._by_node[nid] for nid in query.node_ids if nid in self._by_node]
            hits = self._index.search_filtered(q, allow, k=k, m=self._m)
        else:
            hits = self._index.search(q, k=k, m=self._m)

        nodes, sims, ids = [], [], []
        for internal, score in hits:
            meta = self._payload.get(internal)
            if meta is None:
                continue
            node = metadata_dict_to_node(meta)
            nodes.append(node)
            sims.append(score)
            ids.append(node.node_id)
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)
