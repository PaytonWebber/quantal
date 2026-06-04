"""LangGraph BaseStore backed by quantajump — local, fast agent memory with
semantic search:

    from quantajump.langgraph_store import QuantajumpStore

    store = QuantajumpStore(index={"dims": 384, "embed": embed_fn, "fields": ["text"]})
    store.put(("memories", "alice"), "m1", {"text": "prefers dark mode"})
    hits = store.search(("memories", "alice"), query="ui preferences", limit=5)

`embed` is a callable list[str] -> list[list[float]] (a LangChain Embeddings
object also works). Per-namespace semantic search is served by quantajump's
exact allowlist scoring, so namespace isolation is precise and adds no recall
loss. Without an index config the store is a plain namespaced key-value store.
TTL is accepted but not enforced (this is an in-memory store).
"""

import json
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        Item,
        ListNamespacesOp,
        PutOp,
        SearchItem,
        SearchOp,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("QuantajumpStore needs langgraph: pip install langgraph") from e

from .index import Index


def _now():
    return datetime.now(timezone.utc)


def _normalize(vec):
    a = np.asarray(vec, dtype=np.float32)
    return a / max(float(np.linalg.norm(a)), 1e-30)


class QuantajumpStore(BaseStore):
    supports_ttl = False

    def __init__(self, index=None, *, lib_path=None, m=128):
        self._cfg = index  # {"dims", "embed", "fields"} or None
        self._m = m
        self._lib_path = lib_path
        self._engine = None
        self._items = {}        # (namespace, key) -> dict(value, created_at, updated_at, iid)
        self._iid_to_loc = {}   # internal id -> (namespace, key)

    # --- embedding helpers ---

    def _embed(self, texts):
        embed = self._cfg["embed"]
        fn = getattr(embed, "embed_documents", None)
        vecs = fn(texts) if fn else embed(texts)
        return [_normalize(v) for v in vecs]

    def _embed_query(self, text):
        embed = self._cfg["embed"]
        fn = getattr(embed, "embed_query", None)
        return _normalize(fn(text) if fn else embed([text])[0])

    def _text_for(self, value, index_spec):
        fields = index_spec if isinstance(index_spec, (list, tuple)) else self._cfg.get("fields", ["$"])
        if fields == ["$"] or fields == "$":
            return json.dumps(value, sort_keys=True, default=str)
        parts = [str(value[f]) for f in fields if f in value]
        return " ".join(parts)

    def _engine_for(self):
        if self._engine is None:
            self._engine = Index(dim=int(self._cfg["dims"]), lib_path=self._lib_path)
        return self._engine

    # --- the single required primitive ---

    def batch(self, ops: Iterable[Any]) -> list:
        results = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                results.append(self._put(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:  # unknown op type
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Any]) -> list:
        return self.batch(list(ops))

    # --- op handlers ---

    def _get(self, op: GetOp):
        rec = self._items.get((op.namespace, op.key))
        if rec is None:
            return None
        return Item(value=rec["value"], key=op.key, namespace=op.namespace,
                    created_at=rec["created_at"], updated_at=rec["updated_at"])

    def _put(self, op: PutOp):
        loc = (op.namespace, op.key)
        if op.value is None:  # delete
            rec = self._items.pop(loc, None)
            if rec and rec.get("iid") is not None:
                self._engine.remove(rec["iid"])
                self._iid_to_loc.pop(rec["iid"], None)
            return None

        prev = self._items.get(loc)
        rec = {
            "value": op.value,
            "created_at": prev["created_at"] if prev else _now(),
            "updated_at": _now(),
            "iid": prev.get("iid") if prev else None,
        }
        # (Re)index the vector when configured and not opted out (index=False).
        if self._cfg is not None and op.index is not False:
            if rec["iid"] is not None:  # replace: drop the old vector
                self._engine.remove(rec["iid"])
                self._iid_to_loc.pop(rec["iid"], None)
            vec = self._embed([self._text_for(op.value, op.index)])[0]
            iid = int(self._engine_for().add(vec.reshape(1, -1))[0])
            rec["iid"] = iid
            self._iid_to_loc[iid] = loc
        self._items[loc] = rec
        return None

    def _matches(self, value, flt):
        return not flt or all(value.get(k) == v for k, v in flt.items())

    def _under_prefix(self, namespace, prefix):
        return len(namespace) >= len(prefix) and namespace[: len(prefix)] == tuple(prefix)

    def _search(self, op: SearchOp):
        prefix = tuple(op.namespace_prefix)
        in_scope = [
            (loc, rec) for loc, rec in self._items.items()
            if self._under_prefix(loc[0], prefix) and self._matches(rec["value"], op.filter or {})
        ]

        if op.query and self._cfg is not None and self._engine is not None:
            # Semantic: exact allowlist scoring over the in-scope vectors.
            allow = [rec["iid"] for _, rec in in_scope if rec.get("iid") is not None]
            by_iid = {rec["iid"]: (loc, rec) for loc, rec in in_scope if rec.get("iid") is not None}
            q = self._embed_query(op.query)
            k = (op.offset or 0) + (op.limit or 10)
            hits = self._engine.search_filtered(q, allow, k=k, m=self._m)
            ranked = [(by_iid[i][0], by_iid[i][1], score) for i, score in hits if i in by_iid]
        else:
            # No query: most-recently-updated first, no score.
            in_scope.sort(key=lambda lr: lr[1]["updated_at"], reverse=True)
            ranked = [(loc, rec, None) for loc, rec in in_scope]

        start = op.offset or 0
        window = ranked[start: start + (op.limit or 10)]
        return [
            SearchItem(namespace=loc[0], key=loc[1], value=rec["value"],
                       created_at=rec["created_at"], updated_at=rec["updated_at"], score=score)
            for loc, rec, score in window
        ]

    def _list_namespaces(self, op: ListNamespacesOp):
        seen = set()
        for (namespace, _key) in self._items:
            ns = namespace[: op.max_depth] if op.max_depth is not None else namespace
            if all(self._match_condition(ns, mc) for mc in (op.match_conditions or ())):
                seen.add(ns)
        out = sorted(seen)
        start = op.offset or 0
        return out[start: start + (op.limit or len(out))]

    @staticmethod
    def _match_condition(namespace, mc):
        path = tuple(mc.path)
        if mc.match_type == "prefix":
            return len(namespace) >= len(path) and all(
                p == "*" or p == n for p, n in zip(path, namespace[: len(path)])
            )
        if mc.match_type == "suffix":
            return len(namespace) >= len(path) and all(
                p == "*" or p == n for p, n in zip(path, namespace[-len(path):])
            )
        return True
