# quantal (Python)

Embedded, in-memory vector index with 1-bit graph routing, 3-bit TurboQuant
payloads, exact rerank, and a ctypes binding to the Zig core.

## Install

```bash
pip install -e python      # from a source checkout (builds libs on demand)
```

quantal fixes the vector dimension at compile time, so there is one native
library per dimension. The binding resolves it automatically, in order:

1. **`QUANTAL_LIB`**: an explicit library path (overrides everything).
2. **Bundled binary**: the published wheels carry prebuilt libraries for the
   common embedding dimensions (256, 384, 512, 768, 1024, 1536, 3072), so
   `pip install quantaldb` works with **no toolchain** for those dims.
3. **Cached build** under `~/.cache/quantal/`.
4. **Build on demand**: from a source checkout with `zig` on PATH,
   `Index(dim=N)` builds and caches the library for any other dimension.

Common embedding dimensions can use the bundled wheel; any other dimension
needs either a source checkout with auto-build or a manual
`zig build -Dc-dim=N` + `QUANTAL_LIB`.

### Building wheels

```bash
cd python
python build_libs.py                 # compile bundled dims into quantal/_libs/
python -m build --wheel              # -> dist/quantal-...-py3-none-<platform>.whl
```

`build_libs.py --target x86_64-linux-gnu.2.28` pins glibc so the Linux wheel
is manylinux-compatible. Zig cross-compiles directly, no Docker needed. CI in
`.github/workflows/wheels.yml` builds Linux/macOS/Windows wheels this way.

## Core API

```python
import numpy as np
from quantal import Index

with Index(dim=384) as index:
    ids = index.add(vectors)               # (n, 384) float32 -> auto ids
    hits = index.search(query, k=10)        # -> [(id, score), ...] cosine
    scores, ids, counts = index.search_batch(queries, k=10, threads=0)
    index.remove(ids[0])
    index.save("docs.tq")

index = Index.load("docs.tq")               # dimension read from the file
```

`search_filtered(query, allowlist, k)` restricts results to a set of ids. It
uses exact scoring over the allowlist for tenant and ACL filtering.

## LangChain

A one-line swap for the in-memory / FAISS store:

```python
from quantal.langchain import QuantalVectorStore

vs = QuantalVectorStore.from_texts(texts, embedding=my_embeddings,
                                      metadatas=metas)
docs = vs.similarity_search("query", k=5)
retriever = vs.as_retriever(search_kwargs={"k": 5})

vs.save("store.qj")                         # store.qj.tq + store.qj.json
vs = QuantalVectorStore.load("store.qj", embedding=my_embeddings)
```

Vectors are L2-normalized in and out, so scores are cosine similarity. The
engine stores vectors keyed by id; documents and metadata live in a JSON
sidecar persisted next to the `.tq`.

## LlamaIndex

```python
from quantal.llama_index import QuantalVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

store = QuantalVectorStore()                      # dim inferred from nodes
ctx = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex(nodes, storage_context=ctx, embed_model=embed)
hits = index.as_retriever(similarity_top_k=5).retrieve("query")
```

## LangGraph (agent memory)

A `BaseStore` with semantic search for local short/long-term memory in
agent graphs. Per-namespace search is served by quantal's exact allowlist
scoring, so namespaces are isolated precisely.

```python
from quantal.langgraph_store import QuantalStore

store = QuantalStore(index={"dims": 384, "embed": embed_fn, "fields": ["text"]})
store.put(("memories", "alice"), "m1", {"text": "prefers dark mode"})
hits = store.search(("memories", "alice"), query="ui preferences", limit=5)
```

Without an `index` config it is a plain namespaced key-value store.

## Tests

```bash
QUANTAL_LIB=... python python/tests/test_index.py
python python/tests/test_langchain.py      # needs langchain-core
```
