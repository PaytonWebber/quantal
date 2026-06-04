# quantajump (Python)

Embedded, in-memory vector index — 1-bit graph routing + 3-bit TurboQuant
payloads + exact rerank — with a ctypes binding to the Zig core.

## Install

```bash
pip install -e python      # from a source checkout (builds libs on demand)
```

quantajump fixes the vector dimension at compile time, so there is one native
library per dimension. The binding resolves it automatically:

- **Source checkout + `zig` on PATH** (recommended for dev): `Index(dim=N)`
  builds `libquantajump` for that dimension on first use and caches it under
  `~/.cache/quantajump/`.
- **Prebuilt library**: build once with
  `zig build -Doptimize=ReleaseFast -Dc-dim=N` and set
  `QUANTAJUMP_LIB=zig-out/lib/libquantajump.so`.

(Shipping prebuilt per-dimension wheels via cibuildwheel is the remaining
packaging step.)

## Core API

```python
import numpy as np
from quantajump import Index

with Index(dim=384) as index:
    ids = index.add(vectors)               # (n, 384) float32 -> auto ids
    hits = index.search(query, k=10)        # -> [(id, score), ...] cosine
    scores, ids, counts = index.search_batch(queries, k=10, threads=0)
    index.remove(ids[0])
    index.save("docs.tq")

index = Index.load("docs.tq")               # dimension read from the file
```

`search_filtered(query, allowlist, k)` restricts results to a set of ids
(exact scoring over the allowlist — for tenant/ACL filtering).

## LangChain

A one-line swap for the in-memory / FAISS store:

```python
from quantajump.langchain import QuantajumpVectorStore

vs = QuantajumpVectorStore.from_texts(texts, embedding=my_embeddings,
                                      metadatas=metas)
docs = vs.similarity_search("query", k=5)
retriever = vs.as_retriever(search_kwargs={"k": 5})

vs.save("store.qj")                         # store.qj.tq + store.qj.json
vs = QuantajumpVectorStore.load("store.qj", embedding=my_embeddings)
```

Vectors are L2-normalized in and out, so scores are cosine similarity. The
engine stores vectors keyed by id; documents and metadata live in a JSON
sidecar persisted next to the `.tq`.

## Tests

```bash
QUANTAJUMP_LIB=... python python/tests/test_index.py
python python/tests/test_langchain.py      # needs langchain-core
```
