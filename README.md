# quantal

A small, fast vector index. It finds nearest neighbors in three stages: a
1-bit graph routes a query to a few hundred candidates, 3-bit TurboQuant codes
rerank those, and an exact pass settles the final order. The result is
sub-linear search that still returns the true nearest neighbors.

It's embedded and in-memory — a library you link, not a server you run (think
SQLite, not Pinecone) — written in Zig with a C ABI, Python bindings, and
drop-in LangChain / LlamaIndex / LangGraph stores. Built on Google Research's
[TurboQuant](https://arxiv.org/abs/2504.19874).

## Where it fits

quantal is built for high-dimensional embeddings (the 384–3072 range of modern
embedding models). There it wins on the recall/latency frontier because the
graph only ever looks at a small fraction of the data, while the exact rerank
keeps recall high. On low-dimensional data the routing code is auto-widened so
it stays competitive there too.

## Results

DBpedia (OpenAI `text-embedding-3-large`, d=1536), measured against
[turbovec](https://github.com/RyanCodrai/turbovec) on the same machine (a
Ryzen 7640U laptop), at matched recall:

| corpus | single-thread | multi-thread (12 cores) |
|--------|--------------|-------------------------|
| 100k vectors | ~5.7× faster | ~17.9× faster |
| 1M vectors | ~14× faster | ~48× faster |

The gap widens with corpus size because a flat scan is O(n) per query while
graph routing is roughly O(log n). Full methodology, the glove-100 result
(where it initially lost and the fix that followed), and the routing-bit
sweeps are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Install

No PyPI release yet. From a source checkout with [Zig](https://ziglang.org)
on your PATH:

```bash
pip install -e python
```

The native library is compiled per dimension and built on first use, so
`Index(dim=384)` just works. To use it without a toolchain, build a wheel
(`cd python && python build_libs.py && python -m build --wheel`) — it bundles
prebuilt libraries for the common embedding dimensions.

## Usage

Python:

```python
import numpy as np
from quantal import Index

index = Index(dim=768)
ids = index.add(vectors)                 # (n, 768) float32, auto-assigned ids
hits = index.search(query, k=10)         # -> [(id, score), ...], cosine
index.save("docs.tq")
```

Zig:

```zig
const quantal = @import("quantal");
var index = try quantal.Index(768, 16, 768).init(allocator, 200, 42);
```

Drop-in framework stores:

```python
from quantal.langchain import QuantalVectorStore        # swap for FAISS/InMemory
from quantal.llama_index import QuantalVectorStore
from quantal.langgraph_store import QuantalStore        # agent memory
```

## How it works

1. **Route (1-bit).** Each vector is preconditioned with a random rotation and
   reduced to a sign-bit (SimHash) code. An HNSW-style graph navigates these by
   Hamming distance — cheap, and enough to reach the right neighborhood.
2. **Rerank (3-bit).** Candidates are scored with 3-bit TurboQuant codes via a
   lookup-table kernel, with a per-vector scalar that keeps the inner-product
   estimate unbiased.
3. **Exact.** The top of that pool is rescored with the stored vectors (fp32 or
   int8), so quantization noise can't reorder the final results.

Queries are allocation-free: the search path takes a preallocated context and
never touches the allocator.

## Building (Zig)

```bash
zig build test                              # run the test suite
zig build bench -Doptimize=ReleaseFast -- synthetic --n 100000
zig build -Doptimize=ReleaseFast -Dc-dim=768   # build the C library + binaries
```

## Status

Research-grade and benchmarked on a single laptop; numbers should be taken as
directional, not a leaderboard. The C ABI, Python package, and framework
wrappers are tested; PyPI/crates publishing and the multi-platform CI wheels
are wired but not yet released.

## References

The methods quantal builds on:

- **TurboQuant** — A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni. *TurboQuant:
  Online Vector Quantization with Near-optimal Distortion Rate*, 2025.
  [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — the random rotation,
  Lloyd-Max scalar quantization, and unbiased inner-product correction.
- **QJL** — A. Zandieh, M. Daliri, I. Han. *QJL: 1-Bit Quantized JL Transform
  for KV Cache Quantization with Zero Overhead*, 2024.
  [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) — the unbiased 1-bit
  transform behind the inner-product stage.
- **HNSW** — Yu. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate
  Nearest Neighbor Search using Hierarchical Navigable Small World Graphs*,
  2016. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) — the routing graph.
- **SimHash** — M. Charikar. *Similarity Estimation Techniques from Rounding
  Algorithms*, STOC 2002 — sign-random-projection codes; the multi-bit routing.
- **Johnson–Lindenstrauss lemma** — W. B. Johnson, J. Lindenstrauss, 1984 —
  random projection preserves angles, the basis for sub-`dim` routing codes.
- **Lloyd–Max quantization** — S. P. Lloyd, *Least Squares Quantization in PCM*
  (1982); J. Max, *Quantizing for Minimum Distortion* (1960) — the scalar codebook.

Datasets and baseline:

- **turbovec** — the comparison baseline.
  [github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)
- **ANN-Benchmarks** — M. Aumüller, E. Bernhardsson, A. Faithfull, 2020.
  [github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
  — the glove-100 protocol.
- **GloVe** — J. Pennington, R. Socher, C. D. Manning. *GloVe: Global Vectors
  for Word Representation*, EMNLP 2014.
- **DBpedia OpenAI embeddings** —
  [Qdrant/dbpedia-entities-openai3-...-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M)
  on Hugging Face.

## License

MIT. See [LICENSE](LICENSE).
