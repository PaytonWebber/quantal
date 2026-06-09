# quantal

[![PyPI](https://img.shields.io/pypi/v/quantaldb.svg)](https://pypi.org/project/quantaldb/)
[![CI](https://github.com/PaytonWebber/quantal/actions/workflows/ci.yml/badge.svg)](https://github.com/PaytonWebber/quantal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A full-precision graph needs 6.3 GB to search 1M OpenAI embeddings. quantal
does it in 2.6 GB at 0.99 recall@10, faster.

quantal is a vector index you link into an application, not a server you
run: a Zig core behind a C ABI, Python bindings, and drop-in stores for
LangChain, LlamaIndex, and LangGraph. The routing graph is
[HNSW](https://arxiv.org/abs/1603.09320) over 1-bit SimHash codes; the
scoring is Google Research's
[TurboQuant](https://arxiv.org/abs/2504.19874) at 3 bits, settled by an
exact rerank pass over a compact stored representation.

It began as a question: how close can a quantized index get to a
full-precision graph on the recall/QPS frontier while spending less
memory? The [Results](#results) section reports the measurements on
standard datasets against same-machine baselines, including where it
does not win.

## Quick start

```bash
pip install quantaldb
```

The package is `quantaldb`; the import is `quantal`:

```python
import numpy as np
from quantal import Index

index = Index(dim=1536)
ids = index.add(vectors)                 # (n, 1536) float32, auto-assigned ids
hits = index.search(query, k=10)         # -> [(id, score), ...], cosine
index.save("docs.tq")
index.memory_bytes                       # exact heap bytes, not RSS
```

Already on a framework? The stores are drop-in:

```python
from quantal.langchain import QuantalVectorStore        # swap for FAISS/InMemory
from quantal.llama_index import QuantalVectorStore
from quantal.langgraph_store import QuantalStore        # agent memory
```

Wheels bundle prebuilt libraries for the common embedding dimensions
(256/384/512/768/1024/1536/3072) on Linux, macOS (Apple Silicon), and
Windows, with AVX-512/AVX2 variants picked at load time, so `pip install`
runs within a few percent of a native build. For other dimensions,
install from a source checkout with [Zig](https://ziglang.org) 0.16 on
your PATH (`pip install -e python`) and the right library is built on
first use.

Zig:

```zig
const quantal = @import("quantal");
var index = try quantal.Index(768, 16, 768).init(allocator, 200, 42);
```

## Where It Fits

quantal sits between a flat quantizer and a full-precision graph: sub-linear
search like the graph, compact quantized storage like the quantizer, and a
rerank stage that uses stored vectors instead of ending on compressed-code
scores.

|                 | quantal                        | flat quantizer (turbovec) | full-precision graph (hnswlib) |
|-----------------|--------------------------------|---------------------------|--------------------------------|
| Query path      | graph-routed                   | linear scan, O(n)         | graph-routed                   |
| Index memory    | codes + sq8/fp32 rerank store | smallest (codes only)     | largest (fp32 held in graph)   |
| Final scoring   | fp32 exact, or sq8 rerank      | quantized code scores     | fp32 distances                 |

Reach for **quantal** with high-dimensional embeddings (384-3072) when you want
search that stays fast as the corpus grows, with lower memory than a
full-precision graph. Reach for a **flat quantizer** when absolute minimum
memory is the priority and a per-query scan is acceptable. Reach for a
**full-precision graph** when you can hold fp32 vectors in RAM and want the
highest recall tail.

Do not use quantal for low-dimensional vectors when a full-precision graph fits
in memory. The GloVe-100 result below is the example.

## Results

quantal was measured against standard same-machine baselines on the
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) recall/QPS view:
hnswlib and FAISS-HNSW for full-precision graph search, turbovec for flat
TurboQuant scan, and FAISS-IVFPQ for compressed IVF-PQ. Same machine (a Ryzen
7640U laptop), cosine via L2-normalized inner product for every index,
recall@10 against exact top-10 ground truth, QPS measured single-thread one
query at a time. Harness: [`benchmarks/ann_frontier.py`](benchmarks/ann_frontier.py).
recall@10 is the average fraction of the true top-10 neighbors returned; matched
recall means the speed comparison is made at the same result quality.

### High-dimensional embeddings (DBpedia, text-embedding-3-large, d=1536)

At 1M vectors, single thread. The table samples three points from the plotted
frontier, not separate tuned runs. Cells show QPS at the measured recall for
the first operating point that meets each threshold:

![DBpedia 1M: recall vs QPS](docs/frontier_dbpedia1m.svg)

| minimum recall@10 | quantal      | hnswlib      | FAISS-HNSW   |
|-------------------|--------------|--------------|--------------|
| >=0.95            | 1770 @ 0.965 | 1606 @ 0.954 | 1502 @ 0.961 |
| >=0.975           | 1174 @ 0.981 | 931 @ 0.977  | 841 @ 0.978  |
| >=0.99            | 739 @ 0.991  | 372 @ 0.992  | 453 @ 0.991  |

Across these measured targets, quantal returns more queries per second than
either full-precision graph while using a smaller serialized index: ~2.6 GB
against ~6.3 GB. The tradeoff is the extreme recall tail. In this 1M run,
quantal tops out at 0.9937 recall@10; FAISS-HNSW reaches 0.9958 at lower QPS.

![DBpedia 1M: index memory](docs/memory_dbpedia1m.svg)

The 100k frontier has the same shape, and the full-precision graphs reach
farther into the extreme recall tail:

![DBpedia 100k: recall vs QPS](docs/frontier_dbpedia100k.svg)

The compressed baselines define the low-memory corner, not the high-recall
frontier. At 1M, turbovec's linear scan uses 771 MB and runs at ~13-24 QPS. On
the 100k slice, FAISS-IVFPQ uses 15 MB, but recall@10 plateaus at 0.486 at this
dimensionality.

### Low-dimensional data (GloVe-100)

Below the 384-3072 range the result reverses. On GloVe-100 (d=100, 1.2M vectors)
the graph is faster across the whole frontier: at matched ~0.84 recall, hnswlib
returns about 4000 QPS to quantal's about 1700, and there is no memory advantage
(601 vs 649 MB), since fp32 vectors are small when the dimension is small. At low
dimension a full-precision graph is the better tool; the auto-widened routing
code narrows the gap but does not close it.

![GloVe-100: recall vs QPS](docs/frontier_glove100.svg)

Full methodology, raw logs, and the regeneration scripts
([`benchmarks/ann_frontier.py`](benchmarks/ann_frontier.py),
[`benchmarks/plot_frontier.py`](benchmarks/plot_frontier.py)) are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

Reproduce the 1M DBpedia run and redraw the README frontier plots:

```bash
python benchmarks/ann_frontier.py \
  --base data/dbpedia1536_1m_base.fvecs \
  --query data/dbpedia1536_1m_query.fvecs \
  --indexes quantal,hnswlib,faiss-hnsw,turbovec \
  --out benchmarks/frontier_dbpedia1m.json
python benchmarks/plot_frontier.py \
  benchmarks/frontier_dbpedia100k.json \
  benchmarks/frontier_dbpedia1m.json \
  benchmarks/frontier_glove100.json \
  --outdir docs
```

## How it works

1. **Route (1-bit).** Each vector is preconditioned with a random rotation and
   reduced to a sign-bit (SimHash) code. An HNSW-style graph navigates these by
   Hamming distance. This is cheap and enough to reach the right neighborhood.
2. **Score (3-bit).** Candidates are scored with 3-bit TurboQuant codes via a
   lookup-table kernel, with a per-vector scalar that keeps the inner-product
   estimate unbiased.
3. **Final rerank.** The top of that pool is rescored from the rerank store.
   The fp32 mode computes exact inner products inside the candidate pool. The
   default sq8 mode stores one byte per coordinate plus one scale per vector and
   was measured close to fp32 at much lower memory.

Queries are allocation-free: the search path takes a preallocated context and
never touches the allocator.

## Building (Zig)

```bash
zig build test                              # run the test suite
zig build bench -Doptimize=ReleaseFast -- synthetic --n 100000
zig build -Doptimize=ReleaseFast -Dc-dim=768   # build the C library + binaries
```

## Status

Benchmarked on a single laptop. The runs are single-machine and single-run, so
expect some variance. The frontier also reproduces under the independent
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) protocol; the
adapter lives in [benchmarks/ann-benchmarks/](benchmarks/ann-benchmarks/) and
is submitted upstream. The Zig core and C ABI are tested in CI on Linux,
macOS, and Windows; the Python package and framework wrappers are tested on
Linux and macOS, with wheel smoke tests on every published platform. Wheels
are published to PyPI as `quantaldb`; a Rust crate is not published yet.

## How this was built

The research direction, architecture, and benchmarking are human. The
implementation (the Zig core, the Python bindings, the framework wrappers) was
written with AI assistance. Every number in this README was measured, and the
generated code was reviewed and tested.

## References

The methods quantal builds on:

- **TurboQuant**: A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni. *TurboQuant:
  Online Vector Quantization with Near-optimal Distortion Rate*, 2025.
  [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), the random rotation,
  Lloyd-Max scalar quantization, and unbiased inner-product correction.
- **QJL**: A. Zandieh, M. Daliri, I. Han. *QJL: 1-Bit Quantized JL Transform
  for KV Cache Quantization with Zero Overhead*, 2024.
  [arXiv:2406.03482](https://arxiv.org/abs/2406.03482), the unbiased 1-bit
  transform behind the inner-product stage.
- **HNSW**: Yu. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate
  Nearest Neighbor Search using Hierarchical Navigable Small World Graphs*,
  2016. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320), the routing graph.
- **SimHash**: M. Charikar. *Similarity Estimation Techniques from Rounding
  Algorithms*, STOC 2002. Sign-random-projection codes and multi-bit routing.
- **Johnson-Lindenstrauss lemma**: W. B. Johnson, J. Lindenstrauss, 1984.
  random projection preserves angles, the basis for sub-`dim` routing codes.
- **Lloyd-Max quantization**: S. P. Lloyd, *Least Squares Quantization in PCM*
  (1982); J. Max, *Quantizing for Minimum Distortion* (1960). The scalar codebook.

Datasets and baseline:

- **turbovec**: the comparison baseline.
  [github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)
- **ANN-Benchmarks**: M. Aumüller, E. Bernhardsson, A. Faithfull, 2020.
  [github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
  for the glove-100 protocol.
- **GloVe**: J. Pennington, R. Socher, C. D. Manning. *GloVe: Global Vectors
  for Word Representation*, EMNLP 2014.
- **DBpedia OpenAI embeddings**:
  [Qdrant/dbpedia-entities-openai3-...-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M)
  on Hugging Face.

## License

MIT. See [LICENSE](LICENSE).
