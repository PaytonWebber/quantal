# quantajump vs turbovec — same machine, same data, same metric

- **Hardware:** AMD Ryzen 5 7640U (Zen 4, AVX-512), 6C/12T, single-threaded unless noted
- **Data:** DBpedia-entities OpenAI3 text-embedding-3-large, d=1536, first 100,000
  vectors as base + next 1,000 as queries (HF: `Qdrant/dbpedia-entities-openai3-...-1536-1M`),
  L2-normalized; identical arrays fed to both systems
- **Metric:** recall1@k — exact FP32 top-1 found within approximate top-k (turbovec's
  published metric); ground truth via exact matmul
- **turbovec:** v0.7.0 from PyPI (`bit_width` 2/4); batch timing = 1000-query batch / 1000;
  ST = `RAYON_NUM_THREADS=1`, MT = default (12 threads)
- **quantajump:** 3-bit payloads + 1-bit routing graph + exact FP32 stage-3 rerank
  (`max_edges=16`, `ef_construction=200`, default symmetric stage-2 scoring),
  single-threaded
- Reproduce: `zig build bench -Doptimize=ReleaseFast -- fvecs data/dbpedia1536_base.fvecs
  --query-file data/dbpedia1536_query.fvecs --queries 1000 --recall-curve --m 64,128,256,512`
  and `python3 benchmarks/turbovec_compare.py`

## Single-threaded, DBpedia-1536, 100k vectors

| system | config | recall1@1 | recall1@4 | ms/query (ST) |
|---|---|---|---|---|
| turbovec | 2-bit flat | 0.884 | 0.996 | 1.443 |
| turbovec | 4-bit flat | 0.964 | 1.000 | 2.717 |
| **quantajump** | m=64 | 0.936 | 0.937 | **0.351** |
| **quantajump** | m=128 | **0.970** | 0.971 | **0.479** |
| **quantajump** | m=256 | **0.983** | 0.984 | **0.812** |
| **quantajump** | m=512 | **0.995** | 0.996 | **1.347** |

- At matched recall1@1 (~0.97): quantajump is **5.7× faster** than turbovec 4-bit ST
  (0.479 vs 2.717 ms/query).
- quantajump m=512 exceeds turbovec 4-bit recall (0.995 vs 0.964) at **2× lower latency**.
- turbovec with all 12 threads (4-bit, 1.046 ms/q) is still 2.2× slower than
  single-threaded quantajump at equal recall.

## Where turbovec wins (honest ledger)

| dimension | turbovec | quantajump |
|---|---|---|
| Resident memory | **75.5 MiB** (4-bit, no originals) | 715 MiB (75.5 payloads + 54 graph + 586 fp32 rerank store) |
| Build time (100k, ST) | **10.1 s** (flat, no graph) | 33.2 s (Gram-Schmidt init + serial HNSW) |
| recall1@k tail | →1.0 by k=4 (exhaustive) | plateaus at routing recall (raise m to push it) |
| Recall1@k convergence guarantee | exhaustive scan, unconditional | requires the true NN to be routed into the beam |

Notes:
- quantajump's memory is dominated by the fp32 stage-3 rerank store; an SQ8 store
  (int8) cuts it to ~222 MiB at negligible recall cost, and `store_originals=false`
  drops it to 129 MiB (recall then capped by 3-bit scoring, like turbovec is by 4-bit).
- The latency gap grows with corpus size: turbovec scans 100% of vectors per query;
  quantajump evaluates ~1–3% and grows ~logarithmically.
