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

## Multi-threaded (12 threads, same data, same machine)

| system | config | recall1@1 | QPS (MT) | µs/query |
|---|---|---|---|---|
| turbovec | 2-bit flat, batch | 0.884 | 1,764 | 567 |
| turbovec | 4-bit flat, batch | 0.964 | 956 | 1,046 |
| **quantajump** | m=128, sq8 | **0.964** | **17,156** | **58.3** |
| **quantajump** | m=512, sq8 | **0.990** | **6,609** | **151.3** |

At matched recall1@1 (0.964), quantajump answers **17.9× more queries/second**.
GloVe-100/100k MT: 85.6k QPS at m=128.

Build (100k DBpedia-1536): quantajump 33.2s serial → **5.7s** with the batched
parallel builder (12 threads; plan-parallel/commit-serial, growing batch) —
on par with turbovec's ~5s MT ingest. Batched-build recall is identical to the
serial build (GloVe 0.746/0.884 vs 0.739/0.883).

## Where turbovec wins (honest ledger)

| dimension | turbovec | quantajump |
|---|---|---|
| Resident memory | **75.5 MiB** (4-bit, no originals) | 276 MiB with sq8 (75.5 payloads + 54 graph + 147 sq8 store) |
| recall1@k tail | →1.0 by k=4 (exhaustive) | plateaus at routing recall (raise m to push it) |
| Recall1@k convergence guarantee | exhaustive scan, unconditional | requires the true NN to be routed into the beam |
| Maturity | PyPI/crates releases, framework integrations | Zig library + C ABI + ctypes Python wrapper; deletes, allowlist filtering, save/load — no packaged releases yet |

Notes:
- The latency gap grows with corpus size: turbovec scans 100% of vectors per query;
  quantajump evaluates ~1–3% and grows ~logarithmically.

## SQ8 rerank store (now the default)

int8 rerank records (1 byte/coord + one f32 scale per vector) instead of fp32:

| store | DBpedia-1536 R@1 (m=128 / 512) | rerank store size | total resident |
|---|---|---|---|
| fp32 | 0.970 / 0.995 | 586 MiB | 715 MiB |
| **sq8** | 0.966 / 0.991 | **147 MiB** | **276 MiB** |
| none | capped by 3-bit scoring | 0 | 129 MiB |

~0.4pp recall for a 4x smaller rerank store; quantajump's total memory premium over
turbovec 4-bit (75.5 MiB) drops to ~3.7x — the price of +3pp recall at 2-5x lower
latency.

## TQ+ per-coordinate calibration: measured, kept opt-in, no gain

Implemented per-coordinate shift/scale calibration (fit on the first 1000 adds,
frozen, decoded through the inverse affine inside the LUT kernel so scoring stays
in raw rotated space). Findings on GloVe-100/100k, the drift-prone case turbovec
cites (+1.4pp at 2-bit for their quantile variant):

- With stage-3 exact rerank: recall identical to baseline (stage 3 already erases
  stage-2 precision differences; stage 2 only selects the rerank pool).
- Without rerank (`--rerank none`): 1@1 0.658 -> 0.627 — slightly negative; our
  per-vector mean/std calibration already adapts to drift, leaving nothing for the
  per-coordinate fit to reclaim at 3-bit.
- Calibrating the 1-bit routing signs is decisively harmful (1@1 0.739 -> 0.544):
  centering strips the shared mean component that raw-cosine ranking weights, so
  routing bits stay raw by design.

`--tq-plus` remains available and serialized, default off.
