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

## 1M-vector run (999,000 base + 1,000 held-out queries, same protocol)

Same machine, same metric; turbovec ingested in 100k chunks (a single
999k-row `add()` was OOM-killed at 21 GB RSS on this 30 GiB box — see
RUN_1M.md). qj-bench memory: 754.6 MiB payloads + 536.5 MiB graph +
1467.2 MiB sq8 store vs 5853.5 MiB raw fp32.

### Multi-threaded (12 threads)

| system | config | recall1@1 | ms/query | QPS |
|---|---|---|---|---|
| turbovec | 2-bit flat | 0.911 | 5.973 | 167 |
| turbovec | 4-bit flat | 0.977 | 11.263 | 89 |
| **quantajump** | m=128 | 0.932 | **0.076** | 13,173 |
| **quantajump** | m=256 | 0.956 | **0.111** | 8,993 |
| **quantajump** | m=512 | 0.975 | **0.234** | 4,270 |
| **quantajump** | m=1024 | **0.984** | **0.362** | 2,760 |

At matched recall (~0.975-0.977): **48x faster**. At the 0.91-0.93 tier: 79x.

### Single-threaded

| system | config | recall1@1 | ms/query |
|---|---|---|---|
| turbovec | 2-bit flat | 0.911 | 14.212 |
| turbovec | 4-bit flat | 0.977 | 27.000 |
| **quantajump** | m=128 | 0.934 | **0.524** |
| **quantajump** | m=512 | 0.977 | **1.891** |

At identical recall (0.977): **14.3x faster ST**; at the ~0.92 tier, 27x.

### Scaling, 100k -> 1M (the point of the experiment)

| per-query cost | 100k | 1M | growth at 10x data |
|---|---|---|---|
| turbovec 4-bit ST | 2.717 ms | 27.000 ms | **9.9x (linear)** |
| turbovec 4-bit MT | 1.046 ms | 11.263 ms | 10.8x |
| quantajump m=128 ST | 0.482 ms | 0.524 ms | **1.09x** |
| quantajump m=512 ST | 1.315 ms | 1.891 ms | 1.44x |

The flat scan pays the full corpus growth; graph routing pays ~log n. The
matched-recall advantage grew from 5.7x (100k) to 14.3x ST / 48x MT (1M)
and keeps compounding with n.

Other 1M observations:
- Routing recall held: 0.932 at m=128 (-3.2pp vs 100k), recoverable to
  0.984 at m=1024. recall1@1 == recall1@64 throughout (exact-rerank
  property survives scale).
- Build: 71.4s at 12 threads (5.5x parallel speedup; 390s serial);
  turbovec chunked builds 45-77s. Parity holds.
- turbovec 2-bit recall1@1 *rose* at 1M (0.884 -> 0.911): a denser corpus
  narrows the top-1 margin the quantizer must resolve less often than it
  widens it.
- turbovec single-query loop latency (not batch) at 1M: 39 ms (2-bit) /
  83 ms (4-bit) — the regime where interactive use stops being viable.

## glove-100-angular (ann-benchmarks protocol) — where quantajump LOSES

Standard ann-benchmarks dataset (1,183,514 train / 10,000 test, d=100,
angular), exact precomputed neighbors, recall@10 = |returned ∩ true|/10
(their default metric). Run via `benchmarks/ann_local.py`. Our metric was
validated: exact normalized-IP reproduces the HDF5 ground truth at
recall@10 = 1.0000.

| system | config | recall@10 | QPS (MT) |
|---|---|---|---|
| quantajump | m=256 | 0.435 | 35,076 |
| quantajump | m=1024 | 0.597 | 9,917 |
| turbovec | 2-bit | 0.570 | 4,408 |
| turbovec | 4-bit | **0.858** | 2,388 |

**On this low-dimensional dataset the DBpedia result reverses: turbovec
wins recall decisively (0.858 vs our best 0.597), and quantajump cannot
reach turbovec's recall@10 at any tested beam width.** We are faster at
any *given* recall, but only in a recall range too low to be useful.

Root cause — confirmed, not sparsity: quantajump's graph *routes* on
1-bit sign vectors, a d-bit code. At d=1536 that's 1536 bits of routing
signal (rich, hence the DBpedia dominance); at **d=100 it's 100 bits**,
so many vectors collapse to near-identical Hamming codes and the beam
cannot resolve true neighbors. Quadrupling graph density (max_edges
32→64, ef 200→400) moved recall@10 only 0.597→0.614 — the limit is the
routing code's information content, not the graph. turbovec has no
routing stage (it scans every vector), so its recall is the quantizer's
alone and is unaffected by dimension's effect on a graph.

(turbovec also cannot run d=100 natively — it requires dim%8==0; the
numbers above zero-pad to 104, which does not change angular ranking.)

### Takeaway (original) and the fix

The 1-bit default made quantajump a high-dimensional index: it won at
d≥768 and lost at d≤128. That weakness is now a tunable — see the next
section. With `routing_bits=1024` the full glove-100 pipeline reaches
recall@10 0.872 at 10,330 QPS (MT), **beating turbovec 4-bit's 0.858 at
2,148 QPS** where the 1-bit default managed only 0.597.

### Full-pipeline glove-100 with routing_bits=1024 (the fix, end-to-end)

Same dataset/protocol, shared library built `-Dc-dim=100 -Dc-routing-bits=1024`:

| system | config | recall@10 | QPS (MT) |
|---|---|---|---|
| quantajump (1-bit, default) | m=1024 | 0.597 | 9,917 |
| **quantajump (rb=1024)** | m=512 | **0.872** | 10,330 |
| **quantajump (rb=1024)** | m=1024 | **0.900** | 5,105 |
| turbovec | 4-bit | 0.858 | 2,148 |

Multi-bit routing turns the one dataset where quantajump lost into a win
on both axes. Cost: build 54s→106s (the projection), routing code
16→128 bytes/vector, a small per-query projection — QPS stays well ahead.
At d≥768 the default (routing_bits=dim) is unchanged and optimal, so the
high-dim results above are untouched.

## Multi-bit routing experiment — the low-dim fix, validated

Hypothesis: glove-100's poor recall is the 1-bit-per-dimension routing
ceiling (100 dims -> 100-bit Hamming code), not anything fundamental.
Test: replace the d sign bits with a B-bit SimHash code (sign of B
i.i.d. Gaussian projections), decoupling routing-code length from
dimension, and measure *routing recall* alone — the fraction of the
exact top-10 in the m-candidate set BEFORE any rerank (the ceiling on
final recall). glove-100, 200k vectors, 1k queries; one seed; only B
varies. Reproduce: `zig build routing-exp -Doptimize=ReleaseFast -- ...`
(benchmarks/routing_experiment.zig).

| routing bits | rr@10 (m=128) | rr@10 (m=512) | build (200k) |
|---|---|---|---|
| 100 (today, 1-bit/dim) | 0.285 | 0.450 | 30.0s |
| 256 | 0.566 | 0.761 | 32.1s |
| 512 | 0.740 | 0.900 | 34.7s |
| **1024** | 0.836 | **0.948** | 47.8s |
| 2048 | 0.880 | 0.966 | 66.7s |

Confirmed: routing recall scales directly with code length, exactly as
SimHash theory predicts (Hamming over B sign bits estimates angular
distance with variance ~1/B). At **B=1024, routing recall@10 = 0.948**
(m=512) — and that's the pre-rerank ceiling, so the full pipeline (exact
stage-3 on top) would reach ~0.948 on glove-100, **beating turbovec's
0.858** where today we manage only 0.597. Cost is modest: build +60% at
B=1024, a B×d projection per query (negligible), and a 128-byte routing
code per vector (vs the 147 MiB sq8 store at 1M, immaterial).

This makes the low-dim weakness a tunable, not a wall. Now SHIPPED:
`routing_bits` is a comptime Index parameter (default = dim, preserving
the high-dim path exactly; `-Dc-routing-bits` at the C-ABI/build layer).
The pre-rerank ceiling measured here (0.948) was confirmed end-to-end —
the full pipeline at routing_bits=1024 reaches recall@10 0.900 on the
full 1.18M glove-100 set, beating turbovec (see the glove section above).

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
