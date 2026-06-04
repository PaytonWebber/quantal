# 1M-vector benchmark runbook (DBpedia OpenAI3, d=1536)

Goal: settle the scaling claim — quantal's ~O(log n) routing vs turbovec's
O(n) scan — with measured data at 999k base vectors + 1k held-out queries.

## Resources (verified on this machine)

- Disk: ~9.1 GB parquet shards (deleted after conversion) + 6.2 GB fvecs
- RAM: 30 GiB total. Each run peaks ~9-10 GiB (6.2 GiB dataset + index).
  **Run the two benchmarks sequentially, never concurrently.**
- Time estimates: conversion ~10 min; quantal build ~60-120 s (12T) +
  ground truth ~15-20 min (computed once per quantal-bench invocation) + searches;
  turbovec build ~50-100 s + 1k-query flat scans (~30 s/bit-width MT,
  ~6x that ST).

## Prep (done)

1. Shards downloaded to `data/train-000NN-of-00026.parquet` (26 files).
2. Convert (also done — fvecs present, parquet removed):
   ```
   python3 benchmarks/convert_dbpedia.py
   ```
   Produces `data/dbpedia1536_1m_base.fvecs` (999,000 vectors) and
   `data/dbpedia1536_1m_query.fvecs` (1,000 vectors).
3. Build the bench binary:
   ```
   zig build -Doptimize=ReleaseFast -Dc-dim=128
   ```

## Run 1: quantal

```bash
# single-threaded recall curve (ground truth dominates wall time)
./zig-out/bin/quantal-bench fvecs data/dbpedia1536_1m_base.fvecs \
    --query-file data/dbpedia1536_1m_query.fvecs --queries 1000 --max-base 999000 \
    --recall-curve --m 128,256,512,1024 --threads 12

# note: --threads 12 parallelizes BUILD and SEARCH; for the ST search
# numbers run again with --threads 1 (build will also be serial and slow,
# so capture both from the threaded run first, then a --threads 1 pass
# with a reduced --m list if ST latency is wanted).
./zig-out/bin/quantal-bench fvecs data/dbpedia1536_1m_base.fvecs \
    --query-file data/dbpedia1536_1m_query.fvecs --queries 1000 --max-base 999000 \
    --recall-curve --m 128,512
```

## Run 2: turbovec (after run 1 finishes)

```bash
# multi-threaded (default rayon, 12 threads)
QJ_BASE=data/dbpedia1536_1m_base.fvecs \
QJ_QUERY=data/dbpedia1536_1m_query.fvecs \
QJ_N=999000 python3 benchmarks/turbovec_compare.py

# single-threaded
RAYON_NUM_THREADS=1 QJ_BASE=data/dbpedia1536_1m_base.fvecs \
QJ_QUERY=data/dbpedia1536_1m_query.fvecs \
QJ_N=999000 python3 benchmarks/turbovec_compare.py
```

## What to record in RESULTS.md

- recall1@{1,2,4,...,64} per m / per bit-width
- ms/query ST and MT at matched recall
- build time, resident memory (quantal-bench prints its breakdown)
- The headline ratio: ms/query at recall1@1 ~= 0.96-0.97, 1M scale,
  vs the 5.7x measured at 100k.

## Expectations (to be confirmed or falsified)

- turbovec per-query cost grows ~10x from the 100k run (linear scan):
  ~27 ms ST / ~10 ms MT at 4-bit.
- quantal per-query cost grows ~1.5-2.5x (deeper graph, larger beam to
  hold recall): ~1-2 ms ST at recall ~0.96+.
- Routing recall at 1M is the genuine unknown — if recall1@1 at m=512
  drops well below ~0.95, the recall-tail work (next-steps item 3) gets
  promoted.
