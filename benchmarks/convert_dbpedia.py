"""Converts the DBpedia-OpenAI3-1536 parquet shards into fvecs.

Streams all shards in data/, writing the first --base rows to the base file
and the next --queries rows to the query file. Usage:

    python3 benchmarks/convert_dbpedia.py \
        --shards "data/train-*.parquet" \
        --base-out data/dbpedia1536_1m_base.fvecs --base 999000 \
        --query-out data/dbpedia1536_1m_query.fvecs --queries 1000
"""

import argparse
import glob
import struct

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="data/train-*.parquet")
    parser.add_argument("--base-out", default="data/dbpedia1536_1m_base.fvecs")
    parser.add_argument("--base", type=int, default=999_000)
    parser.add_argument("--query-out", default="data/dbpedia1536_1m_query.fvecs")
    parser.add_argument("--queries", type=int, default=1_000)
    args = parser.parse_args()

    shards = sorted(glob.glob(args.shards))
    if not shards:
        raise SystemExit(f"no shards match {args.shards}")
    emb_col = next(c for c in pq.read_schema(shards[0]).names if "embedding" in c.lower())
    need = args.base + args.queries

    written = 0
    with open(args.base_out, "wb") as fb, open(args.query_out, "wb") as fq:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=4096, columns=[emb_col]):
                for emb in batch.column(0):
                    v = emb.as_py()
                    rec = struct.pack("<i", len(v)) + struct.pack(f"<{len(v)}f", *v)
                    (fb if written < args.base else fq).write(rec)
                    written += 1
                    if written == need:
                        print(f"wrote {args.base} base + {args.queries} query vectors, dim {len(v)}")
                        return
                if written % 102_400 == 0:
                    print(f"  {written}/{need}", flush=True)
        raise SystemExit(f"shards exhausted at {written}/{need} rows")


if __name__ == "__main__":
    main()
