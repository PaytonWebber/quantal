#!/usr/bin/env python3
"""Recall/QPS frontier: quantal vs hnswlib and FAISS (HNSW, IVFPQ, flat), plus turbovec.

Places quantal on the standard ann-benchmarks recall/QPS frontier against the
recognized baselines, same machine, same protocol, so "competitive with the
field" is a measured claim rather than an assertion.

Protocol (ann-benchmarks conventions):
  - Standard HDF5 dataset (train / test / neighbors / distance attr).
  - Cosine, computed as inner product over L2-normalized vectors for EVERY
    index, so they all compute the same thing.
  - recall@k against the dataset's precomputed exact neighbors.
  - QPS measured single-thread, one query at a time (the leaderboard axis).
    All libraries are pinned to one thread so the comparison is apples-to-apples.
  - Index memory reported as serialized index size (the only measure consistent
    across libraries). Build time reported separately, never folded into QPS.

Each index sweeps its own query-time quality knob to trace its Pareto curve:
quantal `m`, hnswlib/FAISS-HNSW `efSearch`, FAISS-IVFPQ `nprobe`, turbovec
`bit_width`.

    python benchmarks/ann_frontier.py data/glove-100-angular.hdf5 \
        --indexes quantal,hnswlib,faiss-hnsw,faiss-ivfpq,faiss-flat,turbovec \
        --nq 1000 --out benchmarks/frontier_glove100.json

Environment: faiss-cpu and hnswlib have no Python 3.14 wheels; use a 3.11/3.12
venv: `pip install numpy h5py faiss-cpu hnswlib`. quantal is imported from
../python (its library for the dataset's dim must be buildable or bundled).
Any index whose package is missing, or that fails to build, is skipped with a
note rather than aborting the run.
"""

# Pin BLAS/OpenMP to a single thread BEFORE importing numpy/faiss so the
# single-thread QPS measurement is honest.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import tempfile
import time

import numpy as np

try:
    import h5py
except ImportError:
    raise SystemExit("ann_frontier needs h5py: pip install h5py")

DEFAULT_INDEXES = "quantal,hnswlib,faiss-hnsw,faiss-ivfpq,faiss-flat,turbovec"


# ---- shared helpers -------------------------------------------------------

def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def load_fvecs(path, max_n=0):
    """Read an .fvecs file ([int32 dim][dim float32] per row)."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise SystemExit(f"{path} is empty")
    dim = int(raw[0])
    row = dim + 1
    n = raw.size // row
    if max_n and max_n < n:
        n = max_n
    arr = raw[: n * row].reshape(n, row)
    return np.ascontiguousarray(arr[:, 1:].view(np.float32), dtype=np.float32)


def compute_truth(base, query, k):
    """Exact top-k (inner product) ground truth via a FAISS flat index, using
    all cores (this is not part of the timed measurement)."""
    import faiss
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    _, ids = index.search(query, k)
    faiss.omp_set_num_threads(1)
    return ids.astype(np.int64)


def recall_at_k(returned, truth, k):
    hits = 0
    for r, t in zip(returned, truth):
        hits += len(set(int(x) for x in r[:k]) & set(int(x) for x in t[:k]))
    return hits / (len(returned) * k)


def file_size_of(write_fn):
    """Serialize an index via `write_fn(path)` and return its byte size."""
    with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
        path = f.name
    try:
        write_fn(path)
        return os.path.getsize(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def time_queries(query_fn, test, k):
    """Run `query_fn(vec) -> list_of_ids` per query, single-thread, return
    (returned_ids, qps)."""
    t0 = time.perf_counter()
    returned = [query_fn(q) for q in test]
    elapsed = time.perf_counter() - t0
    qps = len(test) / elapsed if elapsed > 0 else 0.0
    return returned, qps


def time_batch(batch_fn, test):
    """Run `batch_fn(test)` once (the whole query set, all threads) and return
    throughput in queries/sec. Used for the multi-threaded number; the result
    set is identical to the single-thread pass, so recall is reused."""
    t0 = time.perf_counter()
    batch_fn(test)
    elapsed = time.perf_counter() - t0
    return len(test) / elapsed if elapsed > 0 else 0.0


def result(knob, recall, qps, qps_mt=None):
    r = {"knob": knob, "recall": round(recall, 4), "qps": round(qps, 1)}
    if qps_mt is not None:
        r["qps_mt"] = round(qps_mt, 1)
    return r


# ---- index runners --------------------------------------------------------
# Each returns {"build_s", "bytes", "points": [result(...), ...]} or None to skip.

def run_quantal(train, test, truth, k, args):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    try:
        from quantal import Index
    except Exception as e:  # noqa: BLE001
        print(f"quantal: skipped ({e})")
        return None
    dim = train.shape[1]
    try:
        idx = Index(dim=dim)
    except Exception as e:  # noqa: BLE001
        print(f"quantal: skipped (no library for dim {dim}: {e})")
        return None

    t0 = time.perf_counter()
    idx.add(train)
    build_s = time.perf_counter() - t0
    size = file_size_of(idx.save)

    points = []
    for m in args.m_grid:
        def query(q, m=m):
            _, ids, counts = idx.search_batch(q.reshape(1, -1), k=k, m=m, threads=1)
            return ids[0, : counts[0]]
        returned, qps = time_queries(query, test, k)
        qps_mt = time_batch(lambda t, m=m: idx.search_batch(t, k=k, m=m, threads=args.threads), test)
        points.append(result(f"m={m}", recall_at_k(returned, truth, k), qps, qps_mt))
    return {"build_s": round(build_s, 2), "bytes": size, "points": points}


def run_hnswlib(train, test, truth, k, args):
    try:
        import hnswlib
    except ImportError:
        print("hnswlib: not installed, skipping")
        return None
    dim = train.shape[1]
    n = train.shape[0]
    p = hnswlib.Index(space="ip", dim=dim)  # inner product on normalized = cosine
    t0 = time.perf_counter()
    p.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.hnsw_m)
    p.add_items(train, np.arange(n), num_threads=os.cpu_count())
    build_s = time.perf_counter() - t0
    size = file_size_of(p.save_index)

    points = []
    for ef in args.ef_grid:
        p.set_ef(ef)
        def query(q, p=p):
            labels, _ = p.knn_query(q.reshape(1, -1), k=k, num_threads=1)
            return labels[0]
        returned, qps = time_queries(query, test, k)
        qps_mt = time_batch(lambda t, p=p: p.knn_query(t, k=k, num_threads=args.threads), test)
        points.append(result(f"ef={ef}", recall_at_k(returned, truth, k), qps, qps_mt))
    return {"build_s": round(build_s, 2), "bytes": size, "points": points}


def _faiss():
    import faiss
    faiss.omp_set_num_threads(1)
    return faiss


def run_faiss_flat(train, test, truth, k, args):
    try:
        faiss = _faiss()
    except ImportError:
        print("faiss: not installed, skipping flat")
        return None
    dim = train.shape[1]
    index = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter()
    index.add(train)
    build_s = time.perf_counter() - t0
    size = file_size_of(lambda path: faiss.write_index(index, path))

    def query(q):
        _, ids = index.search(q.reshape(1, -1), k)
        return ids[0]
    returned, qps = time_queries(query, test, k)
    return {"build_s": round(build_s, 2), "bytes": size,
            "points": [result("exact", recall_at_k(returned, truth, k), qps)]}


def run_faiss_hnsw(train, test, truth, k, args):
    try:
        faiss = _faiss()
    except ImportError:
        print("faiss: not installed, skipping hnsw")
        return None
    dim = train.shape[1]
    index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction
    t0 = time.perf_counter()
    index.add(train)
    build_s = time.perf_counter() - t0
    size = file_size_of(lambda path: faiss.write_index(index, path))

    points = []
    for ef in args.ef_grid:
        index.hnsw.efSearch = ef
        def query(q, index=index):
            _, ids = index.search(q.reshape(1, -1), k)
            return ids[0]
        faiss.omp_set_num_threads(1)
        returned, qps = time_queries(query, test, k)
        faiss.omp_set_num_threads(args.threads)
        qps_mt = time_batch(lambda t, index=index: index.search(t, k), test)
        faiss.omp_set_num_threads(1)
        points.append(result(f"ef={ef}", recall_at_k(returned, truth, k), qps, qps_mt))
    return {"build_s": round(build_s, 2), "bytes": size, "points": points}


def run_faiss_ivfpq(train, test, truth, k, args):
    try:
        faiss = _faiss()
    except ImportError:
        print("faiss: not installed, skipping ivfpq")
        return None
    dim = train.shape[1]
    # m_pq must divide dim; pick the largest sensible divisor <= 64.
    m_pq = next((d for d in (64, 48, 32, 24, 16, 12, 10, 8, 4, 2, 1)
                 if d <= dim and dim % d == 0), 1)
    nlist = args.ivf_nlist
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    index.train(train)
    index.add(train)
    build_s = time.perf_counter() - t0
    size = file_size_of(lambda path: faiss.write_index(index, path))

    points = []
    for nprobe in args.nprobe_grid:
        index.nprobe = nprobe
        def query(q, index=index):
            _, ids = index.search(q.reshape(1, -1), k)
            return ids[0]
        faiss.omp_set_num_threads(1)
        returned, qps = time_queries(query, test, k)
        faiss.omp_set_num_threads(args.threads)
        qps_mt = time_batch(lambda t, index=index: index.search(t, k), test)
        faiss.omp_set_num_threads(1)
        points.append(result(f"nprobe={nprobe}", recall_at_k(returned, truth, k), qps, qps_mt))
    return {"build_s": round(build_s, 2), "bytes": size, "points": points,
            "config": f"nlist={nlist},m_pq={m_pq},nbits=8"}


def run_turbovec(train, test, truth, k, args):
    try:
        from turbovec import TurboQuantIndex
    except ImportError:
        print("turbovec: not installed, skipping")
        return None
    dim = train.shape[1]
    if dim % 8 != 0:
        pad = 8 - dim % 8
        train = np.pad(train, ((0, 0), (0, pad)))
        test = np.pad(test, ((0, 0), (0, pad)))
        dim += pad
    points = []
    build_s = None
    size = None
    for bits in args.bit_grid:
        idx = TurboQuantIndex(dim=dim, bit_width=bits)
        t0 = time.perf_counter()
        for i in range(0, len(train), 100_000):
            idx.add(train[i:i + 100_000])
        build_s = round(time.perf_counter() - t0, 2)
        size = file_size_of(idx.write) if hasattr(idx, "write") else None

        def query(q, idx=idx):
            _, ids = idx.search(q.reshape(1, -1), k)
            return ids[0]
        returned, qps = time_queries(query, test, k)
        points.append(result(f"bits={bits}", recall_at_k(returned, truth, k), qps))
    return {"build_s": build_s, "bytes": size, "points": points}


RUNNERS = {
    "quantal": run_quantal,
    "hnswlib": run_hnswlib,
    "faiss-hnsw": run_faiss_hnsw,
    "faiss-ivfpq": run_faiss_ivfpq,
    "faiss-flat": run_faiss_flat,
    "turbovec": run_turbovec,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", help="HDF5 with train/test/neighbors (ann-benchmarks format)")
    ap.add_argument("--base", help="fvecs base/corpus file (alternative to an HDF5 dataset)")
    ap.add_argument("--query", help="fvecs query file (used with --base)")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "angular"],
                    help="for fvecs mode; both normalize and use inner product")
    ap.add_argument("--indexes", default=DEFAULT_INDEXES)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=0, help="cap test queries (0 = all)")
    ap.add_argument("--max-train", type=int, default=0, help="subsample train (0 = all)")
    ap.add_argument("--out", default=None)
    # quality-knob grids
    ap.add_argument("--m-grid", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--ef-grid", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--nprobe-grid", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--bit-grid", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ivf-nlist", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                    help="threads for the multi-threaded (batched) QPS number")
    args = ap.parse_args()

    if args.base:
        # fvecs mode: load base/query, normalize, compute exact truth ourselves.
        name = os.path.basename(args.base)
        train = load_fvecs(args.base, args.max_train)
        test = load_fvecs(args.query, args.nq)
        distance = args.metric
        train, test = normalize(train), normalize(test)
        truth = compute_truth(train, test, args.k)
    else:
        if not args.dataset:
            raise SystemExit("provide an HDF5 dataset, or --base/--query fvecs files")
        name = os.path.basename(args.dataset)
        with h5py.File(args.dataset, "r") as f:
            train = np.array(f["train"], dtype=np.float32)
            test = np.array(f["test"], dtype=np.float32)
            truth = np.array(f["neighbors"], dtype=np.int64)
            distance = f.attrs.get("distance", "angular")
        if args.max_train and args.max_train < train.shape[0]:
            train = train[: args.max_train]
            print("WARNING: --max-train subsamples the corpus; recall is not "
                  "comparable to the dataset's precomputed neighbors.")
        if args.nq and args.nq < test.shape[0]:
            test = test[: args.nq]
            truth = truth[: args.nq]
        if distance == "angular":
            train, test = normalize(train), normalize(test)
        truth = truth[:, : args.k]
    print(f"dataset {name}: train {train.shape}, test {test.shape}, "
          f"metric {distance}, k={args.k}\n")

    out = {"dataset": name, "k": args.k,
           "n_train": int(train.shape[0]), "n_test": int(test.shape[0]),
           "metric": distance, "indexes": {}}

    for name in args.indexes.split(","):
        name = name.strip()
        runner = RUNNERS.get(name)
        if runner is None:
            print(f"{name}: unknown index, skipping")
            continue
        print(f"=== {name} ===")
        try:
            res = runner(train, test, truth, args.k, args)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: failed ({type(e).__name__}: {e})")
            continue
        if res is None:
            continue
        out["indexes"][name] = res
        mb = f"{res['bytes'] / 1e6:.1f}MB" if res.get("bytes") else "n/a"
        print(f"  build {res.get('build_s')}s, index {mb}")
        print(f"  {'knob':>14} {'recall@'+str(args.k):>10} {'QPS(1T)':>10} {'QPS('+str(args.threads)+'T)':>11}")
        for p in res["points"]:
            mt = f"{p['qps_mt']:>11.1f}" if p.get("qps_mt") is not None else f"{'-':>11}"
            print(f"  {p['knob']:>14} {p['recall']:>10.4f} {p['qps']:>10.1f} {mt}")
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
