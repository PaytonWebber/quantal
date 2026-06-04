"""Local ann-benchmarks-protocol run: standard HDF5 dataset, recall@k vs QPS.

Uses the exact dataset, split, ground truth, and recall metric the
ann-benchmarks harness uses (github.com/erikbern/ann-benchmarks), so the
numbers are directly comparable to its published leaderboard — without the
Docker orchestration. Runs quantajump and, if installed, turbovec.

    python benchmarks/ann_local.py data/glove-100-angular.hdf5

The HDF5 carries 'train', 'test', 'neighbors' (precomputed exact top-100),
and a 'distance' attr ('angular' -> cosine). recall@k = mean over test
queries of |returned_k ∩ true_k| / k, the harness default with k=10.
"""

import os
import sys
import time

import h5py
import numpy as np

K = 10
M_GRID = [64, 128, 256, 512, 1024]


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def recall_at_k(returned, truth, k):
    hits = 0
    for r, t in zip(returned, truth):
        hits += len(set(r[:k]) & set(t[:k]))
    return hits / (len(returned) * k)


def run_quantajump(train, test, truth, angular, lib_dir):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    from quantajump import Index

    dim = train.shape[1]
    lib = os.path.join(lib_dir, f"libquantajump.so")
    if not os.path.exists(lib):
        print(f"  (build first: zig build -Doptimize=ReleaseFast -Dc-dim={dim})")
        return
    idx = Index(lib)
    if idx.dim != dim:
        print(f"  SKIP: library built for dim {idx.dim}, dataset is {dim}")
        return

    t0 = time.perf_counter()
    idx.add(np.arange(train.shape[0], dtype=np.uint64), train)
    build_s = time.perf_counter() - t0
    print(f"quantajump: built {len(idx)} vectors in {build_s:.1f}s")
    print(f"  {'m':>5} {'recall@1':>9} {'recall@'+str(K):>10} {'QPS(1T)':>10} {'QPS(allT)':>11}")

    nq = test.shape[0]
    for m in M_GRID:
        # Single-thread QPS (the leaderboard's standard axis).
        t0 = time.perf_counter()
        _, ids_st, counts_st = idx.search(test, k=K, m=m, threads=1)
        qps_st = nq / (time.perf_counter() - t0)
        # All-thread QPS.
        t0 = time.perf_counter()
        idx.search(test, k=K, m=m, threads=os.cpu_count())
        qps_mt = nq / (time.perf_counter() - t0)

        returned = [ids_st[i, : counts_st[i]] for i in range(nq)]
        rec10 = recall_at_k(returned, truth, K)
        rec1 = recall_at_k(returned, truth, 1)
        print(f"  {m:>5} {rec1:>9.4f} {rec10:>10.4f} {qps_st:>10.0f} {qps_mt:>11.0f}")


def run_turbovec(train, test, truth, angular):
    try:
        from turbovec import TurboQuantIndex
    except ImportError:
        print("turbovec: not installed, skipping")
        return
    dim = train.shape[1]
    nq = test.shape[0]
    # turbovec requires dim % 8 == 0; zero-pad if needed (an extra constant
    # column does not change angular/IP ranking on normalized vectors).
    if dim % 8 != 0:
        pad = (8 - dim % 8)
        print(f"  note: turbovec requires dim%8==0; zero-padding {dim}->{dim + pad}")
        train = np.pad(train, ((0, 0), (0, pad)))
        test = np.pad(test, ((0, 0), (0, pad)))
        dim += pad
    print(f"  {'bits':>5} {'recall@'+str(K):>10} {'QPS':>10}")
    for bits in (2, 4):
        idx = TurboQuantIndex(dim=dim, bit_width=bits)
        for i in range(0, len(train), 100_000):
            idx.add(train[i:i + 100_000])
        t0 = time.perf_counter()
        _, ids = idx.search(test, K)  # honours RAYON_NUM_THREADS
        qps = nq / (time.perf_counter() - t0)
        rec = recall_at_k(ids, truth, K)
        print(f"  {bits:>5} {rec:>10.4f} {qps:>10.0f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/glove-100-angular.hdf5"
    lib_dir = sys.argv[2] if len(sys.argv) > 2 else "zig-out/lib"
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        truth = np.array(f["neighbors"], dtype=np.int64)
        distance = f.attrs.get("distance", "angular")
    angular = distance == "angular"
    print(f"dataset: {path}")
    print(f"  train {train.shape}, test {test.shape}, metric {distance}, k={K}\n")

    if angular:
        train, test = normalize(train), normalize(test)

    run_quantajump(train, test, truth, angular, lib_dir)
    print()
    run_turbovec(train, test, truth, angular)


if __name__ == "__main__":
    main()
