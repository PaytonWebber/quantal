"""Dump a controlled subset of an ann-benchmarks angular HDF5 to fvecs/ivecs
for the routing experiment: N train vectors, Q queries, and the subset-exact
top-10 (recomputed over the N-vector subset, normalized so IP == cosine).

    python benchmarks/dump_hdf5_subset.py data/glove-25-angular.hdf5 /tmp/g25 [N] [Q]
"""

import struct
import sys

import h5py
import numpy as np


def main():
    path = sys.argv[1]
    prefix = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    q = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000

    with h5py.File(path, "r") as f:
        train = np.array(f["train"][:n], dtype=np.float32)
        test = np.array(f["test"][:q], dtype=np.float32)
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-30)
    test /= np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-30)

    gt = np.empty((q, 10), dtype=np.int32)
    for i in range(0, q, 100):
        gt[i:i + 100] = np.argpartition(-(test[i:i + 100] @ train.T), 10, axis=1)[:, :10]

    def wvecs(p, a, fmt, cast):
        with open(p, "wb") as fo:
            for row in a:
                fo.write(struct.pack("<i", len(row)))
                fo.write(row.astype(cast).tobytes())

    wvecs(f"{prefix}_trn.fvecs", train, "f", "<f4")
    wvecs(f"{prefix}_tst.fvecs", test, "f", "<f4")
    wvecs(f"{prefix}_gt.ivecs", gt, "i", "<i4")
    print(f"{prefix}: train {train.shape}, queries {test.shape}, dim {train.shape[1]}")


if __name__ == "__main__":
    main()
