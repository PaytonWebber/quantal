"""Compile the bundled quantal libraries into quantal/_libs/.

One shared library per dimension (quantal fixes dim at compile time). We
prebuild the dimensions real embedding models use so the wheel works with no
toolchain; uncommon dimensions fall back to build-on-demand at runtime.

    python build_libs.py                      # host target, default dims
    python build_libs.py --target x86_64-linux-gnu.2.28   # pin glibc (manylinux)
    python build_libs.py --dims 384,768,1536
    python build_libs.py --x86-variants       # also bundle .v3/.v4 SIMD builds

On x86-64, a single portable library must target the baseline ISA, which
leaves the SIMD kernels several times slower than a native build.
--x86-variants additionally compiles each dimension for x86-64-v3 (AVX2)
and x86-64-v4 (AVX-512); the loader picks the best variant the running
CPU supports.

Run from the python/ directory of a source checkout (zig must be on PATH).
Zig cross-compiles to a pinned glibc directly, so manylinux-compatible Linux
wheels need no Docker.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

# The embedding dimensions worth shipping prebuilt:
#   256  text-embedding-3-small (truncated), matryoshka
#   384  all-MiniLM-L6-v2, bge-small
#   512  CLIP, distiluse
#   768  bge-base, e5-base, mpnet, nomic-embed
#   1024 bge-large, e5-large, mxbai-embed-large
#   1536 OpenAI text-embedding-3-small / ada-002
#   3072 OpenAI text-embedding-3-large
DEFAULT_DIMS = [256, 384, 512, 768, 1024, 1536, 3072]

_EXT = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default=",".join(map(str, DEFAULT_DIMS)))
    ap.add_argument("--target", default=None, help="zig -Dtarget (e.g. x86_64-linux-gnu.2.28)")
    ap.add_argument("--ext", default=_EXT, help="output library extension")
    ap.add_argument("--x86-variants", action="store_true",
                    help="also build x86-64-v3/v4 SIMD variants (.v3/.v4 suffix)")
    args = ap.parse_args()

    if shutil.which("zig") is None:
        sys.exit("zig not found on PATH")
    root = pathlib.Path(__file__).resolve().parent.parent  # repo root (has build.zig)
    if not (root / "build.zig").is_file():
        sys.exit(f"build.zig not found at {root}; run from a source checkout")

    out_dir = pathlib.Path(__file__).resolve().parent / "quantal" / "_libs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (suffix, -Dcpu). The unsuffixed library stays the portable baseline so
    # older loaders and non-variant platforms keep working; pinning the
    # baseline explicitly also keeps native-target builds (e.g. Windows CI)
    # from inheriting whatever CPU the build machine has.
    variants = [("", "x86_64"), (".v3", "x86_64_v3"), (".v4", "x86_64_v4")] \
        if args.x86_variants else [("", None)]

    dims = [int(d) for d in args.dims.split(",") if d]
    for dim in dims:
        for suffix, cpu in variants:
            cmd = ["zig", "build", "-Doptimize=ReleaseFast", f"-Dc-dim={dim}"]
            if args.target:
                cmd.append(f"-Dtarget={args.target}")
            if cpu:
                cmd.append(f"-Dcpu={cpu}")
            print("building dim", dim, " ".join(cmd))
            subprocess.run(cmd, cwd=root, check=True)
            # Windows emits a prefixless DLL in bin/; Linux/macOS put
            # libquantal.<ext> in lib/. Bundled name is uniform (no lib prefix).
            if sys.platform == "win32":
                built = root / "zig-out" / "bin" / "quantal.dll"
            else:
                built = root / "zig-out" / "lib" / ("libquantal" + _EXT)
            dest = out_dir / f"quantal-dim{dim}{suffix}{args.ext}"
            shutil.copy(built, dest)
            print("  ->", dest, f"({dest.stat().st_size // 1024} KiB)")

    print(f"\nbundled {len(dims) * len(variants)} libraries into {out_dir}")


if __name__ == "__main__":
    main()
