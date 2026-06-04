"""Compile the bundled quantal libraries into quantal/_libs/.

One shared library per dimension (quantal fixes dim at compile time). We
prebuild the dimensions real embedding models use so the wheel works with no
toolchain; uncommon dimensions fall back to build-on-demand at runtime.

    python build_libs.py                      # host target, default dims
    python build_libs.py --target x86_64-linux-gnu.2.28   # pin glibc (manylinux)
    python build_libs.py --dims 384,768,1536

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
    args = ap.parse_args()

    if shutil.which("zig") is None:
        sys.exit("zig not found on PATH")
    root = pathlib.Path(__file__).resolve().parent.parent  # repo root (has build.zig)
    if not (root / "build.zig").is_file():
        sys.exit(f"build.zig not found at {root}; run from a source checkout")

    out_dir = pathlib.Path(__file__).resolve().parent / "quantal" / "_libs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = [int(d) for d in args.dims.split(",") if d]
    for dim in dims:
        cmd = ["zig", "build", "-Doptimize=ReleaseFast", f"-Dc-dim={dim}"]
        if args.target:
            cmd.append(f"-Dtarget={args.target}")
        print("building dim", dim, " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)
        built = root / "zig-out" / "lib" / ("libquantal" + _EXT)
        dest = out_dir / f"libquantal-dim{dim}{args.ext}"
        shutil.copy(built, dest)
        print("  ->", dest, f"({dest.stat().st_size // 1024} KiB)")

    print(f"\nbundled {len(dims)} libraries into {out_dir}")


if __name__ == "__main__":
    main()
