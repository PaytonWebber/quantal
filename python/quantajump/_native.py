"""Loads the quantajump shared library for a given dimension via ctypes.

quantajump fixes the vector dimension at compile time, so there is one shared
library per dimension. This module hides that: given a dimension it returns a
ready ctypes handle, building and caching the library on first use when a Zig
toolchain and the project source are available. Resolution order:

  1. QUANTAJUMP_LIB env var (an explicit .so path), or an explicit path arg.
  2. A cached build at ~/.cache/quantajump/libquantajump-dim<N>.so.
  3. Build it with `zig build -Dc-dim=N` if the source tree and `zig` are
     found, then cache it.

If none apply, a clear error explains how to produce the library.
"""

import ctypes
import os
import pathlib
import shutil
import subprocess
import sys

_u64p = ctypes.POINTER(ctypes.c_uint64)
_f32p = ctypes.POINTER(ctypes.c_float)
_usizep = ctypes.POINTER(ctypes.c_size_t)

_SIGS = {
    "qj_dim": ([], ctypes.c_size_t),
    "qj_routing_bits": ([], ctypes.c_size_t),
    "qj_index_create": ([ctypes.c_size_t, ctypes.c_uint64], ctypes.c_void_p),
    "qj_index_destroy": ([ctypes.c_void_p], None),
    "qj_index_add_batch": ([ctypes.c_void_p, _u64p, _f32p, ctypes.c_size_t, ctypes.c_size_t], ctypes.c_int32),
    "qj_index_remove": ([ctypes.c_void_p, ctypes.c_uint64], ctypes.c_int32),
    "qj_index_len": ([ctypes.c_void_p], ctypes.c_size_t),
    "qj_index_save": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int32),
    "qj_index_load": ([ctypes.c_char_p], ctypes.c_void_p),
    "qj_context_create": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_void_p),
    "qj_context_destroy": ([ctypes.c_void_p], None),
    "qj_search_filtered": (
        [ctypes.c_void_p, ctypes.c_void_p, _f32p, _u64p, ctypes.c_size_t, ctypes.c_size_t, _u64p, _f32p],
        ctypes.c_size_t,
    ),
    "qj_search_batch": (
        [ctypes.c_void_p, _f32p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, _u64p, _f32p, _usizep],
        ctypes.c_int32,
    ),
}

_cache = {}  # (resolved path) -> bound CDLL


def _bind(path):
    lib = ctypes.CDLL(path)
    for name, (argtypes, restype) in _SIGS.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, restype
    return lib


def _project_root():
    # Source checkout: walk up from this file looking for build.zig.
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "build.zig").is_file():
            return parent
    env = os.environ.get("QUANTAJUMP_SRC")
    return pathlib.Path(env) if env else None


def _cache_dir():
    base = os.environ.get("QUANTAJUMP_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "quantajump"
    )
    pathlib.Path(base).mkdir(parents=True, exist_ok=True)
    return base


def _build(dim):
    root = _project_root()
    if root is None or shutil.which("zig") is None:
        return None
    out = os.path.join(_cache_dir(), f"libquantajump-dim{dim}.so")
    subprocess.run(
        ["zig", "build", "-Doptimize=ReleaseFast", f"-Dc-dim={dim}"],
        cwd=root, check=True,
    )
    shutil.copy(os.path.join(root, "zig-out", "lib", "libquantajump.so"), out)
    return out


def load(dim=None, lib_path=None):
    """Returns (lib, dim) for the requested dimension. `lib_path` (or
    QUANTAJUMP_LIB) overrides resolution and may serve any dimension."""
    explicit = lib_path or os.environ.get("QUANTAJUMP_LIB")
    if explicit:
        lib = _cache.get(explicit) or _bind(explicit)
        _cache[explicit] = lib
        got = int(lib.qj_dim())
        if dim is not None and dim != got:
            raise ValueError(f"{explicit} is dim {got}, requested {dim}")
        return lib, got

    if dim is None:
        raise ValueError("provide dim=... (or set QUANTAJUMP_LIB to a prebuilt .so)")

    cached = os.path.join(_cache_dir(), f"libquantajump-dim{dim}.so")
    path = cached if os.path.exists(cached) else _build(dim)
    if path is None:
        raise RuntimeError(
            f"no quantajump library for dim {dim}. Build one with\n"
            f"    zig build -Doptimize=ReleaseFast -Dc-dim={dim}\n"
            f"and point QUANTAJUMP_LIB at zig-out/lib/libquantajump.so, or run "
            f"from a source checkout with `zig` on PATH for automatic builds."
        )
    lib = _cache.get(path) or _bind(path)
    _cache[path] = lib
    return lib, int(lib.qj_dim())
