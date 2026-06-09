"""Loads the quantal shared library for a given dimension via ctypes.

quantal fixes the vector dimension at compile time, so there is one shared
library per dimension. This module hides that: given a dimension it returns a
ready ctypes handle. Resolution order:

  1. QUANTAL_LIB env var (an explicit library path), or an explicit arg.
  2. A binary bundled in the installed wheel (quantal/_libs/) — covers the
     common embedding dimensions, so `pip install` works with no toolchain.
     On x86-64 the wheel carries SIMD variants per dimension (.v4 for
     AVX-512, .v3 for AVX2, unsuffixed baseline) and the best one the
     running CPU supports is picked here.
  3. A cached build at ~/.cache/quantal/.
  4. Build it with `zig build -Dc-dim=N` if the source tree and `zig` are
     found (the long tail of uncommon dimensions), then cache it.

If none apply, a clear error explains how to produce the library.
"""

import ctypes
import os
import pathlib
import platform
import shutil
import subprocess
import sys

# Native shared-library extension for this platform.
_LIB_EXT = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")


def _lib_name(dim):
    # Uniform bundled/cache name across platforms (only the extension varies).
    return f"quantal-dim{dim}{_LIB_EXT}"


def _built_artifact(root):
    """Path of the dynamic library `zig build` just produced. Windows puts a
    prefixless DLL in bin/; Linux and macOS put libquantal.<ext> in lib/."""
    if sys.platform == "win32":
        return os.path.join(root, "zig-out", "bin", "quantal.dll")
    return os.path.join(root, "zig-out", "lib", "libquantal" + _LIB_EXT)

_u64p = ctypes.POINTER(ctypes.c_uint64)
_f32p = ctypes.POINTER(ctypes.c_float)
_usizep = ctypes.POINTER(ctypes.c_size_t)

_SIGS = {
    "quantal_dim": ([], ctypes.c_size_t),
    "quantal_routing_bits": ([], ctypes.c_size_t),
    "quantal_index_create": ([ctypes.c_size_t, ctypes.c_uint64], ctypes.c_void_p),
    "quantal_index_destroy": ([ctypes.c_void_p], None),
    "quantal_index_add_batch": ([ctypes.c_void_p, _u64p, _f32p, ctypes.c_size_t, ctypes.c_size_t], ctypes.c_int32),
    "quantal_index_remove": ([ctypes.c_void_p, ctypes.c_uint64], ctypes.c_int32),
    "quantal_index_len": ([ctypes.c_void_p], ctypes.c_size_t),
    "quantal_index_memory_bytes": ([ctypes.c_void_p], ctypes.c_size_t),
    "quantal_search": (
        [ctypes.c_void_p, ctypes.c_void_p, _f32p, ctypes.c_size_t, _u64p, _f32p],
        ctypes.c_size_t,
    ),
    "quantal_index_save": ([ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int32),
    "quantal_index_load": ([ctypes.c_char_p], ctypes.c_void_p),
    "quantal_context_create": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_void_p),
    "quantal_context_destroy": ([ctypes.c_void_p], None),
    "quantal_search_filtered": (
        [ctypes.c_void_p, ctypes.c_void_p, _f32p, _u64p, ctypes.c_size_t, ctypes.c_size_t, _u64p, _f32p],
        ctypes.c_size_t,
    ),
    "quantal_search_batch": (
        [ctypes.c_void_p, _f32p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, _u64p, _f32p, _usizep],
        ctypes.c_int32,
    ),
}

_cache = {}  # (resolved path) -> bound CDLL


def _bind(path):
    lib = ctypes.CDLL(path)
    for name, (argtypes, restype) in _SIGS.items():
        fn = getattr(lib, name, None)  # older libraries may lack newer entries
        if fn is not None:
            fn.argtypes, fn.restype = argtypes, restype
    return lib


def _project_root():
    # Source checkout: walk up from this file looking for build.zig.
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "build.zig").is_file():
            return parent
    env = os.environ.get("QUANTAL_SRC")
    return pathlib.Path(env) if env else None


def _x86_level():
    """The x86-64 microarchitecture level of this CPU (4, 3, or 0)."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return 0
    if sys.platform == "win32":
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
        # PF_AVX512F_INSTRUCTIONS_AVAILABLE / PF_AVX2_INSTRUCTIONS_AVAILABLE
        return 4 if present(41) else 3 if present(40) else 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
            else:
                return 0
    except OSError:
        return 0
    if {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"} <= flags:
        return 4
    # avx2+fma+bmi2 implies the rest of x86-64-v3 on every shipped CPU.
    if {"avx2", "fma", "bmi2"} <= flags:
        return 3
    return 0


_X86_LEVEL = _x86_level()


def _bundled(dim):
    """The best prebuilt wheel library this CPU supports, if present."""
    lib_dir = os.path.join(os.path.dirname(__file__), "_libs")
    name, ext = f"quantal-dim{dim}", _LIB_EXT
    suffixes = [".v4", ".v3", ""] if _X86_LEVEL >= 4 else [".v3", ""] if _X86_LEVEL == 3 else [""]
    for suffix in suffixes:
        p = os.path.join(lib_dir, name + suffix + ext)
        if os.path.exists(p):
            return p
    return None


def _cache_dir():
    base = os.environ.get("QUANTAL_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "quantal"
    )
    pathlib.Path(base).mkdir(parents=True, exist_ok=True)
    return base


def _build(dim):
    root = _project_root()
    if root is None or shutil.which("zig") is None:
        return None
    out = os.path.join(_cache_dir(), _lib_name(dim))
    subprocess.run(
        ["zig", "build", "-Doptimize=ReleaseFast", f"-Dc-dim={dim}"],
        cwd=root, check=True,
    )
    shutil.copy(_built_artifact(root), out)
    return out


def load(dim=None, lib_path=None):
    """Returns (lib, dim) for the requested dimension. `lib_path` (or
    QUANTAL_LIB) overrides resolution and may serve any dimension."""
    explicit = lib_path or os.environ.get("QUANTAL_LIB")
    if explicit:
        lib = _cache.get(explicit) or _bind(explicit)
        _cache[explicit] = lib
        got = int(lib.quantal_dim())
        if dim is not None and dim != got:
            raise ValueError(f"{explicit} is dim {got}, requested {dim}")
        return lib, got

    if dim is None:
        raise ValueError("provide dim=... (or set QUANTAL_LIB to a prebuilt library)")

    cached = os.path.join(_cache_dir(), _lib_name(dim))
    path = _bundled(dim) or (cached if os.path.exists(cached) else _build(dim))
    if path is None:
        raise RuntimeError(
            f"no quantal library for dim {dim}. This wheel bundles binaries "
            f"for the common embedding dimensions; for others, build one with\n"
            f"    zig build -Doptimize=ReleaseFast -Dc-dim={dim}\n"
            f"and set QUANTAL_LIB to zig-out/lib/libquantal{_LIB_EXT}, or "
            f"run from a source checkout with `zig` on PATH for automatic builds."
        )
    lib = _cache.get(path) or _bind(path)
    _cache[path] = lib
    return lib, int(lib.quantal_dim())
