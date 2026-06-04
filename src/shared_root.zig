//! Root of the shared library build. Pulling in the main module makes the
//! C ABI exports (src/c_api.zig) part of the artifact; the module wrapper
//! exists so this artifact can link libc — std.Thread's libc-free spawn
//! path depends on Zig-controlled process startup (static TLS init), which
//! never runs when the library is dlopen'd into a foreign process such as
//! Python. With libc linked, spawn goes through pthread_create instead.

comptime {
    _ = @import("quantal");
}
