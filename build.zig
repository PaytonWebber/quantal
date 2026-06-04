const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const c_dim = b.option(usize, "c-dim", "Vector dimension the C API is compiled for") orelse 1536;
    const c_max_edges = b.option(usize, "c-max-edges", "Graph degree the C API is compiled for") orelse 32;
    const c_routing_bits = b.option(usize, "c-routing-bits", "SimHash routing-code length (0 = use c-dim; raise for low-dim datasets)") orelse 0;

    const options = b.addOptions();
    options.addOption(usize, "c_dim", c_dim);
    options.addOption(usize, "c_max_edges", c_max_edges);
    options.addOption(usize, "c_routing_bits", c_routing_bits);

    const mod = b.addModule("quantajump", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addOptions("build_options", options);

    const lib = b.addLibrary(.{
        .name = "quantajump",
        .linkage = .static,
        .root_module = mod,
    });
    b.installArtifact(lib);

    // Shared library for FFI consumers (e.g. the Python ctypes wrapper).
    // Links libc: std.Thread's libc-free spawn path requires Zig-controlled
    // process startup, which a dlopen'd library never gets.
    const shared = b.addLibrary(.{
        .name = "quantajump",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/shared_root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "quantajump", .module = mod }},
        }),
    });
    b.installArtifact(shared);

    const bench_exe = b.addExecutable(.{
        .name = "qj-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "quantajump", .module = mod }},
        }),
    });
    b.installArtifact(bench_exe);
    const run_bench = b.addRunArtifact(bench_exe);
    if (b.args) |args| run_bench.addArgs(args);
    const bench_step = b.step("bench", "Run the benchmark harness");
    bench_step.dependOn(&run_bench.step);

    const route_exe = b.addExecutable(.{
        .name = "qj-route",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/qj_route.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "quantajump", .module = mod }},
        }),
    });
    b.installArtifact(route_exe);

    const routing_exp = b.addExecutable(.{
        .name = "routing-exp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/routing_experiment.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "quantajump", .module = mod }},
        }),
    });
    const run_routing_exp = b.addRunArtifact(routing_exp);
    if (b.args) |args| run_routing_exp.addArgs(args);
    b.step("routing-exp", "Run the multi-bit routing experiment").dependOn(&run_routing_exp.step);

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
