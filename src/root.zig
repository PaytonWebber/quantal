//! quantajump: two-stage cascading vector index.
//! Stage 1 routes through an HNSW-style graph over 1-bit sign vectors;
//! Stage 2 reranks candidates with 3-bit TurboQuant payloads.

const std = @import("std");

pub const bitvec = @import("bitvec.zig");
pub const turboquant = @import("turboquant.zig");
pub const heap = @import("heap.zig");
pub const rotation = @import("rotation.zig");
pub const graph = @import("graph.zig");
pub const index = @import("index.zig");
pub const c_api = @import("c_api.zig");

pub const RandomRotation = rotation.RandomRotation;
pub const GraphNode = graph.GraphNode;
pub const RoutingGraph = graph.RoutingGraph;
pub const Index = index.Index;

// Force semantic analysis of the C exports so they land in the artifact.
comptime {
    _ = c_api;
}

pub const BitVector = bitvec.BitVector;
pub const hammingDistance = bitvec.hammingDistance;
pub const TQ3Chunk = turboquant.TQ3Chunk;
pub const TurboQuantPayload = turboquant.TurboQuantPayload;
pub const SearchResult = heap.SearchResult;

test {
    std.testing.refAllDecls(@This());
    _ = bitvec;
    _ = turboquant;
    _ = heap;
    _ = rotation;
    _ = graph;
    _ = index;
    _ = c_api;
    _ = @import("tests.zig");
}
