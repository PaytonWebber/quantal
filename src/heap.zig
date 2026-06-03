//! Fixed-capacity binary heaps over caller-provided buffers. No allocation
//! happens here; capacity management is the caller's responsibility, which
//! keeps the query path allocation-free.

const std = @import("std");

pub fn BinaryHeap(comptime T: type, comptime before: fn (T, T) bool) type {
    return struct {
        const Self = @This();

        items: []T,
        len: usize = 0,

        pub fn fromBuffer(buffer: []T) Self {
            return .{ .items = buffer };
        }

        pub fn push(self: *Self, value: T) void {
            std.debug.assert(self.len < self.items.len);
            var i = self.len;
            self.items[i] = value;
            self.len += 1;
            while (i > 0) {
                const parent = (i - 1) / 2;
                if (!before(self.items[i], self.items[parent])) break;
                std.mem.swap(T, &self.items[i], &self.items[parent]);
                i = parent;
            }
        }

        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;
            const root = self.items[0];
            self.len -= 1;
            if (self.len > 0) {
                self.items[0] = self.items[self.len];
                self.siftDown(0);
            }
            return root;
        }

        pub fn peek(self: *const Self) ?T {
            return if (self.len == 0) null else self.items[0];
        }

        fn siftDown(self: *Self, start: usize) void {
            var i = start;
            while (true) {
                var best = i;
                const left = 2 * i + 1;
                const right = 2 * i + 2;
                if (left < self.len and before(self.items[left], self.items[best])) best = left;
                if (right < self.len and before(self.items[right], self.items[best])) best = right;
                if (best == i) return;
                std.mem.swap(T, &self.items[i], &self.items[best]);
                i = best;
            }
        }
    };
}

pub const SearchResult = struct {
    id: u64,
    score: f32,
};

fn lowerScore(a: SearchResult, b: SearchResult) bool {
    return a.score < b.score;
}

/// Tracks the K highest-scoring results in a caller-provided buffer using a
/// min-heap rooted at the current worst kept score.
pub const TopK = struct {
    heap: BinaryHeap(SearchResult, lowerScore),

    pub fn fromBuffer(buffer: []SearchResult) TopK {
        return .{ .heap = .fromBuffer(buffer) };
    }

    pub fn offer(self: *TopK, result: SearchResult) void {
        if (self.heap.len < self.heap.items.len) {
            self.heap.push(result);
        } else if (result.score > self.heap.peek().?.score) {
            _ = self.heap.pop();
            self.heap.push(result);
        }
    }

    /// In-place heapsort: drains the min-heap from the back, leaving the
    /// buffer sorted by descending score. Returns the result count.
    pub fn sortDescending(self: *TopK) usize {
        const count = self.heap.len;
        while (self.heap.pop()) |worst| {
            self.heap.items[self.heap.len] = worst;
        }
        return count;
    }
};

test "binary heap orders by comparator" {
    const lessThan = struct {
        fn f(a: u32, b: u32) bool {
            return a < b;
        }
    }.f;
    var buf: [8]u32 = undefined;
    var heap = BinaryHeap(u32, lessThan).fromBuffer(&buf);
    for ([_]u32{ 5, 1, 4, 2, 8, 3 }) |v| heap.push(v);

    var prev: u32 = 0;
    while (heap.pop()) |v| {
        try std.testing.expect(v >= prev);
        prev = v;
    }
    try std.testing.expectEqual(@as(?u32, null), heap.pop());
}

test "TopK keeps the k best, sorted descending" {
    var buf: [3]SearchResult = undefined;
    var topk = TopK.fromBuffer(&buf);
    for ([_]f32{ 0.1, 0.9, 0.4, 0.7, 0.2, 0.8 }, 0..) |s, i| {
        topk.offer(.{ .id = i, .score = s });
    }
    const count = topk.sortDescending();
    try std.testing.expectEqual(@as(usize, 3), count);
    try std.testing.expectEqual(@as(u64, 1), buf[0].id); // 0.9
    try std.testing.expectEqual(@as(u64, 5), buf[1].id); // 0.8
    try std.testing.expectEqual(@as(u64, 3), buf[2].id); // 0.7
}

test "TopK with fewer offers than capacity" {
    var buf: [10]SearchResult = undefined;
    var topk = TopK.fromBuffer(&buf);
    topk.offer(.{ .id = 1, .score = 0.5 });
    topk.offer(.{ .id = 2, .score = 0.6 });
    const count = topk.sortDescending();
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(u64, 2), buf[0].id);
}
