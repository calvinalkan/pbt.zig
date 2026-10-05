# pbt.zig

Property-based testing for Zig that draws the inputs where bugs hide most.

## Install

pbt.zig requires Zig 0.16.

```sh
zig fetch --save git+https://github.com/calvinalkan/pbt.zig
```

```zig
// build.zig
const pbt = b.dependency("pbt", .{});
test_module.addImport("pbt", pbt.module("pbt"));
```

## Usage

A property test draws many inputs and checks something that must hold for
every one of them:

```zig
test "sort orders any slice" {
    // `zig build test` picks a new seed for every run.
    var prng = pbt.Prng.init(std.testing.random_seed);
    var storage: [4096]u32 = undefined;

    // Each iteration draws one input and checks it.
    for (0..1024) |_| {
        // `intExponential` draws mostly short lengths and sometimes long ones.
        // With a mean of 256, about 63% of the lengths fall below 256, and 5%
        // reach 768 or more.
        const values = storage[0..@min(prng.intExponential(usize, 256), storage.len)];

        // A value is an edge such as 0, 255, or 4_294_967_295, a number people
        // type such as 16 or 1_000, or plain noise.
        const ValueKind = enum { edge, literal, any };

        // `enumWeights` draws a new weight for each kind in every case, such as
        // 900 for edges, 10 for literals, and 0 for noise. One case then holds
        // nothing but edges, and the next holds mostly noise with a rare
        // literal.
        const value_kind_weights = prng.enumWeights(ValueKind, .full);

        // `enumWeighted` picks each value's kind by the case's weights.
        for (values) |*value| {
            value.* = switch (prng.enumWeighted(ValueKind, value_kind_weights)) {
                .edge => prng.intEdge(u32),
                .literal => prng.intLiteral(u32),
                .any => prng.intAny(u32),
            };
        }

        // Sorting must leave any slice in order.
        std.mem.sort(u32, values, {}, std.sort.asc(u32));
        try std.testing.expect(std.sort.isSorted(u32, values, {}, std.sort.asc(u32)));
    }
}
```

Every `zig build test` uses a new seed, so repeated runs keep exploring new
inputs. A failing run prints its seed, and `zig build test --seed 0x… -- "sort"`
replays the failure exactly.

`src/prng.zig` and `src/exhaustive.zig` document every function with
examples.

## Inputs where bugs hide most

Uniform random values almost never hit the values where code breaks. A random
`u32` is practically never 0, 255, or 65_535, and a random slice practically
never has all its items equal. `Prng` draws such values on purpose:

| Function | Draws, for example |
| --- | --- |
| `intEdge(u16)` | 0, 1, 127, 255, 32_767, 65_535 |
| `intLiteral(u32)` | numbers people type, such as 1, 3, 16, 64, 255, 999, or 1_000 |
| `floatEdge(f32)` | 0, the smallest subnormal, 1, the largest f32, ∞, NaN, the largest f16, and 2^24, where `x + 1 == x` first holds |
| `floatLiteral(f32)` | 0.5, 2.5, 0.1, 1_000 |
| `indexEdge(items)` | the first, second, second-to-last, or last index |
| `intLogUniform(usize, 0, 4_096)` | sizes where 1 to 15 come up as often as 256 to 4_095 |
| `fillPattern(…)` | slices that are all equal, have one odd item, repeat, or come in runs |
| `nudge(u8, 255, 1)` | 254 or 255, to catch off-by-one errors at a boundary |
| `intAny`, `floatBetween`, … | plain uniform noise |

### Swarm testing

In the usage example, `enumWeights` draws new weights for every case. This is
called swarm testing. With equal weights, a slice of 256 values almost never
holds only edges. With new weights for every case, some cases hold only edges
and others hold mostly noise. A test that draws operations such as insert and
remove can weight them the same way, so one case never removes and another
removes almost all the time.

Will Wilson's [talk on swarm testing](https://www.youtube.com/watch?v=wzfC7Q-xNik)
explains why these ideas find bugs reliably.

## Exhaustively testing everything in a small space

`Exhaustive` runs a test body once for every sequence of choices the body can
make, so the test checks every input instead of a sample. It suits input
domains small enough to run in full, such as every slice of up to three small
values.
matklad's post
[Generate All the Things](https://matklad.github.io/2021/11/07/generate-all-the-things.html)
explains the technique. `Exhaustive` has the same choice methods as `Prng`, such
as `intBetween`, `index`, and `pick`.

```zig
test "sort orders every short slice" {
    var exhaustive: pbt.Exhaustive = .{};

    while (!exhaustive.done()) {
        var storage: [3]u8 = undefined;
        const values = storage[0..exhaustive.intBetween(usize, 0, 3)];
        for (values) |*value| value.* = exhaustive.intBetween(u8, 0, 2);

        std.mem.sort(u8, values, {}, std.sort.asc(u8));
        try std.testing.expect(std.sort.isSorted(u8, values, {}, std.sort.asc(u8)));
    }
}
```

The loop runs 40 cases, one for every slice of length 0 to 3 whose values are
0, 1, or 2.

## Compared with Zig's fuzzer

`zig build test --fuzz` runs the `std.testing.fuzz` tests. The fuzzer mutates
its input and keeps every mutation that reaches new code, so it climbs toward
code that only an exact input reaches, such as a branch on a magic number, a
keyword, or a checksum. Parsers, decoders, and file formats are full of such
code.

Coverage helps only when a failing input runs different code than a passing
one. Many bugs don't. A cache that evicts the wrong entry after it has been
full twice, or a sum that loses precision for certain values, runs the same
lines whether it fails or not, so the fuzzer gets no signal to follow. `Prng`
draws edges, typed literals, and lopsided mixes from the first case on, which
is what finds such bugs. pbt.zig tests also run in every `zig build test`,
take milliseconds, and replay a failure exactly from its seed.

### Combining `Prng` with Zig's fuzzer

Write the contract you want to test as a function that takes its randomness
source as a parameter. The same check then runs as a fast test with `Prng` in
every `zig build test`, and as a long, coverage-guided search under
`zig build test --fuzz`:

```zig
const Source = union(enum) {
    prng: *pbt.Prng,
    smith: *std.testing.Smith,

    // The method must be `inline`. The fuzzer tells decisions apart by the
    // code address that calls `Smith`, so each call must land in the property.
    inline fn intBetween(source: Source, comptime T: type, min: T, max: T) T {
        return switch (source) {
            .prng => |prng| prng.intBetween(T, min, max),
            .smith => |smith| smith.valueRangeAtMost(T, min, max),
        };
    }
};

fn checkSortOrders(source: Source) !void {
    var storage: [16]u32 = undefined;
    const values = storage[0..source.intBetween(u8, 0, storage.len)];
    for (values) |*value| value.* = source.intBetween(u32, 0, 1_000);

    std.mem.sort(u32, values, {}, std.sort.asc(u32));
    try std.testing.expect(std.sort.isSorted(u32, values, {}, std.sort.asc(u32)));
}

// Fast: 1024 cases in every `zig build test`.
test "sort orders any slice" {
    var prng = pbt.Prng.init(std.testing.random_seed);
    for (0..1024) |_| try checkSortOrders(.{ .prng = &prng });
}

// Long: `zig build test --fuzz` keeps searching for new coverage.
test "fuzz: sort orders any slice" {
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, smith: *std.testing.Smith) !void {
            try checkSortOrders(.{ .smith = smith });
        }
    }.testOne, .{});
}
```

A property can also take `source: anytype`. A `*pbt.Prng` and a
`*pbt.Exhaustive` then fit as they are, and the fuzzer needs a struct whose
methods wrap `Smith` the same way, `inline` included.

Note: in Zig 0.16.0, `--fuzz` builds only in a release mode, such as with
`zig build test --fuzz -Doptimize=ReleaseSafe`, because of Zig issues
[#36326](https://codeberg.org/ziglang/zig/issues/36326) and
[#30655](https://codeberg.org/ziglang/zig/issues/30655).

### Long runs and shrinking

A `Prng` never runs out of draws. One 64-bit seed yields as many values as a
run needs, and the seed replays a run of millions of draws as exactly as a
short test. `Prng` also works outside tests, so a simulator or a workload
generator can use it, while `std.testing.fuzz` runs only inside the test
runner.

pbt.zig does not shrink a failing input to a smaller one. Sizes are mostly
small and values mostly simple, so a failing input is often small already.

## Credits

- `Exhaustive` is adapted from TigerBeetle's
  [`exhaustigen.zig`](https://github.com/tigerbeetle/tigerbeetle/blob/47aeb2212a255273dda508288412e537d11e4b7c/src/testing/exhaustigen.zig),
  which implements matklad's
  [Generate All the Things](https://matklad.github.io/2021/11/07/generate-all-the-things.html).
- `Prng` is inspired by TigerBeetle's
  [`prng.zig` and `fuzz.zig`](https://github.com/tigerbeetle/tigerbeetle/tree/47aeb2212a255273dda508288412e537d11e4b7c/src).
- The values that `Prng` favors and its swarm weights follow Will Wilson's
  [talk on swarm testing](https://www.youtube.com/watch?v=wzfC7Q-xNik).

TigerBeetle is licensed under the
[Apache License 2.0](https://github.com/tigerbeetle/tigerbeetle/blob/47aeb2212a255273dda508288412e537d11e4b7c/LICENSE).
