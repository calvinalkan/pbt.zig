//! `pbt` is a library of property-test utilities for Zig code. `Prng` draws
//! random inputs from a seed, and `Exhaustive` runs a test body once for every
//! sequence of choices the body can make.

pub const Prng = @import("prng.zig");
pub const Exhaustive = @import("exhaustive.zig");

test {
    _ = Prng;
    _ = Exhaustive;
    // `histogram` checks the shares of `Prng`'s draws in `Prng`'s own tests,
    // so it is tested here but not exported.
    _ = @import("histogram.zig");
}
