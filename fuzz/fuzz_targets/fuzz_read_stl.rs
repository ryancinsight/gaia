// Fuzz target: STL parser (IndexedMesh path).
//
// Run with:
//   cargo +nightly fuzz run fuzz_read_stl -- -max_len=65536
//
// The parser must never panic on arbitrary input; it may return `Err`.
//
// The path is `gaia::infrastructure::io::…`, not `gaia::io::…`: `io` has been
// nested under `infrastructure` since the layer split, and this file named the
// old path for as long as it did because nothing on the ordinary toolchain
// compiles it — `cargo fuzz` needs nightly, and CI runs the pinned stable one.
// `tests/fuzz_entry_points.rs` now pins these paths on stable, so a stale path
// breaks that test instead of silently disabling the target.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // All errors are acceptable; panics are not.
    let _ = gaia::infrastructure::io::stl::fuzz_read_stl(data);
});
