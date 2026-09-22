// Fuzz target: PLY parser.
//
// Run with:
//   cargo +nightly fuzz run fuzz_read_ply -- -max_len=65536
//
// The parser must never panic on arbitrary input; it may return `Err`. PLY is
// the format with the most file-controlled structure ahead of the data — the
// header declares element counts and property lists that the body is then read
// against — so the header path is as interesting as the body.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // All errors are acceptable; panics are not.
    let _ = gaia::infrastructure::io::ply::fuzz_read_ply(data);
});
