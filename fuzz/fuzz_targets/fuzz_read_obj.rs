// Fuzz target: OBJ parser.
//
// Run with:
//   cargo +nightly fuzz run fuzz_read_obj -- -max_len=65536
//
// The parser must never panic on arbitrary input; it may return `Err`. OBJ is
// the format whose fields are the least constrained — face records carry
// position, texcoord and normal indices, any of which may be absent — so the
// index-resolution path is the interesting one.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // All errors are acceptable; panics are not.
    let _ = gaia::infrastructure::io::obj::fuzz_read_obj(data);
});
