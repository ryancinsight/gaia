//! Keeps the `fuzz/` targets honest on a stable toolchain.
//!
//! `cargo fuzz` needs nightly, and CI runs the pinned stable toolchain, so
//! nothing in this repository compiles `fuzz/fuzz_targets/*.rs` on an ordinary
//! push. Two of those targets came to name `gaia::io::stl::…`, a path that has
//! not existed since `io` moved under `infrastructure`, and a third named
//! `read_stl_he`, a function the crate no longer has. Neither was noticed,
//! because a fuzz target that does not compile looks exactly like a fuzz target
//! that does: nothing runs it.
//!
//! This test is the stable-toolchain stand-in. Binding each entry point to a
//! `fn` pointer fails to compile if the path stops resolving or the signature
//! changes, so a stale target now breaks the build here rather than quietly
//! claiming coverage it does not have.

use gaia::domain::core::error::MeshResult;
use gaia::infrastructure::io::{obj, ply, stl};
use gaia::IndexedMesh;

/// The exact path and signature each fuzz target calls.
#[test]
fn every_fuzz_target_entry_point_resolves() {
    let _: fn(&[u8]) -> MeshResult<IndexedMesh> = stl::fuzz_read_stl;
    let _: fn(&[u8]) -> MeshResult<IndexedMesh> = obj::fuzz_read_obj;
    let _: fn(&[u8]) -> MeshResult<IndexedMesh> = ply::fuzz_read_ply;
}

/// The contract those entry points advertise: arbitrary bytes, no panic.
///
/// This is a smoke sweep over the shapes that break parsers most often — empty
/// input, a truncated header, a non-finite field, an index that cannot be
/// represented — and not a substitute for `cargo fuzz`, which explores the
/// space this list can only sample.
#[test]
fn every_fuzz_entry_point_tolerates_adversarial_input() {
    let inputs: &[&[u8]] = &[
        b"",
        b"\n",
        b"\0",
        b"\xff\xfe\xfd",
        b"ply",
        b"solid",
        b"v",
        b"f",
        b"nan inf -inf\n",
        b"v 0 0 0\nf 1 2 3\n",
        b"ply\nformat ascii 1.0\nend_header\n",
        b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\nnan nan nan\n",
        b"18446744073709551615 18446744073709551615 18446744073709551615\n",
    ];

    for input in inputs {
        // Every outcome is acceptable except a panic.
        let _ = stl::fuzz_read_stl(input);
        let _ = obj::fuzz_read_obj(input);
        let _ = ply::fuzz_read_ply(input);
    }
}
