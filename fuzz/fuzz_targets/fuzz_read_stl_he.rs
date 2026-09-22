// REMOVED — do not re-register this target.
//
// This file used to be `[[bin]] fuzz_read_stl_he`, a target for `read_stl_he`.
// That function does not exist: the crate went watertight-first indexed and the
// half-edge STL reader went with it. The target therefore never compiled, and
// nothing noticed because `cargo fuzz` needs nightly while CI runs the pinned
// stable toolchain — so no build in this repository ever read this file.
//
// There is no replacement target to write here. Nothing converts an
// `IndexedMesh` into a `HalfEdgeMesh`, and the only `_he` function left in the
// crate is `write_vtk_he`, a writer — fuzzing a writer is a different contract
// from "parsing arbitrary bytes must not panic", which is what the other three
// targets assert.
//
// It is no longer listed in `fuzz/Cargo.toml`, so it is inert. Delete it with
// `git rm fuzz/fuzz_targets/fuzz_read_stl_he.rs`.
