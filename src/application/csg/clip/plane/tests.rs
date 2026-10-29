//! Plane clip/refine regression tests.
//!
//! ## Families
//! | Module | What it tests |
//! |--------|----------------|
//! | `equation` | classification, construction, edge intersection |
//! | `classify` | above, below, straddling, coplanar |
//! | `clip` | keep, remove, split, coplanar, degenerate plane |
//! | `refine` | batch splitting, region ids, determinism |
//! | `fixtures` | shared pool/point/face fixtures |
//! Split from a single 703-line module as part of the GAIA-013 file-size burn-down.

mod classify;
mod clip;
mod equation;
mod fixtures;
mod refine;
