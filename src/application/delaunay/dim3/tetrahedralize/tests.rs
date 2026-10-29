//! 3D tetrahedralization regression tests.
//!
//! ## Families
//! | Module | What it tests |
//! |--------|----------------|
//! | `insertion` | point retention through insert/finalize |
//! | `circumsphere` | strictly-external rejection |
//! | `precision` | f32 strict-Delaunay oracle, f32/f64 agreement |
//! Split from a single 718-line module as part of the GAIA-013 file-size burn-down.

mod circumsphere;
mod insertion;
mod precision;
