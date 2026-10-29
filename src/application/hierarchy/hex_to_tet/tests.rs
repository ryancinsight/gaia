//! Hex-to-tet conversion regression tests.
//!
//! ## Families
//! | Module | What it tests |
//! |--------|----------------|
//! | `conversion` | structured/branching conversion, boundary orientation |
//! | `properties` | key invariance, bounded membership, capacity bound |
//! Split from a single 743-line module as part of the GAIA-013 file-size burn-down.

mod conversion;
mod properties;
