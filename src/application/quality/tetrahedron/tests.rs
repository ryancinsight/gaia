//! Tetrahedral quality regression tests.
//!
//! ## Families
//! | Module | What it tests |
//! |--------|----------------|
//! | `metrics` | native shape metrics: analytical values, slivers, invariance |
//! | `criteria` | classify/assess paths, constructor validation, typed units |
//! | `report` | invalid-cell counting without defaulted metrics |
//! Split from a single 745-line module as part of the GAIA-013 file-size burn-down.

mod criteria;
mod metrics;
mod report;
