//! Watertight mesh verification and repair.
//!
//! Critical for CFD: a watertight mesh has no boundary edges, consistent
//! outward orientation, and a finite positive enclosed volume. Self-
//! intersection detection is available as a separate BVH-accelerated CSG
//! operation; the constant-time edge report does not silently perform that
//! potentially expensive scan.

pub mod check;
pub mod repair;
pub mod seal;

pub use check::{assert_watertight, check_watertight, WatertightReport};
pub use repair::MeshRepair;
