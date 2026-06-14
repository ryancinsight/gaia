//! Canonical CSG module surface (application-layer path).

pub mod arrangement;
pub mod boolean;
pub mod boolean_solid;
pub mod broad_phase;
pub mod clip;
pub mod coplanar;
pub mod corefine;
pub mod detect_self_intersect;
pub(crate) mod diagnostics;
pub mod intersect;
pub(crate) mod predicates3d;
pub mod reconstruct;

pub use boolean::{csg_boolean, csg_boolean_nary, BooleanOp, CsgNode};
pub use boolean_solid::BooleanSolid;
pub use detect_self_intersect::detect_self_intersections;
pub use intersect::IntersectionType;
