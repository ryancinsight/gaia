//! Compatibility shim for the former `n_way` arrangement entrypoint.
//!
//! The canonical implementation now lives in [`super::boolean_csg`]. This
//! module remains as a thin re-export so existing call sites continue to build
//! while all boolean paths share one implementation surface.

pub use super::boolean_csg::{csg_boolean, BooleanOp};

/// Former name of [`BooleanCandidatePair`](super::boolean_csg::BooleanCandidatePair).
pub type NWayCandidatePair = super::boolean_csg::BooleanCandidatePair;
/// Former name of [`BooleanFragmentRecord`](super::boolean_csg::BooleanFragmentRecord).
pub type NWayFragRecord = super::boolean_csg::BooleanFragmentRecord;
