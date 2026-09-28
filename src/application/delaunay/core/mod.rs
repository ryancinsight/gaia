//! Core shared Delaunay components.

mod shared;

pub(crate) use shared::{canonical_edge, segment_cross_point, segments_cross_proper};
