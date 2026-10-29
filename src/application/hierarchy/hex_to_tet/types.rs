//! Shared hex-decomposition vocabulary: face keys and capacity bounds.

use crate::domain::core::index::VertexId;

pub(super) type TriKey = [VertexId; 3];

pub(super) const HEX_VERTEX_COUNT: usize = 8;
pub(super) const HEX_MAX_NEIGHBORS: usize = HEX_VERTEX_COUNT - 1;
pub(super) const HEX_MAX_TETRAHEDRA: usize = 6;
pub(super) const TETRAHEDRON_FACE_COUNT: usize = 4;
