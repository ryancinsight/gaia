//! Shared plane-clip test fixtures.

use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

pub(super) fn csg_pool() -> VertexPool {
    VertexPool::for_csg()
}

pub(super) fn p(x: f64, y: f64, z: f64) -> Point3r {
    Point3r::new(x, y, z)
}

pub(super) fn n_up() -> Vector3r {
    Vector3r::new(0.0, 0.0, 1.0)
}

/// Helper: build a face from 3 points, returning (face, pool).
pub(super) fn face_from_pts(a: Point3r, b: Point3r, c: Point3r) -> (FaceData, VertexPool) {
    let mut pool = csg_pool();
    let n = n_up();
    let va = pool.insert_or_weld(a, n);
    let vb = pool.insert_or_weld(b, n);
    let vc = pool.insert_or_weld(c, n);
    let face = FaceData::new(va, vb, vc, RegionId::INVALID);
    (face, pool)
}
