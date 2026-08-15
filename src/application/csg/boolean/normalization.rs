//! Operand-coordinate normalization for indexed Boolean operations.
//!
//! This leaf owns the coordinate-frame contract used by arrangement predicates.
//! Stable unit-scale operands remain borrowed; only out-of-band geometry is
//! materialized in a normalized frame and restored after reconstruction.

use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::aabb::Aabb;
use crate::domain::mesh::IndexedMesh;

/// Coordinate transform used when arrangement predicates need normalized coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CoordinateTransform {
    origin: Point3r,
    scale: f64,
}

impl CoordinateTransform {
    fn identity() -> Self {
        Self {
            origin: Point3r::origin(),
            scale: 1.0,
        }
    }

    fn is_identity(self) -> bool {
        self.origin == Point3r::origin() && self.scale == 1.0
    }

    #[inline]
    fn apply(self, position: Point3r) -> Point3r {
        Point3r::new(
            (position.x - self.origin.x) * self.scale,
            (position.y - self.origin.y) * self.scale,
            (position.z - self.origin.z) * self.scale,
        )
    }

    #[inline]
    fn inverse(self, position: Point3r) -> Point3r {
        let inverse_scale = 1.0 / self.scale;
        Point3r::new(
            self.origin.x + position.x * inverse_scale,
            self.origin.y + position.y * inverse_scale,
            self.origin.z + position.z * inverse_scale,
        )
    }
}

pub(super) enum NormalizedOperand<'a> {
    Borrowed(&'a IndexedMesh),
    Owned(Box<IndexedMesh>),
}

impl NormalizedOperand<'_> {
    pub(super) fn as_mesh(&self) -> &IndexedMesh {
        match self {
            Self::Borrowed(mesh) => mesh,
            Self::Owned(mesh) => mesh,
        }
    }
}

/// Normalize operand coordinates only outside the stable unit-scale band.
///
/// Arrangement predicates contain absolute guards calibrated for ordinary
/// unit-scale geometry. The `0.5..=10.0` band is the no-rescale contract for
/// common unit-scale inputs; scale-regression coverage exercises both sides of
/// these thresholds. Small and large operands are translated and scaled to a
/// unit diagonal so those guards remain meaningful. Common-scale inputs stay
/// in their original coordinates: this preserves exact decimal coplanarity in
/// axis-aligned solids instead of replacing it with a rounded irrational scale.
/// Stable operands are borrowed through [`NormalizedOperand`]; only the
/// out-of-band path materializes a transformed mesh.
pub(super) fn normalization_transform<'a, I>(meshes: I) -> CoordinateTransform
where
    I: IntoIterator<Item = &'a IndexedMesh>,
{
    let mut combined_bb = Aabb::empty();
    for mesh in meshes {
        combined_bb = combined_bb.union(&mesh.bounding_box());
    }

    let extent = combined_bb.max - combined_bb.min;
    let diagonal = extent.norm();
    if !diagonal.is_finite() || diagonal <= 0.0 {
        return CoordinateTransform::identity();
    }

    const STABLE_DIAGONAL_MIN: f64 = 0.5;
    const STABLE_DIAGONAL_MAX: f64 = 10.0;
    if (STABLE_DIAGONAL_MIN..=STABLE_DIAGONAL_MAX).contains(&diagonal) {
        return CoordinateTransform::identity();
    }

    CoordinateTransform {
        origin: combined_bb.min,
        scale: 1.0 / diagonal,
    }
}

pub(super) fn normalize_operand(
    mesh: &IndexedMesh,
    transform: CoordinateTransform,
) -> NormalizedOperand<'_> {
    if transform.is_identity() {
        return NormalizedOperand::Borrowed(mesh);
    }

    let mut normalized = mesh.clone();
    let vertex_count = normalized.vertices.len();
    for index in 0..vertex_count {
        let vertex_id = VertexId::new(index as u32);
        let position = *normalized.vertices.position(vertex_id);
        normalized
            .vertices
            .set_position(vertex_id, transform.apply(position));
    }
    normalized.vertices.rescale_spatial_hash(transform.scale);
    NormalizedOperand::Owned(Box::new(normalized))
}

pub(super) fn denormalize_result(
    mut mesh: IndexedMesh,
    transform: CoordinateTransform,
) -> IndexedMesh {
    if transform.is_identity() {
        return mesh;
    }

    let vertex_count = mesh.vertices.len();
    for index in 0..vertex_count {
        let vertex_id = VertexId::new(index as u32);
        let position = *mesh.vertices.position(vertex_id);
        mesh.vertices
            .set_position(vertex_id, transform.inverse(position));
    }
    mesh.vertices.rescale_spatial_hash(1.0 / transform.scale);
    mesh
}
