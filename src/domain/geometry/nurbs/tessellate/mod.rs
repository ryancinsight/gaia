//! Curvature-adaptive tessellation of NURBS curves and surfaces.
//!
//! # Surface tessellation algorithm
//!
//! 1. Start with a `min_segments` x `min_segments` coarse parameter grid.
//! 2. For each quad cell evaluate the surface normal at all 4 corners.
//! 3. If the maximum angle between any two corner normals exceeds
//!    `max_angle_deg`, bisect the longer parameter edge and recurse.
//! 4. Collect leaf quads; triangulate each into 2 triangles.
//! 5. Pass every vertex position through `IndexedMesh::add_vertex_pos`, which
//!    uses the built-in `VertexPool` spatial-hash to weld coincident vertices.
//!
//! # Curve tessellation algorithm
//!
//! 1. Start with `min_segments` uniform parameter samples.
//! 2. For each segment evaluate the tangent at both endpoints.
//! 3. If the angle between tangents exceeds `max_angle_deg`, insert the
//!    midpoint and recurse on both halves.
//! 4. Return the ordered list of 3-D sample positions.

mod curve;
mod options;
mod surface;

#[cfg(test)]
mod tests;

pub use curve::tessellate_curve;
pub use options::TessellationOptions;
pub use surface::tessellate_surface;

use crate::domain::core::scalar::Scalar;
use leto::geometry::UnitVector3;

/// Angle in degrees between two unit vectors (clamped to [0, 180]).
#[inline]
fn angle_deg<T: Scalar>(a: UnitVector3<T>, b: UnitVector3<T>) -> T {
    let cos_t = a
        .into_inner()
        .dot(b.into_inner())
        .clamp(<T as Scalar>::from_f64(-1.0), <T as Scalar>::from_f64(1.0));
    cos_t.acos().to_degrees()
}
