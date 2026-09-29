//! Axis-aligned box (cuboid) primitive.

use super::{PrimitiveError, PrimitiveMesh};
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds an axis-aligned box (cuboid) with the given dimensions.
///
/// The box is positioned with one corner at `origin` and extends
/// `(width, height, depth)` along the `+X`, `+Y`, `+Z` axes respectively.
///
/// ## Output
///
/// - 8 unique vertices, 12 faces (6 quads × 2 triangles)
/// - `RegionId(1)` on all faces
/// - `signed_volume = width × height × depth > 0`
///
/// ## Example
///
/// ```rust
/// use gaia::{Cube, primitives::PrimitiveMesh};
/// let mesh = Cube::unit().build().unwrap();
/// assert_eq!(mesh.vertex_count(), 8);
/// assert_eq!(mesh.face_count(), 12);
/// ```
#[derive(Clone, Debug)]
pub struct Cube {
    /// Corner closest to (−∞, −∞, −∞).
    pub origin: Point3r,
    /// Extent along +X (mm).
    pub width: f64,
    /// Extent along +Y (mm).
    pub height: f64,
    /// Extent along +Z (mm).
    pub depth: f64,
}

impl Cube {
    /// Unit cube `[0, 1]³` at the origin.
    #[must_use]
    pub fn unit() -> Self {
        Self {
            origin: Point3r::origin(),
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        }
    }

    /// Cube of side `s` centred at the origin.
    #[must_use]
    pub fn centred(s: f64) -> Self {
        let h = s / 2.0;
        Self {
            origin: Point3r::new(-h, -h, -h),
            width: s,
            height: s,
            depth: s,
        }
    }
}

impl Default for Cube {
    fn default() -> Self {
        Self::unit()
    }
}

impl PrimitiveMesh for Cube {
    fn build(&self) -> Result<IndexedMesh, PrimitiveError> {
        build(self)
    }
}

fn build(cube: &Cube) -> Result<IndexedMesh, PrimitiveError> {
    let width = cube.width;
    let height = cube.height;
    let depth = cube.depth;
    if width <= 0.0 || height <= 0.0 || depth <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "all dimensions must be > 0, got ({width}, {height}, {depth})"
        )));
    }

    let region = RegionId::new(1);
    let mut mesh = IndexedMesh::new();

    let origin_x = cube.origin.x;
    let origin_y = cube.origin.y;
    let origin_z = cube.origin.z;

    // 8 corner positions
    let corners = [
        Point3r::new(origin_x, origin_y, origin_z), // 0 left-bottom-back
        Point3r::new(origin_x + width, origin_y, origin_z), // 1 right-bottom-back
        Point3r::new(origin_x + width, origin_y + height, origin_z), // 2 right-top-back
        Point3r::new(origin_x, origin_y + height, origin_z), // 3 left-top-back
        Point3r::new(origin_x, origin_y, origin_z + depth), // 4 left-bottom-front
        Point3r::new(origin_x + width, origin_y, origin_z + depth), // 5 right-bottom-front
        Point3r::new(origin_x + width, origin_y + height, origin_z + depth), // 6 right-top-front
        Point3r::new(origin_x, origin_y + height, origin_z + depth), // 7 left-top-front
    ];

    // 6 quads: (corner indices [CCW from outside], outward normal)
    // Winding is CCW when viewed from outside the face.
    let quads: &[([usize; 4], Vector3r)] = &[
        ([0, 3, 2, 1], -Vector3r::z()), // −Z back face
        ([4, 5, 6, 7], Vector3r::z()),  // +Z front face
        ([0, 1, 5, 4], -Vector3r::y()), // −Y bottom face
        ([3, 7, 6, 2], Vector3r::y()),  // +Y top face
        ([0, 4, 7, 3], -Vector3r::x()), // −X left face
        ([1, 2, 6, 5], Vector3r::x()),  // +X right face
    ];

    for &(idx, normal) in quads {
        let [i0, i1, i2, i3] = idx;
        let v0 = mesh.add_vertex(corners[i0], normal);
        let v1 = mesh.add_vertex(corners[i1], normal);
        let v2 = mesh.add_vertex(corners[i2], normal);
        let v3 = mesh.add_vertex(corners[i3], normal);
        // Split quad into 2 CCW triangles
        mesh.add_face_with_region(v0, v1, v2, region);
        mesh.add_face_with_region(v0, v2, v3, region);
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::watertight::check::check_watertight;
    use crate::infrastructure::storage::edge_store::EdgeStore;
    use crate::test_support::assert_rejects;
    use eunomia::assert_relative_eq;

    #[test]
    fn cube_is_watertight() {
        let mesh = Cube::unit().build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.is_watertight, "cube must be watertight");
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn cube_volume_correct() {
        let mesh = Cube {
            origin: Point3r::origin(),
            width: 2.0,
            height: 3.0,
            depth: 4.0,
        }
        .build()
        .unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert_relative_eq!(report.signed_volume, 24.0, epsilon = 1e-10);
    }

    #[test]
    fn cube_invalid_dimensions() {
        let result = Cube {
            origin: Point3r::origin(),
            width: -1.0,
            height: 1.0,
            depth: 1.0,
        }
        .build();
        assert_rejects(
            &result,
            "invalid parameter: all dimensions must be > 0, got (-1, 1, 1)",
        );
    }
}
