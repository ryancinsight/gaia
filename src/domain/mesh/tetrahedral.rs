//! Validated tetrahedral volume-mesh construction.
//!
//! Surface [`super::MeshBuilder`] accepts triangle soup. This builder owns the
//! additional volume invariant: every inserted cell has four distinct,
//! finite vertices, positive native-scalar orientation, and one shared face
//! record for each undirected triangle. The resulting boundary faces retain
//! their outward orientation, so [`crate::domain::mesh::IndexedMesh::extract_boundary_mesh`]
//! can produce a watertight surface without reconstructing cell topology.

use leto::geometry::Point3;

use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::Cell;
use crate::infrastructure::storage::face_store::FaceStore;

/// Scalar-independent face identity and insertion state.
///
/// Keeping the registry separate from `IndexedMesh<T>` prevents the hash-table
/// and face-construction path from being copied into every scalar
/// monomorphization. The only generic work remains coordinate access and
/// native-precision orientation.
struct FaceRegistry {
    face_ids: hashbrown::HashMap<[VertexId; 3], FaceId>,
}

impl FaceRegistry {
    fn with_capacity(cell_capacity: usize) -> Self {
        Self {
            face_ids: hashbrown::HashMap::with_capacity(cell_capacity.saturating_mul(4)),
        }
    }

    fn insert_or_get(&mut self, faces: &mut FaceStore, vertices: [VertexId; 3]) -> FaceId {
        let key = canonical_face(vertices);
        if let Some(&face) = self.face_ids.get(&key) {
            return face;
        }

        let face = faces.add_triangle(vertices[0], vertices[1], vertices[2]);
        self.face_ids.insert(key, face);
        face
    }
}

#[inline]
fn canonical_face(mut vertices: [VertexId; 3]) -> [VertexId; 3] {
    if vertices[0] > vertices[1] {
        vertices.swap(0, 1);
    }
    if vertices[1] > vertices[2] {
        vertices.swap(1, 2);
    }
    if vertices[0] > vertices[1] {
        vertices.swap(0, 1);
    }
    vertices
}

fn validate_vertex_ids(
    cell_index: usize,
    vertices: [VertexId; 4],
    vertex_count: usize,
) -> MeshResult<()> {
    for &vertex in &vertices {
        if vertex.as_usize() >= vertex_count {
            return Err(invalid_cell(
                cell_index,
                format!("references vertex {vertex} outside the vertex store"),
            ));
        }
    }

    if vertices
        .iter()
        .enumerate()
        .any(|(index, vertex)| vertices[..index].contains(vertex))
    {
        return Err(invalid_cell(
            cell_index,
            "contains duplicate vertex identifiers",
        ));
    }

    Ok(())
}

fn tetrahedral_cell(faces: [FaceId; 4], vertices: [VertexId; 4]) -> Cell {
    let mut cell = Cell::tetrahedron(
        faces[0].as_usize(),
        faces[1].as_usize(),
        faces[2].as_usize(),
        faces[3].as_usize(),
    );
    cell.vertex_ids = vertices.map(VertexId::as_usize).to_vec();
    cell
}

/// Builder for tetrahedral volume meshes backed by [`IndexedMesh`].
///
/// The builder deduplicates shared triangular faces by their undirected vertex
/// set while preserving the first cell's outward winding. Cell vertex order is
/// normalized to positive native-scalar orientation before the four boundary
/// faces are inserted. Mixed cells are intentionally outside this contract:
/// Gaia's indexed face store is triangular, and current Atlas consumers use
/// tetrahedral volume cells plus triangular surface meshes.
pub struct TetrahedralMeshBuilder<T: Scalar = f64> {
    mesh: IndexedMesh<T>,
    face_registry: FaceRegistry,
}

impl<T: Scalar> TetrahedralMeshBuilder<T> {
    /// Start an empty tetrahedral builder.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    /// Start a builder with capacity for `vertex_capacity` vertices and
    /// `cell_capacity` tetrahedral cells.
    #[must_use]
    pub fn with_capacity(vertex_capacity: usize, cell_capacity: usize) -> Self {
        Self {
            mesh: IndexedMesh::with_capacity(
                vertex_capacity,
                cell_capacity.saturating_mul(4),
                cell_capacity,
            ),
            face_registry: FaceRegistry::with_capacity(cell_capacity),
        }
    }

    /// Add a vertex by position and return its welded identifier.
    pub fn vertex(&mut self, position: Point3<T>) -> VertexId {
        self.mesh.add_vertex_pos(position)
    }

    /// Add a vertex from three scalar coordinates.
    pub fn vertex_xyz(&mut self, x: T, y: T, z: T) -> VertexId {
        self.vertex(Point3::new(x, y, z))
    }

    /// Add a vertex from a coordinate array.
    pub fn vertex_array(&mut self, position: [T; 3]) -> VertexId {
        let [x, y, z] = position;
        self.vertex_xyz(x, y, z)
    }

    /// Add a tetrahedron from four existing vertex identifiers.
    ///
    /// The returned index is the cell index in the built [`IndexedMesh`].
    /// Input order may be either orientation; the builder reverses one pair
    /// when needed so every cell has positive native-scalar orientation.
    ///
    /// # Errors
    ///
    /// Returns [`MeshError::InvalidCell`] when a vertex identifier is out of
    /// range, a coordinate is non-finite, vertices are duplicated, or the
    /// tetrahedron has zero signed volume.
    pub fn tetrahedron(&mut self, vertices: [VertexId; 4]) -> MeshResult<usize> {
        let cell_index = self.mesh.cell_count();
        validate_vertex_ids(cell_index, vertices, self.mesh.vertex_count())?;

        let points = vertices.map(|vertex| *self.mesh.vertices.position(vertex));
        if points.iter().any(|point| {
            !<T as eunomia::NumericElement>::is_finite(point.x)
                || !<T as eunomia::NumericElement>::is_finite(point.y)
                || !<T as eunomia::NumericElement>::is_finite(point.z)
        }) {
            return Err(invalid_cell(
                cell_index,
                "contains a non-finite vertex coordinate",
            ));
        }

        let [a, b, c, d] = points;
        let signed_six_volume = (b - a).cross(c - a).dot(d - a);
        if signed_six_volume == <T as eunomia::NumericElement>::ZERO {
            return Err(invalid_cell(cell_index, "has zero signed volume"));
        }

        let oriented = if signed_six_volume > <T as eunomia::NumericElement>::ZERO {
            vertices
        } else {
            [vertices[0], vertices[2], vertices[1], vertices[3]]
        };

        let [a, b, c, d] = oriented;
        let faces = [
            self.face_registry
                .insert_or_get(&mut self.mesh.faces, [a, c, b]),
            self.face_registry
                .insert_or_get(&mut self.mesh.faces, [a, b, d]),
            self.face_registry
                .insert_or_get(&mut self.mesh.faces, [a, d, c]),
            self.face_registry
                .insert_or_get(&mut self.mesh.faces, [b, c, d]),
        ];
        self.mesh.add_cell(tetrahedral_cell(faces, oriented));
        Ok(cell_index)
    }

    /// Add a tetrahedron from four coordinate arrays.
    pub fn tetrahedron_array(&mut self, positions: [[T; 3]; 4]) -> MeshResult<usize> {
        let vertices = positions.map(|position| self.vertex_array(position));
        self.tetrahedron(vertices)
    }

    /// Finish construction and return the indexed volume mesh.
    #[must_use]
    pub fn build(mut self) -> IndexedMesh<T> {
        self.mesh.rebuild_edges();
        self.mesh
    }
}

impl<T: Scalar> Default for TetrahedralMeshBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn invalid_cell(cell: usize, reason: impl Into<String>) -> MeshError {
    MeshError::InvalidCell {
        cell,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::Scalar;

    fn unit_tetrahedron<T: Scalar>() {
        let mut builder = TetrahedralMeshBuilder::<T>::new();
        let scalar = |value| <T as Scalar>::from_f64(value);
        let vertices = [
            builder.vertex_array([scalar(0.0), scalar(0.0), scalar(0.0)]),
            builder.vertex_array([scalar(1.0), scalar(0.0), scalar(0.0)]),
            builder.vertex_array([scalar(0.0), scalar(1.0), scalar(0.0)]),
            builder.vertex_array([scalar(0.0), scalar(0.0), scalar(1.0)]),
        ];
        let cell = builder.tetrahedron([vertices[0], vertices[2], vertices[1], vertices[3]]);
        assert_eq!(cell.unwrap(), 0);

        let mut mesh = builder.build();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 4);
        assert_eq!(mesh.cell_count(), 1);
        assert_eq!(mesh.boundary_faces().len(), 4);
        assert!(mesh.is_watertight());
        let volume = <T as eunomia::NumericElement>::to_f64(mesh.signed_volume());
        assert!((volume - (1.0 / 6.0)).abs() < 1e-6);
        assert_eq!(mesh.cells()[0].vertex_ids.len(), 4);
    }

    #[test]
    fn supports_f32_and_f64_volume_contracts() {
        unit_tetrahedron::<f32>();
        unit_tetrahedron::<f64>();
    }

    #[test]
    fn deduplicates_shared_faces_and_keeps_boundary_watertight() {
        let mut builder = TetrahedralMeshBuilder::<f64>::new();
        let a = builder.vertex_array([0.0, 0.0, 0.0]);
        let b = builder.vertex_array([1.0, 0.0, 0.0]);
        let c = builder.vertex_array([0.0, 1.0, 0.0]);
        let d = builder.vertex_array([0.0, 0.0, 1.0]);
        let e = builder.vertex_array([0.0, 0.0, -1.0]);
        builder.tetrahedron([a, b, c, d]).unwrap();
        builder.tetrahedron([a, c, b, e]).unwrap();

        let mesh = builder.build();
        assert_eq!(mesh.cell_count(), 2);
        assert_eq!(mesh.face_count(), 7, "one shared face must be stored once");
        assert_eq!(mesh.boundary_faces().len(), 6);
    }

    #[test]
    fn rejects_duplicate_and_degenerate_cells_without_partial_faces() {
        let mut builder = TetrahedralMeshBuilder::<f64>::new();
        let a = builder.vertex_array([0.0, 0.0, 0.0]);
        let b = builder.vertex_array([1.0, 0.0, 0.0]);
        let c = builder.vertex_array([0.0, 1.0, 0.0]);
        let d = builder.vertex_array([0.25, 0.25, 0.0]);

        let duplicate = builder.tetrahedron([a, b, c, a]).unwrap_err();
        assert!(duplicate.to_string().contains("duplicate"));
        assert_eq!(builder.mesh.face_count(), 0);

        let degenerate = builder.tetrahedron([a, b, c, d]).unwrap_err();
        assert!(degenerate.to_string().contains("zero signed volume"));
        assert_eq!(builder.mesh.cell_count(), 0);
        assert_eq!(builder.mesh.face_count(), 0);
    }
}
