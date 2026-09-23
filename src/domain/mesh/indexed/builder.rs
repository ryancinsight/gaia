//! `MeshBuilder`: the ergonomic front end for assembling an indexed mesh.

use super::IndexedMesh;
use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::Scalar;
use leto::geometry::Point3;

/// Ergonomic builder for constructing an [`IndexedMesh<T>`].
pub struct MeshBuilder<T: Scalar = f64> {
    mesh: IndexedMesh<T>,
}

impl<T: Scalar> MeshBuilder<T> {
    /// Start building with default millifluidic tolerances.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mesh: IndexedMesh::new(),
        }
    }

    /// Start building with a custom exact cell grid sizing.
    pub fn with_cell_size(cell_size: T) -> Self {
        Self {
            mesh: IndexedMesh::with_cell_size(cell_size),
        }
    }

    /// Add a vertex by position; returns its [`VertexId`].
    pub fn vertex(&mut self, pos: Point3<T>) -> VertexId {
        self.mesh.add_vertex_pos(pos)
    }

    /// Add a vertex from explicit coordinates; returns its [`VertexId`].
    pub fn vertex_xyz(&mut self, x: T, y: T, z: T) -> VertexId {
        self.vertex(Point3::new(x, y, z))
    }

    /// Add a vertex from a coordinate array; returns its [`VertexId`].
    pub fn vertex_array(&mut self, pos: [T; 3]) -> VertexId {
        let [x, y, z] = pos;
        self.vertex_xyz(x, y, z)
    }

    /// Add a triangle from three vertex IDs.
    pub fn triangle(&mut self, v0: VertexId, v1: VertexId, v2: VertexId) -> FaceId {
        self.mesh.add_face(v0, v1, v2)
    }

    /// Add raw triangle soup — each triple is `(p0, p1, p2)`.
    pub fn add_triangle_soup(&mut self, triangles: &[(Point3<T>, Point3<T>, Point3<T>)]) {
        for (a, b, c) in triangles {
            let va = self.mesh.add_vertex_pos(*a);
            let vb = self.mesh.add_vertex_pos(*b);
            let vc = self.mesh.add_vertex_pos(*c);
            self.mesh.add_face(va, vb, vc);
        }
    }

    /// Add raw triangle soup from coordinate arrays; each triple is `(p0, p1, p2)`.
    pub fn add_triangle_soup_arrays(&mut self, triangles: &[([T; 3], [T; 3], [T; 3])]) {
        for &(a, b, c) in triangles {
            let va = self.vertex_array(a);
            let vb = self.vertex_array(b);
            let vc = self.vertex_array(c);
            self.mesh.add_face(va, vb, vc);
        }
    }

    /// Finalise: build edges and return the mesh.
    pub fn build(mut self) -> IndexedMesh<T> {
        self.mesh.rebuild_edges();
        self.mesh
    }
}

impl<T: Scalar> Default for MeshBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}
