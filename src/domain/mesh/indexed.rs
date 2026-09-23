//! The canonical watertight-first indexed surface mesh.
//!
//! [`IndexedMesh`] is the crate's single source of truth for surface and
//! volume topology, combining the deduplicating vertex pool, face and edge
//! stores, attribute store, cell list, and `GhostCell` half-edge topology.

use leto::geometry::{Point3, Vector3};
// =========================================================================
// IndexedMesh<T> — watertight-first surface mesh, generic over precision
// =========================================================================

use crate::domain::core::index::{FaceId, RegionId, VertexId};
use crate::domain::core::scalar::Scalar;
use crate::domain::topology::Cell;
use crate::infrastructure::storage::attribute::AttributeStore;
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::infrastructure::storage::face_store::FaceStore;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use crate::infrastructure::storage::vertex_pool::DEFAULT_MESH_CELL_SIZE;
use hashbrown::HashMap;

/// A deduplicated, indexed triangle surface mesh — generic over scalar `T`.
///
/// | Type parameter | Precision | Tolerance |
/// |----------------|-----------|-----------|
/// | `f64` (default) | 64-bit | 1 nm |
/// | `f32`          | 32-bit | 10 µm (GPU staging) |
///
/// The default `T = f64` means all existing `IndexedMesh::new()` call-sites
/// continue to compile without any annotation.  New code may write
/// `IndexedMesh::<f32>::new()` to get single-precision geometry at zero
/// additional runtime cost.
///
/// Combines:
/// - [`VertexPool<T>`] — spatial-hash-deduplicated vertex storage
/// - `FaceStore` — indexed triangles with region tags
/// - `EdgeStore` — persistent adjacency (rebuilt on demand)
/// - `AttributeStore` — named per-face scalar channels
#[derive(Clone)]
pub struct IndexedMesh<T: Scalar = f64> {
    /// Deduplicated vertex positions and normals.
    pub vertices: VertexPool<T>,
    /// Indexed triangular faces.
    pub faces: FaceStore,
    /// Edge adjacency (lazily built from faces).
    edges: Option<EdgeStore>,
    /// Per-face scalar attributes.
    pub attributes: AttributeStore<FaceId>,
    /// Volumetric cells (for CFD support).
    pub cells: Vec<Cell>,
    /// Boundary patch names tagged by `FaceId`.
    pub boundary_labels: HashMap<FaceId, std::borrow::Cow<'static, str>>,
}

impl<T: Scalar> IndexedMesh<T> {
    /// Create an empty mesh with default millifluidic tolerances.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: VertexPool::with_tolerance(
                <T as crate::domain::core::scalar::Scalar>::from_f64(DEFAULT_MESH_CELL_SIZE),
                <T as crate::domain::core::scalar::Scalar>::from_f64(DEFAULT_MESH_CELL_SIZE),
            ),
            faces: FaceStore::new(),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::new(),
            boundary_labels: HashMap::new(),
        }
    }

    /// Create with explicit exact grid cell size.
    pub fn with_cell_size(cell_size: T) -> Self {
        Self {
            vertices: VertexPool::new(cell_size),
            faces: FaceStore::new(),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::new(),
            boundary_labels: HashMap::new(),
        }
    }

    /// Create with explicit cell size and tolerance-based welding.
    ///
    /// Vertices within `tolerance` of an existing vertex are welded to it.
    /// Useful for CSG reconstruction where near-duplicate seam vertices
    /// need to be merged at a wider tolerance than the default 1e-4.
    pub fn with_tolerance(cell_size: T, tolerance: T) -> Self {
        Self {
            vertices: VertexPool::with_tolerance(cell_size, tolerance),
            faces: FaceStore::new(),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::new(),
            boundary_labels: HashMap::new(),
        }
    }

    /// Create an empty mesh with reserved capacity to prevent vector resizing.
    #[must_use]
    pub fn with_capacity(
        vertex_capacity: usize,
        face_capacity: usize,
        cell_capacity: usize,
    ) -> Self {
        Self {
            vertices: VertexPool::with_capacity(
                vertex_capacity,
                <T as crate::domain::core::scalar::Scalar>::from_f64(DEFAULT_MESH_CELL_SIZE),
            ),
            faces: FaceStore::with_capacity(face_capacity),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::with_capacity(cell_capacity),
            boundary_labels: HashMap::new(),
        }
    }

    /// Create an empty mesh with reserved capacity and an explicit spatial-hash cell size.
    #[must_use]
    pub fn with_capacity_and_cell_size(
        vertex_capacity: usize,
        face_capacity: usize,
        cell_capacity: usize,
        cell_size: T,
    ) -> Self {
        Self {
            vertices: VertexPool::with_capacity(vertex_capacity, cell_size),
            faces: FaceStore::with_capacity(face_capacity),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::with_capacity(cell_capacity),
            boundary_labels: HashMap::new(),
        }
    }

    /// Create an empty mesh with reserved capacity and explicit tolerance-based welding.
    #[must_use]
    pub fn with_capacity_and_tolerance(
        vertex_capacity: usize,
        face_capacity: usize,
        cell_capacity: usize,
        cell_size: T,
        tolerance: T,
    ) -> Self {
        Self {
            vertices: VertexPool::with_capacity_and_tolerance(
                vertex_capacity,
                cell_size,
                tolerance,
            ),
            faces: FaceStore::with_capacity(face_capacity),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::with_capacity(cell_capacity),
            boundary_labels: HashMap::new(),
        }
    }

    /// Create an empty clone of this mesh that preserves exactly the same
    /// `VertexPool` scalar tolerances (`inv_cell_size` and `tolerance_sq`),
    /// but drops all vertices, faces, and attributes.
    #[must_use]
    pub fn empty_clone(&self) -> Self {
        Self {
            vertices: self.vertices.empty_clone(),
            faces: FaceStore::new(),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::new(),
            boundary_labels: HashMap::new(),
        }
    }

    /// Create an empty clone of this mesh with reserved capacities for vertices,
    /// faces, and boundary labels.
    #[must_use]
    pub fn empty_clone_with_capacity(&self, vertex_capacity: usize, face_capacity: usize) -> Self {
        Self {
            vertices: self.vertices.empty_clone_with_capacity(vertex_capacity),
            faces: FaceStore::with_capacity(face_capacity),
            edges: None,
            attributes: AttributeStore::new(),
            cells: Vec::new(),
            boundary_labels: HashMap::with_capacity(face_capacity),
        }
    }

    // ── Vertex operations ─────────────────────────────────────────────────

    /// Insert a vertex (deduplicated via spatial hash); returns its ID.
    pub fn add_vertex(&mut self, position: Point3<T>, normal: Vector3<T>) -> VertexId {
        self.edges = None;
        self.vertices.insert_or_weld(position, normal)
    }

    /// Insert a vertex that is guaranteed to receive a fresh [`VertexId`],
    /// bypassing the spatial-hash weld tolerance.
    ///
    /// Use this when splitting a pinch or non-manifold vertex: the new
    /// vertex occupies the *same* position as the original and must NOT
    /// be welded back into it.
    pub fn add_vertex_unique(&mut self, position: Point3<T>, normal: Vector3<T>) -> VertexId {
        self.edges = None;
        self.vertices.insert_unique(position, normal)
    }

    /// Insert a vertex by position only (zero normal).
    pub fn add_vertex_pos(&mut self, position: Point3<T>) -> VertexId {
        self.edges = None;
        self.vertices
            .insert_or_weld(position, Vector3::<T>::zeros())
    }

    /// Number of unique vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    // ── Face operations ───────────────────────────────────────────────────

    /// Add a triangle face from three vertex IDs.
    pub fn add_face(&mut self, v0: VertexId, v1: VertexId, v2: VertexId) -> FaceId {
        self.edges = None;
        self.faces.add_triangle(v0, v1, v2)
    }

    /// Add a triangle face with a region tag.
    pub fn add_face_with_region(
        &mut self,
        v0: VertexId,
        v1: VertexId,
        v2: VertexId,
        region: RegionId,
    ) -> FaceId {
        self.edges = None;
        self.faces.add_triangle_with_region(v0, v1, v2, region)
    }

    /// Number of faces.
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Flip the winding order of all faces (swap v1 <-> v2 on every triangle).
    ///
    /// Call this after building a mesh whose face-construction algorithm
    /// produces consistent *inward* normals, to obtain outward normals.
    pub fn flip_faces(&mut self) {
        self.edges = None;
        self.faces
            .iter_mut()
            .for_each(crate::infrastructure::storage::face_store::FaceData::flip);
    }

    // ── Volumetric cell operations ────────────────────────────────────────

    /// Add a volumetric cell.
    pub fn add_cell(&mut self, c: Cell) {
        self.cells.push(c);
    }

    /// Number of cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Immutable slice of all cells.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    // ── Boundary management ─────────────────────────────────────────────

    /// Label a face as a boundary with the given name.
    pub fn mark_boundary(
        &mut self,
        face_id: FaceId,
        label: impl Into<std::borrow::Cow<'static, str>>,
    ) {
        self.boundary_labels.insert(face_id, label.into());
    }

    /// Return the boundary label of a face, if any.
    pub fn boundary_label(&self, face_id: FaceId) -> Option<&str> {
        self.boundary_labels.get(&face_id).map(|c| c.as_ref())
    }

    /// Return face IDs on the geometric boundary (faces belonging to exactly one cell).
    pub fn boundary_faces(&self) -> Vec<FaceId> {
        if self.cells.is_empty() {
            return self.faces.iter_enumerated().map(|(id, _)| id).collect();
        }
        let mut face_cell_count: HashMap<FaceId, usize> = HashMap::with_capacity(self.faces.len());
        for cell in &self.cells {
            for &fv_idx in &cell.faces {
                // In IndexedMesh, Cell.faces holds FaceId cast as usize currently ? wait:
                let id = FaceId::from_usize(fv_idx);
                *face_cell_count.entry(id).or_insert(0) += 1;
            }
        }
        let mut result: Vec<FaceId> = face_cell_count
            .into_iter()
            .filter(|&(_, count)| count == 1)
            .map(|(id, _)| id)
            .collect();
        result.sort_unstable();
        result
    }

    /// Extrude the true subset of boundary faces into a strictly 2D-manifold `IndexedMesh` B-Rep.
    /// This removes all interior volumetric cells and perfectly isolates the structural hull.
    /// Returns an independent, unlinked mesh containing only the outer topologically closed shell,
    /// suitable for outward normal alignment (`orient_outward`) and `.stl` visualization export.
    pub fn extract_boundary_mesh(&self) -> Self {
        let mut b_mesh = Self::new();
        let b_faces = self.boundary_faces();
        let mut old_to_new_vid = vec![None; self.vertices.len()];

        for &fid in &b_faces {
            let face = self.faces.get(fid);
            let mut new_vids = [VertexId::default(); 3];
            for k in 0..3 {
                let old_vid = face.vertices[k];
                let old_idx = old_vid.as_usize();
                new_vids[k] = if let Some(new_vid) = old_to_new_vid[old_idx] {
                    new_vid
                } else {
                    let new_vid = b_mesh.add_vertex_pos(*self.vertices.position(old_vid));
                    old_to_new_vid[old_idx] = Some(new_vid);
                    new_vid
                };
            }
            b_mesh.add_face(new_vids[0], new_vids[1], new_vids[2]);
        }
        b_mesh
    }

    // ── Edge / adjacency access ───────────────────────────────────────────

    /// Get (or lazily build) the edge store.
    pub fn edges(&mut self) -> &EdgeStore {
        if self.edges.is_none() {
            self.edges = Some(EdgeStore::from_face_store(&self.faces));
        }
        self.edges
            .as_ref()
            .expect("invariant: edges is Some after the if-branch sets it")
    }

    /// Force rebuild of edge adjacency.
    pub fn rebuild_edges(&mut self) {
        self.edges = Some(EdgeStore::from_face_store(&self.faces));
    }

    /// Immutable view of the edge store (may be stale).
    pub fn edges_ref(&self) -> Option<&EdgeStore> {
        self.edges.as_ref()
    }
}

mod builder;
mod components;
mod measure;
mod normals;
mod orient;

pub use builder::MeshBuilder;

impl<T: Scalar> Default for IndexedMesh<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
