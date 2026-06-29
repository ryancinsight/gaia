//! # Branded Half-Edge Mesh
//!
//! Exposes the `'id`-branded `HalfEdgeMesh` topology kernel.
//! Reading and writing elements of the mesh requires a matching [`GhostToken<'id>`],
//! which is enforced at compile-time with zero runtime cost.
//!
//! ## GhostCell brand contract
//!
//! All mutable access to element data goes through `&mut GhostToken<'id>`.
//! All immutable access goes through `&GhostToken<'id>`.  The `'id` brand is
//! introduced by [`with_mesh`] and cannot escape its closure, so the compiler
//! statically prevents cross-mesh token confusion.

use crate::domain::core::index::{FaceKey, HalfEdgeKey, PatchKey, VertexKey};
use crate::domain::core::scalar::Point3r;
use crate::domain::topology::halfedge::{BoundaryPatch, FaceData, HalfEdgeData, VertexData};
use crate::infrastructure::permission::{GhostCell, GhostToken};
use crate::infrastructure::storage::slotmap_pool::GhostSlotPool;
use slotmap::SlotMap;

/// A branded half-edge surface mesh.
///
/// Ties elements to a compile-time invariant lifetime `'id` (the brand).
/// Direct mutable access requires `&mut GhostToken<'id>`, while reads require `&GhostToken<'id>`.
///
/// # Entry point
///
/// Always create through [`with_mesh`]:
///
/// ```rust,ignore
/// use gaia::with_mesh;
///
/// let result = with_mesh(|mut mesh, mut token| {
///     let vk = mesh.add_vertex(leto::geometry::Point3::origin(), &mut token);
///     mesh.vertex_count()
/// });
/// assert_eq!(result, 1);
/// ```
pub struct HalfEdgeMesh<'id> {
    /// Storage pool for vertices.
    pub vertices: GhostSlotPool<'id, VertexKey, VertexData>,
    /// Storage pool for directed half-edges.
    pub half_edges: GhostSlotPool<'id, HalfEdgeKey, HalfEdgeData>,
    /// Storage pool for faces.
    pub faces: GhostSlotPool<'id, FaceKey, FaceData>,
    /// Boundary patch names.
    pub patches: SlotMap<PatchKey, BoundaryPatch>,
}

impl<'id> HalfEdgeMesh<'id> {
    /// Create a new empty half-edge mesh.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: GhostSlotPool::new(),
            half_edges: GhostSlotPool::new(),
            faces: GhostSlotPool::new(),
            patches: SlotMap::with_key(),
        }
    }

    // ── Capacity ───────────────────────────────────────────────────────────

    /// Number of live vertices.
    #[inline]
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of live directed half-edges.
    #[inline]
    #[must_use]
    pub fn half_edge_count(&self) -> usize {
        self.half_edges.len()
    }

    /// Number of live faces.
    #[inline]
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Returns `true` if the mesh has no vertices, half-edges, or faces.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() && self.half_edges.is_empty() && self.faces.is_empty()
    }

    // ── Iterators ──────────────────────────────────────────────────────────

    /// Iterator over all live vertex keys.
    #[inline]
    pub fn vertex_keys(&self) -> impl Iterator<Item = VertexKey> + use<'_, 'id> {
        self.vertices.keys()
    }

    /// Iterator over all live directed half-edge keys.
    #[inline]
    pub fn half_edge_keys(&self) -> impl Iterator<Item = HalfEdgeKey> + use<'_, 'id> {
        self.half_edges.keys()
    }

    /// Iterator over all live face keys.
    #[inline]
    pub fn face_keys(&self) -> impl Iterator<Item = FaceKey> + use<'_, 'id> {
        self.faces.keys()
    }

    // ── Read access ────────────────────────────────────────────────────────

    /// Look up vertex position inside the
    /// [`crate::infrastructure::permission::GhostCell`].
    #[inline]
    #[must_use]
    pub fn vertex_pos(&self, vk: VertexKey, token: &GhostToken<'id>) -> Option<Point3r> {
        self.vertices
            .get(vk)
            .map(|cell| cell.borrow(token).position)
    }

    /// Trace the face's half-edge loop to collect its bounding vertex keys.
    ///
    /// Returns an empty `Vec` if `fk` is stale or the loop is degenerate.
    #[must_use]
    pub fn face_vertices(&self, fk: FaceKey, token: &GhostToken<'id>) -> Vec<VertexKey> {
        let start_he = match self.faces.get(fk) {
            Some(cell) => cell.borrow(token).half_edge,
            None => return Vec::new(),
        };
        let mut vertices = Vec::new();
        let mut current_he = start_he;
        while let Some(cell) = self.half_edges.get(current_he) {
            let he_data = cell.borrow(token);
            vertices.push(he_data.vertex);
            current_he = he_data.next;
            if current_he == start_he {
                break;
            }
        }
        vertices
    }

    // ── Mutation entry points ──────────────────────────────────────────────

    /// Insert a vertex with `data` and return its key.
    ///
    /// Requires `&mut GhostToken<'id>` to serialise mutation through the brand.
    #[inline]
    pub fn add_vertex(&mut self, data: VertexData, _token: &mut GhostToken<'id>) -> VertexKey {
        self.vertices.insert(GhostCell::new(data))
    }

    /// Insert a directed half-edge with `data` and return its key.
    ///
    /// Requires `&mut GhostToken<'id>` to serialise mutation through the brand.
    #[inline]
    pub fn add_half_edge(
        &mut self,
        data: HalfEdgeData,
        _token: &mut GhostToken<'id>,
    ) -> HalfEdgeKey {
        self.half_edges.insert(GhostCell::new(data))
    }

    /// Insert a face with `data` and return its key.
    ///
    /// Requires `&mut GhostToken<'id>` to serialise mutation through the brand.
    #[inline]
    pub fn add_face(&mut self, data: FaceData, _token: &mut GhostToken<'id>) -> FaceKey {
        self.faces.insert(GhostCell::new(data))
    }

    /// Add a named boundary patch and return its key.
    #[inline]
    pub fn add_patch(&mut self, patch: BoundaryPatch) -> PatchKey {
        self.patches.insert(patch)
    }
}

impl Default for HalfEdgeMesh<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HalfEdgeMesh<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HalfEdgeMesh")
            .field("vertices_count", &self.vertices.len())
            .field("half_edges_count", &self.half_edges.len())
            .field("faces_count", &self.faces.len())
            .field("patches_count", &self.patches.len())
            .finish()
    }
}

/// Canonical entry point — introduces the GhostCell brand `'id`.
///
/// The token cannot escape the closure; return value `R` carries extracted data.
///
/// # Example
///
/// ```rust,ignore
/// use gaia::with_mesh;
///
/// let n_verts = with_mesh(|_mesh, _token| 0_usize);
/// assert_eq!(n_verts, 0);
/// ```
pub fn with_mesh<F, R>(f: F) -> R
where
    F: for<'id> FnOnce(HalfEdgeMesh<'id>, GhostToken<'id>) -> R,
{
    GhostToken::new(|token| f(HalfEdgeMesh::new(), token))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::index::HalfEdgeKey;
    use crate::domain::topology::halfedge::{FaceData, HalfEdgeData, VertexData};
    use leto::geometry::{Point3, UnitVector3, Vector3};

    #[test]
    fn empty_mesh_invariants() {
        with_mesh(|mesh, _token| {
            assert!(mesh.is_empty());
            assert_eq!(mesh.vertex_count(), 0);
            assert_eq!(mesh.half_edge_count(), 0);
            assert_eq!(mesh.face_count(), 0);
        });
    }

    #[test]
    fn add_vertex_roundtrip() {
        with_mesh(|mut mesh, mut token| {
            let data = VertexData::new(Point3::new(1.0, 2.0, 3.0));
            let vk = mesh.add_vertex(data, &mut token);
            assert_eq!(mesh.vertex_count(), 1);
            let pos = mesh.vertex_pos(vk, &token).expect("vertex must be live");
            assert_eq!(pos.x, 1.0);
            assert_eq!(pos.y, 2.0);
            assert_eq!(pos.z, 3.0);
        });
    }

    #[test]
    fn vertex_keys_iterator_yields_all_live() {
        with_mesh(|mut mesh, mut token| {
            let _vk0 = mesh.add_vertex(VertexData::new(Point3::origin()), &mut token);
            let _vk1 = mesh.add_vertex(VertexData::new(Point3::new(1.0, 0.0, 0.0)), &mut token);
            let keys: Vec<_> = mesh.vertex_keys().collect();
            assert_eq!(keys.len(), 2);
        });
    }

    #[test]
    fn half_edge_keys_iterator() {
        with_mesh(|mut mesh, mut token| {
            // Build a minimal valid half-edge pair (twin involution).
            // We insert two placeholder half-edges; fixing twin is not required
            // for key iteration correctness.
            let dummy_vk = mesh.add_vertex(VertexData::new(Point3::origin()), &mut token);

            let he0_key = HalfEdgeKey::default(); // placeholder, not in map yet
            let he1_key = HalfEdgeKey::default();

            let he0_data = HalfEdgeData {
                vertex: dummy_vk,
                face: None,
                twin: he1_key,
                next: he0_key,
                prev: he0_key,
            };
            let he1_data = HalfEdgeData {
                vertex: dummy_vk,
                face: None,
                twin: he0_key,
                next: he1_key,
                prev: he1_key,
            };
            mesh.add_half_edge(he0_data, &mut token);
            mesh.add_half_edge(he1_data, &mut token);
            assert_eq!(mesh.half_edge_count(), 2);
            let he_keys: Vec<_> = mesh.half_edge_keys().collect();
            assert_eq!(he_keys.len(), 2);
        });
    }

    #[test]
    fn add_face_and_trace_vertices() {
        with_mesh(|mut mesh, mut token| {
            let va = mesh.add_vertex(VertexData::new(Point3::new(0.0, 0.0, 0.0)), &mut token);
            let vb = mesh.add_vertex(VertexData::new(Point3::new(1.0, 0.0, 0.0)), &mut token);
            let vc = mesh.add_vertex(VertexData::new(Point3::new(0.0, 1.0, 0.0)), &mut token);

            // Insert 3 half-edges with a consistent loop: he0→he1→he2→he0
            // We must insert them to get real keys before wiring.
            let dummy_key = HalfEdgeKey::default();
            let he0_key = mesh.add_half_edge(
                HalfEdgeData {
                    vertex: va,
                    face: None,
                    twin: dummy_key,
                    next: dummy_key,
                    prev: dummy_key,
                },
                &mut token,
            );
            let he1_key = mesh.add_half_edge(
                HalfEdgeData {
                    vertex: vb,
                    face: None,
                    twin: dummy_key,
                    next: dummy_key,
                    prev: dummy_key,
                },
                &mut token,
            );
            let he2_key = mesh.add_half_edge(
                HalfEdgeData {
                    vertex: vc,
                    face: None,
                    twin: dummy_key,
                    next: dummy_key,
                    prev: dummy_key,
                },
                &mut token,
            );

            // Wire loop: he0.next = he1, he1.next = he2, he2.next = he0
            mesh.half_edges[he0_key].borrow_mut(&mut token).next = he1_key;
            mesh.half_edges[he1_key].borrow_mut(&mut token).next = he2_key;
            mesh.half_edges[he2_key].borrow_mut(&mut token).next = he0_key;

            let normal = UnitVector3::new_normalize(Vector3::z());
            let fk = mesh.add_face(FaceData::new(he0_key, normal), &mut token);

            let face_verts = mesh.face_vertices(fk, &token);
            // face_vertices starts from he0 (vertex=va), then he1.next (vb), then he2.next (vc)
            assert_eq!(face_verts.len(), 3);
            assert_eq!(face_verts[0], va);
            assert_eq!(face_verts[1], vb);
            assert_eq!(face_verts[2], vc);
        });
    }

    #[test]
    fn debug_output_includes_counts() {
        with_mesh(|mesh, _token| {
            let s = format!("{mesh:?}");
            assert!(s.contains("vertices_count"));
            assert!(s.contains("half_edges_count"));
            assert!(s.contains("faces_count"));
        });
    }
}
