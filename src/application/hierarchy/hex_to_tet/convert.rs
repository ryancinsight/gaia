//! Hex-to-tet conversion driver and tetrahedron emission.

use hashbrown::{hash_map::Entry, HashMap};

use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::{Cell, ElementType};
use crate::infrastructure::storage::face_store::FaceStore;

use super::types::{TriKey, HEX_MAX_TETRAHEDRA, TETRAHEDRON_FACE_COUNT};
use super::HexToTetConverter;

/// Canonicalize a triangle vertex triplet for orientation-invariant hashing.
///
/// # Theorem — Canonical-Key Equivalence
///
/// Two triangles with identical vertex sets but opposite winding map to the same
/// sorted key; triangles with different vertex sets map to different keys.
/// Therefore this key is a complete invariant for unoriented triangle identity,
/// suitable for deduplicating shared faces in hex-to-tet conversion. ∎
#[inline]
pub(super) fn canonical_tri_key(nodes: [VertexId; 3]) -> TriKey {
    let mut key = nodes;
    if key[0] > key[1] {
        key.swap(0, 1);
    }
    if key[1] > key[2] {
        key.swap(1, 2);
    }
    if key[0] > key[1] {
        key.swap(0, 1);
    }
    key
}

/// Upper bound for the face identities inserted during conversion.
///
/// A hexahedral cell produces at most six tetrahedra, and each tetrahedron
/// submits four triangular faces to the registry. Preserved cells submit each
/// of their existing faces once. Shared faces are deduplicated after this
/// reservation, so the bound is sufficient without depending on scalar `T`.
pub(super) fn estimated_face_map_capacity(cells: &[Cell]) -> usize {
    cells.iter().fold(0, |capacity, cell| {
        let additions = match cell.element_type {
            ElementType::Hexahedron => HEX_MAX_TETRAHEDRA * TETRAHEDRON_FACE_COUNT,
            _ => cell.faces.len(),
        };
        capacity.saturating_add(additions)
    })
}

impl HexToTetConverter {
    /// Decompose all hexahedral cells in a mesh into tetrahedra
    pub fn convert<T: Scalar>(mesh: &IndexedMesh<T>) -> IndexedMesh<T> {
        let mut new_mesh =
            mesh.empty_clone_with_capacity(mesh.vertex_count(), mesh.faces.len() * 3);
        new_mesh.cells.reserve(mesh.cells.len() * 6);

        // 1. Copy all vertices exactly (retaining their IDs)
        new_mesh.vertices = mesh.vertices.clone();

        // 2. Identify boundary faces and map them
        // Key: canonical triangle vertex triplet, Value: new FaceId
        let mut face_map: HashMap<TriKey, FaceId> =
            HashMap::with_capacity(estimated_face_map_capacity(&mesh.cells));

        // 3. Process cells
        for c in &mesh.cells {
            if c.element_type == ElementType::Hexahedron {
                if let Some(hex_vertices) = Self::collect_unique_hex_vertices(c, &mesh.faces) {
                    let length_scale = Self::characteristic_length(mesh, &hex_vertices);
                    let tol_factor = <T as Scalar>::from_f64(1e-12);
                    let volume_tol = length_scale * length_scale * length_scale * tol_factor;

                    let mut decomposed = false;

                    // Prefer recovered topological ordering to avoid decomposition
                    // bias from incidental face/vertex iteration order.
                    if let Some(recovered_order) =
                        Self::recover_hex_vertex_order(c, mesh, volume_tol)
                        && let Some((tets, _)) =
                            Self::select_hex_decomposition(mesh, recovered_order, volume_tol)
                    {
                        for &nodes in tets.as_slice() {
                            Self::add_tet(&mut new_mesh, &mut face_map, nodes);
                        }
                        decomposed = true;
                    }

                    if !decomposed
                        && let Some((tets, _)) =
                            Self::select_hex_decomposition(mesh, hex_vertices, volume_tol)
                    {
                        for &nodes in tets.as_slice() {
                            Self::add_tet(&mut new_mesh, &mut face_map, nodes);
                        }
                        decomposed = true;
                    }

                    if !decomposed {
                        // Final safeguard: keep only non-degenerate tetrahedra.
                        for nodes in Self::hex_six_tet_pattern(hex_vertices) {
                            if Self::is_non_degenerate_tet(mesh, nodes, volume_tol) {
                                Self::add_tet(&mut new_mesh, &mut face_map, nodes);
                            }
                        }
                    }
                }
            } else {
                // Keep other cells (e.g. already tetrahedra), remapping faces
                let mut new_faces = Vec::with_capacity(c.faces.len());
                for &f_idx_raw in &c.faces {
                    let f_idx = FaceId::from_usize(f_idx_raw);
                    let face = mesh.faces.get(f_idx);
                    let nf = Self::add_tri_face(
                        &mut new_mesh,
                        &mut face_map,
                        [face.vertices[0], face.vertices[1], face.vertices[2]],
                    );
                    new_faces.push(nf.as_usize());
                }
                let mut new_cell = c.clone();
                new_cell.faces = new_faces;
                new_mesh.add_cell(new_cell);
            }
        }

        // 4. Transfer and decompose boundary markers
        for (&f_idx, label) in &mesh.boundary_labels {
            let face = mesh.faces.get(f_idx);
            let nf = Self::get_tri_face_idx(
                &face_map,
                [face.vertices[0], face.vertices[1], face.vertices[2]],
            );
            if let Some(idx) = nf {
                new_mesh.mark_boundary(idx, label.clone());
            }
        }

        new_mesh
    }

    pub(super) fn collect_unique_hex_vertices(
        cell: &Cell,
        faces: &FaceStore,
    ) -> Option<[VertexId; 8]> {
        let mut vertices = [VertexId::default(); 8];
        let mut vertex_count = 0;
        for &f_idx_raw in &cell.faces {
            let f_idx = FaceId::from_usize(f_idx_raw);
            let face = faces.get(f_idx);
            for &v_idx in &face.vertices {
                if vertices[..vertex_count].contains(&v_idx) {
                    continue;
                }
                let slot = vertices.get_mut(vertex_count)?;
                *slot = v_idx;
                vertex_count += 1;
            }
        }
        (vertex_count == vertices.len()).then_some(vertices)
    }

    fn characteristic_length<T: Scalar>(mesh: &IndexedMesh<T>, vertices: &[VertexId]) -> T {
        let mut max_dist_sq = <T as eunomia::NumericElement>::ZERO;
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let pi = mesh.vertices.position(vertices[i]).coords;
                let pj = mesh.vertices.position(vertices[j]).coords;
                let dist_sq = (pj - pi).norm_squared();
                if dist_sq > max_dist_sq {
                    max_dist_sq = dist_sq;
                }
            }
        }
        eunomia::NumericElement::sqrt(max_dist_sq)
    }

    pub(super) fn tet_six_volume<T: Scalar>(mesh: &IndexedMesh<T>, nodes: [VertexId; 4]) -> T {
        let p0 = mesh.vertices.position(nodes[0]).coords;
        let p1 = mesh.vertices.position(nodes[1]).coords;
        let p2 = mesh.vertices.position(nodes[2]).coords;
        let p3 = mesh.vertices.position(nodes[3]).coords;
        eunomia::NumericElement::abs((p1 - p0).cross(p2 - p0).dot(p3 - p0))
    }

    fn is_non_degenerate_tet<T: Scalar>(
        mesh: &IndexedMesh<T>,
        nodes: [VertexId; 4],
        volume_tol: T,
    ) -> bool {
        Self::tet_six_volume_if_valid(mesh, nodes, volume_tol).is_some()
    }

    pub(super) fn tet_six_volume_if_valid<T: Scalar>(
        mesh: &IndexedMesh<T>,
        nodes: [VertexId; 4],
        volume_tol: T,
    ) -> Option<T> {
        for i in 0..4 {
            for j in (i + 1)..4 {
                if nodes[i] == nodes[j] {
                    return None;
                }
            }
        }
        let six_v = Self::tet_six_volume(mesh, nodes);
        (six_v > volume_tol).then_some(six_v)
    }

    /// Add one tetrahedron to `mesh`, deduplicating shared faces via `face_map`.
    ///
    /// ## Face winding theorem
    ///
    /// For a tetrahedron `[v0, v1, v2, v3]` with **positive** signed volume
    /// `(v1−v0)·((v2−v0)×(v3−v0)) > 0`, the outward face normals are:
    ///
    /// | Face (opposite) | Winding | Why outward |
    /// |---|---|---|
    /// | opposite v3 | `[v0, v2, v1]` | n = (v2−v0)×(v1−v0); n·(v3−v0) = −6V < 0 |
    /// | opposite v2 | `[v0, v1, v3]` | n = (v1−v0)×(v3−v0); n·(v2−v0) = −6V < 0 |
    /// | opposite v1 | `[v0, v3, v2]` | n = (v3−v0)×(v2−v0); n·(v1−v0) = −6V < 0 |
    /// | opposite v0 | `[v1, v2, v3]` | n = (v2−v1)×(v3−v1); n·(v0−v1) = −6V < 0 |
    ///
    /// Shared internal faces are deduplicated by canonical key regardless of
    /// the winding passed in, so the first tet's winding is stored and both
    /// incident tets receive the same `FaceId`. ∎
    fn add_tet<T: Scalar>(
        mesh: &mut IndexedMesh<T>,
        face_map: &mut HashMap<TriKey, FaceId>,
        nodes: [VertexId; 4],
    ) {
        let [v0, v1, v2, v3] = nodes;
        // Use outward-facing winding for a positive-volume tet.
        let f0 = Self::add_tri_face(mesh, face_map, [v0, v2, v1]).as_usize(); // opposite v3
        let f1 = Self::add_tri_face(mesh, face_map, [v0, v1, v3]).as_usize(); // opposite v2
        let f2 = Self::add_tri_face(mesh, face_map, [v0, v3, v2]).as_usize(); // opposite v1
        let f3 = Self::add_tri_face(mesh, face_map, [v1, v2, v3]).as_usize(); // opposite v0
        mesh.add_cell(Cell::tetrahedron(f0, f1, f2, f3));
    }

    fn add_tri_face<T: Scalar>(
        mesh: &mut IndexedMesh<T>,
        map: &mut HashMap<TriKey, FaceId>,
        nodes: [VertexId; 3],
    ) -> FaceId {
        let key = canonical_tri_key(nodes);
        match map.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let idx = mesh.add_face(nodes[0], nodes[1], nodes[2]);
                entry.insert(idx);
                idx
            }
        }
    }

    fn get_tri_face_idx(map: &HashMap<TriKey, FaceId>, nodes: [VertexId; 3]) -> Option<FaceId> {
        map.get(&canonical_tri_key(nodes)).copied()
    }
}
