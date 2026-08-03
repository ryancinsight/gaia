//! Hex-to-Tet mesh decomposition
//!
//! Robust decomposition of 8-node hexahedra into tetrahedra.
//!
//! This module is a **volume/FEM tool** — intentional `Mesh<T>` usage for
//! hexahedral/tetrahedral cell topology.

use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::{Cell, ElementType};
use crate::infrastructure::storage::face_store::FaceStore;
use hashbrown::{hash_map::Entry, HashMap};

type TriKey = [VertexId; 3];

/// Stack-resident decomposition storage for the two supported hexahedron
/// patterns. A hexahedron produces at most six tetrahedra, so a fixed buffer
/// removes one heap allocation per converted cell without changing the public
/// cell representation.
#[derive(Clone, Copy)]
struct TetDecomposition {
    tets: [[VertexId; 4]; 6],
    len: usize,
}

impl TetDecomposition {
    fn from_five(tets: [[VertexId; 4]; 5]) -> Self {
        let mut storage = [[VertexId::default(); 4]; 6];
        storage[..5].copy_from_slice(&tets);
        Self {
            tets: storage,
            len: 5,
        }
    }

    fn from_six(tets: [[VertexId; 4]; 6]) -> Self {
        Self { tets, len: 6 }
    }

    fn as_slice(&self) -> &[[VertexId; 4]] {
        &self.tets[..self.len]
    }
}

const HEX_VERTEX_COUNT: usize = 8;
const HEX_MAX_NEIGHBORS: usize = HEX_VERTEX_COUNT - 1;

/// Bounded undirected adjacency for one hexahedral cell.
///
/// The cell has exactly eight unique vertices, so one vertex can be adjacent
/// to at most the other seven.  Keeping the IDs inline removes eight map and
/// vector allocations from the order-recovery probe.  Neighbor sets are
/// deduplicated during insertion and remain bounded to seven entries, so
/// linear membership is cheaper than maintaining a sorted allocation-free
/// set.
#[derive(Clone, Copy)]
struct HexAdjacency {
    ids: [[VertexId; HEX_MAX_NEIGHBORS]; HEX_VERTEX_COUNT],
    lengths: [usize; HEX_VERTEX_COUNT],
}

impl HexAdjacency {
    fn new() -> Self {
        Self {
            ids: [[VertexId::default(); HEX_MAX_NEIGHBORS]; HEX_VERTEX_COUNT],
            lengths: [0; HEX_VERTEX_COUNT],
        }
    }

    fn add_undirected(
        &mut self,
        vertices: &[VertexId; HEX_VERTEX_COUNT],
        a: VertexId,
        b: VertexId,
    ) {
        self.add_directed(vertices, a, b);
        self.add_directed(vertices, b, a);
    }

    fn add_directed(&mut self, vertices: &[VertexId; HEX_VERTEX_COUNT], a: VertexId, b: VertexId) {
        let Some(index) = vertices.iter().position(|&vertex| vertex == a) else {
            return;
        };
        let length = self.lengths[index];
        if self.ids[index][..length].contains(&b) {
            return;
        }
        let Some(slot) = self.ids[index].get_mut(length) else {
            debug_assert!(
                length < HEX_MAX_NEIGHBORS,
                "hexahedral adjacency exceeded seven neighbors"
            );
            return;
        };
        *slot = b;
        self.lengths[index] += 1;
    }

    fn neighbors(
        &self,
        vertices: &[VertexId; HEX_VERTEX_COUNT],
        target: VertexId,
    ) -> Option<&[VertexId]> {
        let index = vertices.iter().position(|&vertex| vertex == target)?;
        self.ids[index].get(..self.lengths[index])
    }
}

/// Canonicalize a triangle vertex triplet for orientation-invariant hashing.
///
/// # Theorem — Canonical-Key Equivalence
///
/// Two triangles with identical vertex sets but opposite winding map to the same
/// sorted key; triangles with different vertex sets map to different keys.
/// Therefore this key is a complete invariant for unoriented triangle identity,
/// suitable for deduplicating shared faces in hex-to-tet conversion. ∎
#[inline]
fn canonical_tri_key(nodes: [VertexId; 3]) -> TriKey {
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

#[inline]
fn all_unique<const N: usize>(values: &[VertexId; N]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(index, value)| !values[..index].contains(value))
}

/// Converter for decomposing hexahedral meshes into tetrahedral ones
pub struct HexToTetConverter;

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
        let mut face_map: HashMap<TriKey, FaceId> = HashMap::with_capacity(mesh.faces.len() * 3);

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
                    {
                        if let Some(tets) =
                            Self::select_hex_decomposition(mesh, recovered_order, volume_tol)
                        {
                            for &nodes in tets.as_slice() {
                                Self::add_tet(&mut new_mesh, &mut face_map, nodes);
                            }
                            decomposed = true;
                        }
                    }

                    if !decomposed {
                        if let Some(tets) =
                            Self::select_hex_decomposition(mesh, hex_vertices, volume_tol)
                        {
                            for &nodes in tets.as_slice() {
                                Self::add_tet(&mut new_mesh, &mut face_map, nodes);
                            }
                            decomposed = true;
                        }
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

    fn collect_unique_hex_vertices(cell: &Cell, faces: &FaceStore) -> Option<[VertexId; 8]> {
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

    fn tet_six_volume<T: Scalar>(mesh: &IndexedMesh<T>, nodes: [VertexId; 4]) -> T {
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
        for i in 0..4 {
            for j in (i + 1)..4 {
                if nodes[i] == nodes[j] {
                    return false;
                }
            }
        }
        Self::tet_six_volume(mesh, nodes) > volume_tol
    }

    fn add_tet<T: Scalar>(
        mesh: &mut IndexedMesh<T>,
        face_map: &mut HashMap<TriKey, FaceId>,
        nodes: [VertexId; 4],
    ) {
        let f0 = Self::add_tri_face(mesh, face_map, [nodes[0], nodes[1], nodes[2]]).as_usize();
        let f1 = Self::add_tri_face(mesh, face_map, [nodes[0], nodes[1], nodes[3]]).as_usize();
        let f2 = Self::add_tri_face(mesh, face_map, [nodes[0], nodes[2], nodes[3]]).as_usize();
        let f3 = Self::add_tri_face(mesh, face_map, [nodes[1], nodes[2], nodes[3]]).as_usize();
        mesh.add_cell(Cell::tetrahedron(f0, f1, f2, f3));
    }

    fn hex_five_tet_pattern(order: [VertexId; 8]) -> [[VertexId; 4]; 5] {
        [
            [order[0], order[1], order[3], order[4]],
            [order[1], order[2], order[3], order[6]],
            [order[4], order[7], order[6], order[3]],
            [order[4], order[6], order[5], order[1]],
            [order[1], order[3], order[4], order[6]],
        ]
    }

    fn hex_six_tet_pattern(order: [VertexId; 8]) -> [[VertexId; 4]; 6] {
        [
            [order[0], order[1], order[2], order[6]],
            [order[0], order[2], order[3], order[6]],
            [order[0], order[3], order[7], order[6]],
            [order[0], order[7], order[4], order[6]],
            [order[0], order[4], order[5], order[6]],
            [order[0], order[5], order[1], order[6]],
        ]
    }

    fn decomposition_min_volume<T: Scalar>(
        mesh: &IndexedMesh<T>,
        tets: &[[VertexId; 4]],
        volume_tol: T,
    ) -> Option<T> {
        let mut min_vol: Option<T> = None;
        for nodes in tets {
            if !Self::is_non_degenerate_tet(mesh, *nodes, volume_tol) {
                return None;
            }
            let six_v = Self::tet_six_volume(mesh, *nodes);
            min_vol = Some(match min_vol {
                Some(v) => {
                    if v < six_v {
                        v
                    } else {
                        six_v
                    }
                }
                None => six_v,
            });
        }
        min_vol
    }

    fn select_hex_decomposition<T: Scalar>(
        mesh: &IndexedMesh<T>,
        order: [VertexId; 8],
        volume_tol: T,
    ) -> Option<TetDecomposition> {
        let five = Self::hex_five_tet_pattern(order);
        let six = Self::hex_six_tet_pattern(order);
        let q5 = Self::decomposition_min_volume(mesh, &five, volume_tol);
        let q6 = Self::decomposition_min_volume(mesh, &six, volume_tol);

        match (q5, q6) {
            (Some(v5), Some(v6)) => {
                if v5 >= v6 {
                    Some(TetDecomposition::from_five(five))
                } else {
                    Some(TetDecomposition::from_six(six))
                }
            }
            (Some(_), None) => Some(TetDecomposition::from_five(five)),
            (None, Some(_)) => Some(TetDecomposition::from_six(six)),
            (None, None) => None,
        }
    }

    fn common_neighbor_excluding(
        vertices: &[VertexId; HEX_VERTEX_COUNT],
        adjacency: &HexAdjacency,
        a: VertexId,
        b: VertexId,
        excluded: &[VertexId],
    ) -> Option<VertexId> {
        let a_neighbors = adjacency.neighbors(vertices, a)?;
        let b_neighbors = adjacency.neighbors(vertices, b)?;
        let mut candidate = None;
        for &n in a_neighbors {
            if Self::contains_neighbor(b_neighbors, n) && !excluded.contains(&n) {
                if candidate.is_some() {
                    return None;
                }
                candidate = Some(n);
            }
        }
        candidate
    }

    /// Membership query on bounded adjacency vectors.
    ///
    /// # Theorem — Bounded Neighbor Membership
    ///
    /// Each adjacency vector has at most seven entries, so linear membership
    /// has a fixed small bound and avoids sorting work in every recovered cell.
    /// The insertion path deduplicates entries, but duplicate handling is not
    /// required for this membership contract. ∎
    #[inline]
    fn contains_neighbor(neighbors: &[VertexId], needle: VertexId) -> bool {
        neighbors.contains(&needle)
    }

    fn build_hex_adjacency(
        cell: &Cell,
        vertices: &[VertexId; HEX_VERTEX_COUNT],
        faces: &FaceStore,
    ) -> HexAdjacency {
        let mut adjacency = HexAdjacency::new();
        for &f_idx_raw in &cell.faces {
            let f_idx = FaceId::from_usize(f_idx_raw);
            let face = faces.get(f_idx);
            let n = face.vertices.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                adjacency.add_undirected(vertices, face.vertices[i], face.vertices[(i + 1) % n]);
            }
        }
        adjacency
    }

    fn recover_hex_vertex_order<T: Scalar>(
        cell: &Cell,
        mesh: &IndexedMesh<T>,
        volume_tol: T,
    ) -> Option<[VertexId; HEX_VERTEX_COUNT]> {
        let vertices = Self::collect_unique_hex_vertices(cell, &mesh.faces)?;
        let adjacency = Self::build_hex_adjacency(cell, &vertices, &mesh.faces);

        let perms = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];

        let mut best_order: Option<[VertexId; 8]> = None;
        let mut best_quality: Option<T> = None;

        for &v0 in &vertices {
            let Some(neigh) = adjacency.neighbors(&vertices, v0) else {
                continue;
            };
            if neigh.len() != 3 {
                continue;
            }

            for perm in &perms {
                let v1 = neigh[perm[0]];
                let v3 = neigh[perm[1]];
                let v4 = neigh[perm[2]];

                let Some(v2) =
                    Self::common_neighbor_excluding(&vertices, &adjacency, v1, v3, &[v0, v4])
                else {
                    continue;
                };
                let Some(v5) =
                    Self::common_neighbor_excluding(&vertices, &adjacency, v1, v4, &[v0, v3])
                else {
                    continue;
                };
                let Some(v7) =
                    Self::common_neighbor_excluding(&vertices, &adjacency, v3, v4, &[v0, v1])
                else {
                    continue;
                };

                let Some(n2) = adjacency.neighbors(&vertices, v2) else {
                    continue;
                };
                let Some(n5) = adjacency.neighbors(&vertices, v5) else {
                    continue;
                };
                let Some(n7) = adjacency.neighbors(&vertices, v7) else {
                    continue;
                };

                let mut v6_candidate = None;
                for &n in n2 {
                    if Self::contains_neighbor(n5, n)
                        && Self::contains_neighbor(n7, n)
                        && n != v0
                        && n != v1
                        && n != v2
                        && n != v3
                        && n != v4
                        && n != v5
                        && n != v7
                    {
                        if v6_candidate.is_some() {
                            v6_candidate = None;
                            break;
                        }
                        v6_candidate = Some(n);
                    }
                }
                let Some(v6) = v6_candidate else {
                    continue;
                };

                let order = [v0, v1, v2, v3, v4, v5, v6, v7];
                if !all_unique(&order) {
                    continue;
                }

                let Some(tets) = Self::select_hex_decomposition(mesh, order, volume_tol) else {
                    continue;
                };
                let quality = tets
                    .as_slice()
                    .iter()
                    .map(|nodes| Self::tet_six_volume(mesh, *nodes))
                    .fold(
                        eunomia::RealField::max_value(),
                        |a, b| if a < b { a } else { b },
                    );

                if best_quality.is_none_or(|best| quality > best) {
                    best_quality = Some(quality);
                    best_order = Some(order);
                }
            }
        }

        best_order
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

#[cfg(test)]
mod tests {
    use super::canonical_tri_key;
    use super::HexToTetConverter;
    use crate::domain::core::index::FaceId;
    use crate::domain::core::index::VertexId;
    use crate::domain::grid::StructuredHexGridBuilder;
    use crate::domain::mesh::IndexedMesh;
    use crate::domain::topology::ElementType;

    fn tet_six_volume(mesh: &IndexedMesh<f64>, cell: &crate::domain::topology::Cell) -> f64 {
        let mut vertices = Vec::new();
        let mut seen: hashbrown::HashSet<_> = hashbrown::HashSet::new();
        for &f_idx_raw in &cell.faces {
            let f_idx = FaceId::from_usize(f_idx_raw);
            let face = mesh.faces.get(f_idx);
            for &v_idx in &face.vertices {
                if seen.insert(v_idx) {
                    vertices.push(v_idx);
                }
            }
        }
        assert_eq!(
            vertices.len(),
            4,
            "Converted tetrahedron must have 4 unique vertices"
        );

        let p0 = mesh.vertices.position(vertices[0]).coords;
        let p1 = mesh.vertices.position(vertices[1]).coords;
        let p2 = mesh.vertices.position(vertices[2]).coords;
        let p3 = mesh.vertices.position(vertices[3]).coords;
        (p1 - p0).cross(p2 - p0).dot(p3 - p0).abs()
    }

    fn assert_no_degenerate_tets(mesh: &IndexedMesh<f64>) {
        let bounds = mesh.bounding_box();
        let length_scale = (bounds.max.coords - bounds.min.coords).norm();
        let volume_tol = length_scale.powi(3) * 1e-12;

        for (i, cell) in mesh.cells().iter().enumerate() {
            if cell.element_type != ElementType::Tetrahedron {
                continue;
            }
            let six_v = tet_six_volume(mesh, cell);
            assert!(
                six_v > volume_tol,
                "Degenerate tetrahedron at cell {i} with 6V={six_v:.3e}, tol={volume_tol:.3e}"
            );
        }
    }

    #[test]
    fn structured_hex_mesh_converts_to_non_degenerate_tets() {
        let hex_mesh = StructuredHexGridBuilder::new(4, 4, 4).build();
        let tet_mesh = HexToTetConverter::convert(&hex_mesh);

        assert_eq!(tet_mesh.cell_count(), 4 * 4 * 4 * 5);
        assert!(tet_mesh
            .cells()
            .iter()
            .all(|c| c.element_type == ElementType::Tetrahedron));
        assert_no_degenerate_tets(&tet_mesh);
    }

    #[test]
    fn branching_mesh_conversion_avoids_degenerate_tets() {
        // Use a larger structured grid to exercise non-trivial tet conversion.
        let hex_mesh = StructuredHexGridBuilder::new(6, 4, 4).build();
        let tet_mesh = HexToTetConverter::convert(&hex_mesh);

        assert_eq!(tet_mesh.cell_count(), 6 * 4 * 4 * 5);
        assert!(tet_mesh
            .cells()
            .iter()
            .all(|c| c.element_type == ElementType::Tetrahedron));
        assert_no_degenerate_tets(&tet_mesh);
    }

    #[test]
    fn adversarial_canonical_tri_key_is_orientation_invariant() {
        let a = VertexId::new(10);
        let b = VertexId::new(2);
        let c = VertexId::new(7);
        let k1 = canonical_tri_key([a, b, c]);
        let k2 = canonical_tri_key([c, b, a]);
        let k3 = canonical_tri_key([b, a, c]);
        assert_eq!(k1, k2);
        assert_eq!(k1, k3);
    }

    #[test]
    fn adversarial_neighbor_contains_matches_linear_membership() {
        let mut v = vec![
            VertexId::new(9),
            VertexId::new(1),
            VertexId::new(7),
            VertexId::new(3),
            VertexId::new(3),
            VertexId::new(2),
        ];
        v.sort_unstable_by_key(|id| id.as_usize());
        v.dedup();
        for probe in 0..12 {
            let p = VertexId::new(probe);
            let linear = v.contains(&p);
            let bounded = HexToTetConverter::contains_neighbor(&v, p);
            assert_eq!(
                bounded, linear,
                "bounded membership must match linear membership"
            );
        }
    }
}
