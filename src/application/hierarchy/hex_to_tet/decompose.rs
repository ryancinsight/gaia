//! Hexahedron order recovery and decomposition-pattern selection.

use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::Cell;
use crate::infrastructure::storage::face_store::FaceStore;

use super::types::{HEX_MAX_NEIGHBORS, HEX_VERTEX_COUNT};
use super::HexToTetConverter;

/// Stack-resident decomposition storage for the two supported hexahedron
/// patterns. A hexahedron produces at most six tetrahedra, so a fixed buffer
/// removes one heap allocation per converted cell without changing the public
/// cell representation.
#[derive(Clone, Copy)]
pub(super) struct TetDecomposition {
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

    pub(super) fn as_slice(&self) -> &[[VertexId; 4]] {
        &self.tets[..self.len]
    }
}

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

#[inline]
fn all_unique<const N: usize>(values: &[VertexId; N]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(index, value)| !values[..index].contains(value))
}

impl HexToTetConverter {
    fn hex_five_tet_pattern(order: [VertexId; 8]) -> [[VertexId; 4]; 5] {
        [
            [order[0], order[1], order[3], order[4]],
            [order[1], order[2], order[3], order[6]],
            [order[4], order[7], order[6], order[3]],
            [order[4], order[6], order[5], order[1]],
            [order[1], order[3], order[4], order[6]],
        ]
    }

    pub(super) fn hex_six_tet_pattern(order: [VertexId; 8]) -> [[VertexId; 4]; 6] {
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
            let six_v = Self::tet_six_volume_if_valid(mesh, *nodes, volume_tol)?;
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

    pub(super) fn select_hex_decomposition<T: Scalar>(
        mesh: &IndexedMesh<T>,
        order: [VertexId; 8],
        volume_tol: T,
    ) -> Option<(TetDecomposition, T)> {
        let five = Self::hex_five_tet_pattern(order);
        let six = Self::hex_six_tet_pattern(order);
        let q5 = Self::decomposition_min_volume(mesh, &five, volume_tol);
        let q6 = Self::decomposition_min_volume(mesh, &six, volume_tol);

        match (q5, q6) {
            (Some(v5), Some(v6)) => {
                if v5 >= v6 {
                    Some((TetDecomposition::from_five(five), v5))
                } else {
                    Some((TetDecomposition::from_six(six), v6))
                }
            }
            (Some(v5), None) => Some((TetDecomposition::from_five(five), v5)),
            (None, Some(v6)) => Some((TetDecomposition::from_six(six), v6)),
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
    pub(super) fn contains_neighbor(neighbors: &[VertexId], needle: VertexId) -> bool {
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

    pub(super) fn recover_hex_vertex_order<T: Scalar>(
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

                let Some((_, quality)) = Self::select_hex_decomposition(mesh, order, volume_tol)
                else {
                    continue;
                };

                if best_quality.is_none_or(|best| quality > best) {
                    best_quality = Some(quality);
                    best_order = Some(order);
                }
            }
        }

        best_order
    }
}
