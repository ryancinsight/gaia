//! Hex vertex-order recovery: map an unordered cell vertex set back onto the
//! canonical hexahedron ordering using face adjacency.

use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::Cell;

use super::{all_unique, HexToTetConverter, HEX_VERTEX_COUNT};

impl HexToTetConverter {
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
