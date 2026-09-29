//! Repair pass: pinch (figure-8) vertex splits.

use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::PackedRows;

/// Split pinch vertices via **link-graph component counting**.
///
/// # Theorem — Link Connectivity Criterion
///
/// On a closed orientable 2-manifold, the link of every interior vertex
/// is a single **connected** cycle.  If the link graph has `k > 1`
/// connected components, the face fan around `v` decomposes into `k`
/// topologically-disjoint patches sharing only the apex `v` — a figure-8
/// (or higher-order) pinch vertex.
///
/// **Proof sketch.**  The face fan around `v` is homeomorphic to a disk,
/// whose boundary is the link.  A connected disk has a connected boundary.
/// Multiple link components implies multiple boundary components, which
/// requires a pinched (non-manifold) apex.  ∎
///
/// # Algorithm
///
/// 1. For each vertex `v`, extract link edges `{a, b}` from every face
///    `[v, a, b]` incident to `v`.
/// 2. Build the link graph (adjacency on link vertices via link edges).
/// 3. Count connected components of the link graph via BFS.
/// 4. If `k > 1` components, use face-adjacency BFS on the fan (traversing
///    through all shared link vertices) to partition faces into `k` groups,
///    then split `v` into `k` copies.
///
/// **Complexity:** `O(Σ_v deg(v)) = O(F)`.
///
/// **Why this avoids Difference regressions:**
///
/// In genus-1 Difference results, every vertex — including those at the
/// hole boundary — has a single connected link cycle.  The link wraps once
/// around the boundary without disconnecting.  Only true figure-8 pinch
/// vertices exhibit disconnected link graphs.
pub(super) fn split_figure8_pinch_vertices(mesh: &mut IndexedMesh) -> usize {
    use std::collections::VecDeque;

    // Build vertex → face-index map.
    let mut vertex_faces: hashbrown::HashMap<VertexId, Vec<usize>> =
        hashbrown::HashMap::with_capacity(mesh.vertices.len());
    for (fi, face) in mesh.faces.iter().enumerate() {
        for &v in &face.vertices {
            vertex_faces.entry(v).or_default().push(fi);
        }
    }

    let mut total_splits: usize = 0;
    // The mesh is mutated while vertices are split. Hash-map key order would
    // otherwise make allocation and face-rewrite order process-seed dependent.
    let mut vertices: Vec<VertexId> = vertex_faces.keys().copied().collect();
    vertices.sort_unstable();

    for v in vertices {
        let face_indices = match vertex_faces.get(&v) {
            Some(fi) if fi.len() >= 2 => fi,
            _ => continue,
        };

        // Build per-face link edge info: for face [v, a, b], link edge = {a, b}.
        let n = face_indices.len();
        let mut face_link_edges: Vec<(usize, VertexId, VertexId)> = Vec::with_capacity(n);

        for &fi in face_indices {
            let face = mesh.faces.get(FaceId::from_usize(fi));
            let verts = &face.vertices;
            let pos = verts
                .iter()
                .position(|&vid| vid == v)
                .expect("invariant: face in vertex_faces[v] must contain vertex v");
            let a = verts[(pos + 1) % 3];
            let b = verts[(pos + 2) % 3];
            face_link_edges.push((fi, a, b));
        }

        // Build edge-to-faces map: for each edge (v, w) at vertex v, collect
        // the local face indices that share that edge.  An edge (v, w) is
        // shared by face_i if w ∈ {a_i, b_i}.
        let mut edge_faces: hashbrown::HashMap<VertexId, Vec<usize>> =
            hashbrown::HashMap::with_capacity(n * 2);
        for (local_idx, &(_, a, b)) in face_link_edges.iter().enumerate() {
            edge_faces.entry(a).or_default().push(local_idx);
            edge_faces.entry(b).or_default().push(local_idx);
        }

        // Face-adjacency BFS through manifold edges at v.
        //
        // Two faces are adjacent only if they share an edge (v, w) where that
        // edge has exactly 2 incident faces (manifold).  Non-manifold edges
        // (> 2 faces) block traversal, splitting the fan into separate
        // components.  This catches both classic figure-8 pinches (disjoint
        // link-graph components) and folded-fan pinches (connected link graph
        // with extra edges).
        let mut visited: Vec<bool> = vec![false; n];
        // Rows are unknown in length until the BFS that fills them
        // terminates, so they are appended to one flat buffer plus an
        // offset table (`PackedRows::from_parts`) rather than allocated
        // per component.
        let mut component_values: Vec<usize> = Vec::with_capacity(n);
        let mut component_offsets: Vec<usize> = vec![0];
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

        for start_local in 0..n {
            if visited[start_local] {
                continue;
            }
            queue.clear();
            queue.push_back(start_local);
            visited[start_local] = true;

            while let Some(local_idx) = queue.pop_front() {
                let (fi, a, b) = face_link_edges[local_idx];
                component_values.push(fi);

                // Traverse through edges (v, a) and (v, b), but only if
                // the edge is manifold (shared by exactly 2 faces at v).
                for &w in &[a, b] {
                    if let Some(adj) = edge_faces.get(&w)
                        && adj.len() == 2
                    {
                        // Manifold edge: traverse to the twin face.
                        for &adj_local in adj {
                            if !visited[adj_local] {
                                visited[adj_local] = true;
                                queue.push_back(adj_local);
                            }
                        }
                    }
                    // Non-manifold edge (> 2 faces): do NOT traverse.
                    // This creates a fan-component boundary, splitting
                    // the vertex.
                }
            }
            component_offsets.push(component_values.len());
        }

        let components = PackedRows::from_parts(component_offsets, component_values);
        if components.len() <= 1 {
            continue;
        }

        // Split: duplicate vertex for each additional component.
        let pos = *mesh.vertices.position(v);
        let normal = *mesh.vertices.normal(v);
        for component in components.iter().skip(1) {
            let new_v = mesh.add_vertex_unique(pos, normal);
            for &fi in component {
                let fid = FaceId::from_usize(fi);
                let face_mut = mesh.faces.get_mut(fid);
                for vref in &mut face_mut.vertices {
                    if *vref == v {
                        *vref = new_v;
                    }
                }
            }
            total_splits += 1;
        }
    }
    if total_splits > 0 {
        tracing::debug!(
            "CSG postprocess: split {} pinch vertices (edge-adjacency fan decomposition)",
            total_splits
        );
    }
    total_splits
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::many_single_char_names,
        reason = "standard point and shared-vertex naming in vertex repair tests"
    )]

    use super::*;
    use crate::domain::core::index::RegionId;
    use crate::domain::core::scalar::{Point3r, Vector3r};
    use crate::infrastructure::storage::face_store::FaceData;

    fn up() -> Vector3r {
        Vector3r::new(0.0, 0.0, 1.0)
    }

    /// Two triangles sharing only vertex `v` (no other shared vertex or
    /// edge) are the minimal figure-8 pinch: the link of `v` has two
    /// components, so `v` splits into two distinct vertices at one position.
    #[test]
    fn two_isolated_faces_at_one_vertex_split_via_link_graph() {
        let mut mesh = IndexedMesh::new();
        let v = mesh.add_vertex(Point3r::new(0.0, 0.0, 0.0), up());
        let a = mesh.add_vertex(Point3r::new(1.0, 0.0, 0.0), up());
        let b = mesh.add_vertex(Point3r::new(0.0, 1.0, 0.0), up());
        let c = mesh.add_vertex(Point3r::new(-1.0, 0.0, 0.0), up());
        let d = mesh.add_vertex(Point3r::new(0.0, -1.0, 0.0), up());
        mesh.faces.push(FaceData::new(v, a, b, RegionId::new(0)));
        mesh.faces.push(FaceData::new(v, c, d, RegionId::new(0)));

        let vertices_before = mesh.vertices.len();
        let splits = split_figure8_pinch_vertices(&mut mesh);

        assert_eq!(splits, 1, "exactly one pinch split for a two-component fan");
        assert_eq!(mesh.vertices.len(), vertices_before + 1);
        assert_eq!(mesh.faces.get(FaceId::from_usize(0)).vertices[0], v);
        let split_v = mesh.faces.get(FaceId::from_usize(1)).vertices[0];
        assert_ne!(split_v, v);
        assert_eq!(*mesh.vertices.position(split_v), *mesh.vertices.position(v));
    }

    /// A single connected fan never splits: the link has one component and
    /// the mesh is left untouched.
    #[test]
    fn manifold_fan_is_not_split() {
        let mut mesh = IndexedMesh::new();
        let v = mesh.add_vertex(Point3r::new(0.0, 0.0, 0.0), up());
        let a = mesh.add_vertex(Point3r::new(1.0, 0.0, 0.0), up());
        let b = mesh.add_vertex(Point3r::new(0.0, 1.0, 0.0), up());
        let c = mesh.add_vertex(Point3r::new(-1.0, 0.0, 0.0), up());
        mesh.faces.push(FaceData::new(v, a, b, RegionId::new(0)));
        mesh.faces.push(FaceData::new(v, b, c, RegionId::new(0)));

        let vertices_before = mesh.vertices.len();
        let splits = split_figure8_pinch_vertices(&mut mesh);
        assert_eq!(splits, 0, "manifold fan: no split");
        assert_eq!(mesh.vertices.len(), vertices_before);
    }

    /// Two faces on one undirected edge `{v, a}` with inconsistent winding
    /// (both carry `v -> a`, as before orientation repair) are adjacent, not
    /// a pinch: splitting `v` would open the edge into two boundary edges.
    #[test]
    fn inconsistently_wound_edge_neighbours_are_not_split() {
        let mut mesh = IndexedMesh::new();
        let v = mesh.add_vertex(Point3r::new(0.0, 0.0, 0.0), up());
        let a = mesh.add_vertex(Point3r::new(1.0, 0.0, 0.0), up());
        let b = mesh.add_vertex(Point3r::new(0.5, 1.0, 0.0), up());
        let c = mesh.add_vertex(Point3r::new(0.5, -1.0, 0.0), up());
        let faces = [
            FaceData::new(v, a, b, RegionId::new(0)),
            FaceData::new(v, a, c, RegionId::new(0)),
        ];
        for face in faces {
            mesh.faces.push(face);
        }

        let vertices_before = mesh.vertices.len();
        assert_eq!(split_figure8_pinch_vertices(&mut mesh), 0);
        assert_eq!(mesh.vertices.len(), vertices_before);
        let after: Vec<FaceData> = mesh.faces.iter().copied().collect();
        assert_eq!(after, faces);
    }
}
