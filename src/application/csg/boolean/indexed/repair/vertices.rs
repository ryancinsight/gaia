//! Repair passes: non-manifold pinch/figure-8 vertex splits.

use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::mesh::IndexedMesh;

/// Split non-manifold "pinch" vertices whose face fan forms a figure-8
/// topology (two or more loops sharing one geometric vertex).
///
/// # Theorem (Pinch Vertex Detection via Half-Edge Multiplicity)
///
/// A vertex `v` in a triangle mesh is a *pinch vertex* if and only if some
/// neighbour vertex `w` is the target of more than one outgoing half-edge
/// `v → w` (equivalently, more than one face has the directed edge `v → w`).
///
/// **Proof sketch.**
/// In a closed oriented 2-manifold, every directed half-edge `v → w` belongs
/// to exactly one face, and its twin `w → v` belongs to exactly one other
/// face.  These two faces share the undirected edge `{v,w}` and are
/// manifold-adjacent.
///
/// At a pinch vertex, the face fan around `v` consists of *k ≥ 2* disjoint
/// cycles that share only `v`.  For the two cycles to share *v* while
/// remaining edge-connected internally, they must share at least one
/// neighbour `w` — otherwise they would form separate connected components
/// trivially.  A shared neighbour `w` means two distinct faces (one per
/// cycle) contain the directed half-edge `v → w`, producing a multiplicity
/// `|outgoing[w]| ≥ 2`.  The converse: if `|outgoing[w]| = 1` for every
/// neighbour `w`, each directed half-edge from `v` appears once, and the fan
/// is a single cycle — hence no pinch.  ∎
///
/// # Algorithm
///
/// 1. Build multi-valued half-edge maps `outgoing[w] → Vec<face_index>` and
///    `incoming[w] → Vec<face_index>` for every face around `v`.
/// 2. BFS through face adjacency, but **refuse to traverse** through any
///    neighbour `w` with `|outgoing[w]| > 1` or `|incoming[w]| > 1` — this
///    is a non-manifold edge that bridges two pinch cycles.
/// 3. If the BFS produces `k > 1` connected components, allocate `k − 1`
///    fresh vertices at the same position and reassign face references.
///
/// **Complexity:** `O(Σ_v deg(v)) = O(F)` where *F* is the face count.
pub(super) fn split_non_manifold_vertices(mesh: &mut IndexedMesh) {
    use std::collections::VecDeque;

    // Step 1: build vertex → face-index map.
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

        // Step 2: build multi-valued half-edge adjacency maps.
        //
        // outgoing[w] = list of face indices with directed half-edge v → w.
        // incoming[w] = list of face indices with directed half-edge w → v.
        //
        // In a manifold mesh, each list has length exactly 1.  A length ≥ 2
        // indicates a non-manifold edge through which the BFS must not
        // traverse (see theorem above).
        let mut outgoing: hashbrown::HashMap<VertexId, Vec<usize>> =
            hashbrown::HashMap::with_capacity(face_indices.len());
        let mut incoming: hashbrown::HashMap<VertexId, Vec<usize>> =
            hashbrown::HashMap::with_capacity(face_indices.len());

        for &fi in face_indices {
            let face = mesh.faces.get(FaceId::from_usize(fi));
            let verts = &face.vertices;
            let pos = verts
                .iter()
                .position(|&vid| vid == v)
                .expect("invariant: face in vertex_faces[v] must contain vertex v");
            let next = verts[(pos + 1) % 3]; // v → next (outgoing half-edge)
            let prev = verts[(pos + 2) % 3]; // prev → v (incoming half-edge)
            outgoing.entry(next).or_default().push(fi);
            incoming.entry(prev).or_default().push(fi);
        }

        // Step 3: BFS to find connected components of the face fan.
        //
        // Two faces are manifold-adjacent around v if they share an edge
        // {v, w} where both outgoing[w] and incoming[w] have exactly one
        // entry (manifold edge).  At a non-manifold edge (multiplicity > 1
        // in either map), we refuse to cross — this is the bridge between
        // pinch cycles.
        let mut visited: hashbrown::HashSet<usize> =
            hashbrown::HashSet::with_capacity(face_indices.len());
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(face_indices.len());

        for &start_fi in face_indices {
            if visited.contains(&start_fi) {
                continue;
            }
            let mut component: Vec<usize> = Vec::with_capacity(face_indices.len());
            queue.clear();
            queue.push_back(start_fi);
            visited.insert(start_fi);

            while let Some(fi) = queue.pop_front() {
                component.push(fi);
                let face = mesh.faces.get(FaceId::from_usize(fi));
                let verts = &face.vertices;
                let pos = verts
                    .iter()
                    .position(|&vid| vid == v)
                    .expect("invariant: face in vertex_faces[v] must contain vertex v");
                let next_v = verts[(pos + 1) % 3];
                let prev_v = verts[(pos + 2) % 3];

                // Manifold neighbour via outgoing edge (v → next_v):
                //   partner is the unique face with incoming half-edge
                //   next_v → v, but only if the edge {v, next_v} is manifold.
                let out_count = outgoing.get(&next_v).map_or(0, |v| v.len());
                if let Some(adj_faces) = incoming.get(&next_v)
                    && out_count == 1
                    && adj_faces.len() == 1
                {
                    let adj_fi = adj_faces[0];
                    if !visited.contains(&adj_fi) {
                        visited.insert(adj_fi);
                        queue.push_back(adj_fi);
                    }
                }
                // Manifold neighbour via incoming edge (prev_v → v):
                //   partner is the unique face with outgoing half-edge
                //   v → prev_v, but only if the edge {v, prev_v} is manifold.
                if let Some(adj_faces) = outgoing.get(&prev_v) {
                    let in_count = incoming.get(&prev_v).map_or(0, |v| v.len());
                    if adj_faces.len() == 1 && in_count == 1 {
                        let adj_fi = adj_faces[0];
                        if !visited.contains(&adj_fi) {
                            visited.insert(adj_fi);
                            queue.push_back(adj_fi);
                        }
                    }
                }
            }
            components.push(component);
        }

        if components.len() <= 1 {
            continue;
        }

        // Step 4: split — create a new vertex for each additional component.
        let pos = *mesh.vertices.position(v);
        let normal = *mesh.vertices.normal(v);
        for component in components.iter().skip(1) {
            let new_v = mesh.add_vertex(pos, normal);
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
            "CSG postprocess: split {} non-manifold pinch vertex instance(s)",
            total_splits
        );
    }
}

/// Second-pass pinch-vertex detector via **link-graph component counting**.
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
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

        for start_local in 0..n {
            if visited[start_local] {
                continue;
            }
            let mut component: Vec<usize> = Vec::with_capacity(n);
            queue.clear();
            queue.push_back(start_local);
            visited[start_local] = true;

            while let Some(local_idx) = queue.pop_front() {
                let (fi, a, b) = face_link_edges[local_idx];
                component.push(fi);

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
            components.push(component);
        }

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

// ── Boundary vertex merging ──────────────────────────────────────────────────
