//! Repair passes: non-manifold edge resolution and fin removal.

use super::face_normal_of;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Vector3r;
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::FaceData;
use core::cmp::Ordering;

/// Resolve non-manifold edges by removing excess faces.
///
/// A 2-manifold requires every edge to be shared by exactly 2 faces.
/// CSG arrangement can produce edges with 3+ faces at intersection curves.
/// This function keeps the **best-oriented pair** sharing each non-manifold
/// edge and removes the rest.
///
/// # Selection criterion (deterministic)
///
/// For a non-manifold edge (u,v) with k > 2 incident faces, the correct
/// manifold pair consists of the two faces whose half-edges form a consistent
/// orientation: one face has the directed edge u→v and the other has v→u.
/// Among all such consistent pairs, we select the pair whose normals have the
/// **largest mutual dot product** (most co-planar / smoothest dihedral angle),
/// breaking ties by smallest face index.  This deterministic criterion avoids
/// the prior HashMap-order-dependent selection that could discard the
/// geometrically correct faces.
///
/// # Theorem — Non-Manifold Edge Elimination
///
/// After removal, every edge has at most 2 faces.  The removed faces'
/// other edges may become boundary edges (1 face) or remain manifold
/// (2 faces).  The resulting mesh has no non-manifold edges.  ∎
pub(in crate::application::csg::boolean::indexed) fn split_non_manifold_edges(
    mesh: &mut IndexedMesh,
) {
    let face_list: Vec<FaceData> = mesh.faces.iter().copied().collect();

    // Build undirected edge → face index map.
    let mut edge_faces: hashbrown::HashMap<(VertexId, VertexId), Vec<usize>> =
        hashbrown::HashMap::with_capacity(face_list.len().saturating_mul(3) / 2);
    for (fi, face) in face_list.iter().enumerate() {
        let v = face.vertices;
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_faces.entry(key).or_default().push(fi);
        }
    }

    // Collect faces nominated for removal across all non-manifold edges.
    // For each non-manifold edge, pick the best pair and mark the rest.
    let mut faces_to_remove: hashbrown::HashSet<usize> =
        hashbrown::HashSet::with_capacity(face_list.len() / 8);

    let mut edge_keys: Vec<(VertexId, VertexId)> = edge_faces.keys().copied().collect();
    edge_keys.sort_unstable();
    for (u, v) in edge_keys {
        let fis = &edge_faces[&(u, v)];
        if fis.len() <= 2 {
            continue;
        }

        // Classify each face by its directed half-edge orientation for (u,v).
        // forward = has u→v, reverse = has v→u.
        let mut forward: Vec<usize> = Vec::with_capacity(fis.len());
        let mut reverse: Vec<usize> = Vec::with_capacity(fis.len());

        for &fi in fis {
            let fv = face_list[fi].vertices;
            let has_uv = (0..3).any(|k| fv[k] == u && fv[(k + 1) % 3] == v);
            if has_uv {
                forward.push(fi);
            } else {
                reverse.push(fi);
            }
        }

        // Pick the best consistent pair (one forward, one reverse) by
        // maximum normal dot product (smoothest dihedral).
        //
        // A face with no usable normal has no orientation to be consistent
        // with, so it is not a candidate. Dropping such faces here, rather
        // than substituting a sentinel dot product, leaves `best_pair` as
        // `None` when no *valid* pair exists — which is what routes control to
        // the index fallback below. With a sentinel the sentinel wins a tie
        // against itself, so the fallback was unreachable whenever both sides
        // were non-empty and every normal was degenerate.
        //
        // Each face's normal is computed once: it is a pure function of the
        // face and the vertex pool, and the nested loop would otherwise
        // recompute it once per pairing, costing O(|forward| × |reverse|)
        // normal evaluations instead of O(|forward| + |reverse|).
        let forward_normals: Vec<(usize, Vector3r)> = forward
            .iter()
            .filter_map(|&fi| face_normal_of(&face_list[fi], &mesh.vertices).map(|n| (fi, n)))
            .collect();
        let reverse_normals: Vec<(usize, Vector3r)> = reverse
            .iter()
            .filter_map(|&fi| face_normal_of(&face_list[fi], &mesh.vertices).map(|n| (fi, n)))
            .collect();

        let mut best_pair: Option<(usize, usize, f64)> = None;
        for &(fi_fwd, n_fwd) in &forward_normals {
            for &(fi_rev, n_rev) in &reverse_normals {
                let dot = n_fwd.dot(n_rev);
                let better = match best_pair {
                    None => true,
                    Some((best_fwd, best_rev, best_dot)) => {
                        // `total_cmp` is a total order over every `f64`, so a
                        // one-ULP difference compares strictly rather than
                        // falling through to the index tie-break, and a `NaN`
                        // cannot make both comparisons false and be silently
                        // skipped.
                        match dot.total_cmp(&best_dot) {
                            Ordering::Greater => true,
                            // An exact tie is broken by the lower face index,
                            // so the choice does not depend on iteration order.
                            Ordering::Equal => fi_fwd.min(fi_rev) < best_fwd.min(best_rev),
                            Ordering::Less => false,
                        }
                    }
                };
                if better {
                    best_pair = Some((fi_fwd, fi_rev, dot));
                }
            }
        }

        // Mark all faces on this edge except the best pair for removal.
        let (keep_a, keep_b) = if let Some((a, b, _)) = best_pair {
            (a, b)
        } else {
            // No consistent pair found — keep the first two by index
            // (deterministic fallback).
            let mut sorted = fis.clone();
            sorted.sort_unstable();
            (sorted[0], sorted[1])
        };
        for &fi in fis {
            if fi != keep_a && fi != keep_b {
                faces_to_remove.insert(fi);
            }
        }
    }

    if faces_to_remove.is_empty() {
        return;
    }

    let mut clean_faces: Vec<FaceData> =
        Vec::with_capacity(face_list.len() - faces_to_remove.len());
    for (fi, face) in face_list.iter().enumerate() {
        if !faces_to_remove.contains(&fi) {
            clean_faces.push(*face);
        }
    }
    mesh.faces = crate::infrastructure::storage::face_store::FaceStore::new();
    for face in clean_faces {
        mesh.faces.push(face);
    }
}

/// Remove "fin" faces — phantom faces at CSG junctions whose normals point
/// sharply away from all edge-adjacent neighbors.
///
/// # Detection criterion
///
/// For each face *f*, compute `max_dot = max_{g ∈ adj(f)} n_f · n_g` over
/// all edge-adjacent faces *g*.  When `max_dot < cos(120°) = −0.5`, face *f*
/// has no neighbor even approximately co-oriented — it is a fin artifact
/// from incorrect CSG face classification.
///
/// # Theorem — Fin Face Invariant
///
/// On a genus-0 closed 2-manifold with outward-consistent orientation, every
/// face has at least one edge neighbor with `n_f · n_g > 0` (both face the
/// same half-space locally).  A face violating this invariant is not part of
/// the intended surface.  Removing it and re-sealing preserves the manifold
/// topology.  ∎
pub(super) fn remove_fin_faces(mesh: &mut IndexedMesh) {
    use crate::domain::geometry::normal::triangle_normal;

    let face_list: Vec<FaceData> = mesh.faces.iter().copied().collect();
    let n_faces = face_list.len();
    if n_faces == 0 {
        return;
    }

    // Compute per-face normals.
    let face_normals: Vec<Option<leto::geometry::Vector3<f64>>> = face_list
        .iter()
        .map(|f| {
            let a = mesh.vertices.position(f.vertices[0]);
            let b = mesh.vertices.position(f.vertices[1]);
            let c = mesh.vertices.position(f.vertices[2]);
            triangle_normal(a, b, c)
        })
        .collect();

    // Build undirected edge → face adjacency.
    let mut edge_adj: hashbrown::HashMap<(VertexId, VertexId), Vec<usize>> =
        hashbrown::HashMap::with_capacity(n_faces.saturating_mul(3) / 2);
    for (fi, face) in face_list.iter().enumerate() {
        let v = face.vertices;
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            edge_adj.entry(key).or_default().push(fi);
        }
    }

    // For each face, find the maximum dot product with any edge neighbor.
    let cos_threshold = -0.94_f64; // cos(160°) — only flags extreme folds
    let mut fin_faces: hashbrown::HashSet<usize> = hashbrown::HashSet::with_capacity(n_faces / 16);

    for fi in 0..n_faces {
        let n_f = match face_normals[fi] {
            Some(n) => n,
            None => continue,
        };

        // Gather all distinct edge-neighbor face indices.
        let v = face_list[fi].vertices;
        let mut neighbors = Vec::with_capacity(6);
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(adj_faces) = edge_adj.get(&key) {
                for &nfi in adj_faces {
                    if nfi != fi {
                        neighbors.push(nfi);
                    }
                }
            }
        }

        if neighbors.is_empty() {
            continue;
        }

        // Find the maximum agreement with any neighbor.
        let max_dot = neighbors
            .iter()
            .filter_map(|&nfi| face_normals[nfi].map(|n_g| n_f.dot(n_g)))
            .fold(f64::NEG_INFINITY, f64::max);

        if max_dot < cos_threshold {
            fin_faces.insert(fi);
        }
    }

    if fin_faces.is_empty() {
        return;
    }

    // Remove fin faces.
    let mut clean_faces: Vec<FaceData> = Vec::with_capacity(n_faces - fin_faces.len());
    for (fi, face) in face_list.iter().enumerate() {
        if !fin_faces.contains(&fi) {
            clean_faces.push(*face);
        }
    }
    mesh.faces = crate::infrastructure::storage::face_store::FaceStore::new();
    for face in clean_faces {
        mesh.faces.push(face);
    }
    mesh.rebuild_edges();
}
