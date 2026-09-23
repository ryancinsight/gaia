//! Repair passes: boundary-vertex and coincident-vertex merging.

use super::collapse::collapse_degenerate_faces;
use super::edges::split_non_manifold_edges;
use super::uf_find;
use crate::domain::core::index::VertexId;
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::FaceData;

/// Merge nearby boundary vertices to close sliver gaps at intersection curves.
///
/// Identifies boundary vertices (those on boundary edges) and merges pairs
/// within a small adaptive tolerance.  This closes small gaps left by CSG
/// arrangement precision limits at complex intersection curves.
///
/// # Theorem — Boundary Vertex Merge Convergence
///
/// Each merge reduces the boundary edge count by exactly 2 (the two half-edges
/// incident to the merged vertex pair become interior).  The process terminates
/// when no further merges are possible (fixed-point).  ∎
pub(super) fn merge_nearby_boundary_vertices_with_mult(mesh: &mut IndexedMesh, merge_mult: f64) {
    // Adaptive tolerance: `merge_mult` fraction of the mean edge length,
    // clamped to [0.01, 0.2] mm.  The escalating repair pipeline calls this
    // with progressively wider multipliers (0.05 → 0.40).
    let mean_edge_len = {
        mesh.rebuild_edges();
        let edges = match mesh.edges_ref() {
            Some(e) => e,
            None => return,
        };
        let (sum, count) = edges
            .iter()
            .map(|e| {
                let pa = mesh.vertices.position(e.vertices.0);
                let pb = mesh.vertices.position(e.vertices.1);
                (pa - pb).norm()
            })
            .fold((0.0_f64, 0usize), |(s, c), d| (s + d, c + 1));
        if count == 0 {
            return;
        }
        sum / count as f64
    };
    // Scale-relative tolerance: `merge_mult` fraction of mean edge length,
    // clamped to [1% .. 20%] of mean edge length.  Using a relative clamp
    // instead of absolute [0.01, 0.2] makes the algorithm scale-invariant:
    // micro-scale geometry (1e-5) gets a proportionally tight tolerance
    // instead of an absolute 0.01 that dwarfs the mesh.
    let tol = mean_edge_len * merge_mult.clamp(0.01, 0.20);
    let max_iter = 30;

    // Pairs that caused a χ decrease (topology-damaging merges); skip on retry.
    let mut skip_pairs: hashbrown::HashSet<(VertexId, VertexId)> =
        hashbrown::HashSet::with_capacity(max_iter);

    for _iter in 0..max_iter {
        mesh.rebuild_edges();
        let edges_ref = match mesh.edges_ref() {
            Some(e) => e,
            None => break,
        };

        // Phase 1: collect boundary vertex IDs.
        let mut boundary_verts: hashbrown::HashSet<VertexId> =
            hashbrown::HashSet::with_capacity(edges_ref.len().saturating_mul(2));
        for edge in edges_ref.iter() {
            if edge.is_boundary() {
                boundary_verts.insert(edge.vertices.0);
                boundary_verts.insert(edge.vertices.1);
            }
        }

        if boundary_verts.is_empty() {
            break;
        }

        // Candidate selection uses strict distance comparisons. Canonical
        // vertex order makes equal-distance choices independent of hash order.
        let mut bv: Vec<VertexId> = boundary_verts.iter().copied().collect();
        bv.sort_unstable();

        // Phase 2: find closest boundary-boundary pair within tolerance,
        // skipping pairs that previously caused a χ decrease.
        //
        // Uses a spatial hash grid with cell size = tol so that only
        // vertices in the 27-cell neighbourhood are compared, reducing
        // worst-case O(B²) to O(B) expected.
        let inv_tol = 1.0 / tol;
        let mut best: Option<(VertexId, VertexId, f64)> = None;
        {
            let mut grid: hashbrown::HashMap<(i64, i64, i64), Vec<usize>> =
                hashbrown::HashMap::with_capacity(bv.len());
            let bv_pos: Vec<leto::geometry::Point3<f64>> =
                bv.iter().map(|&v| *mesh.vertices.position(v)).collect();
            for i in 0..bv.len() {
                let p = &bv_pos[i];
                let cx = (p.x * inv_tol).floor() as i64;
                let cy = (p.y * inv_tol).floor() as i64;
                let cz = (p.z * inv_tol).floor() as i64;
                grid.entry((cx, cy, cz)).or_default().push(i);
            }
            for i in 0..bv.len() {
                let pi = &bv_pos[i];
                let cx = (pi.x * inv_tol).floor() as i64;
                let cy = (pi.y * inv_tol).floor() as i64;
                let cz = (pi.z * inv_tol).floor() as i64;
                for dx in -1..=1_i64 {
                    for dy in -1..=1_i64 {
                        for dz in -1..=1_i64 {
                            if let Some(cell) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &j in cell {
                                    if j <= i {
                                        continue;
                                    }
                                    let pair_key = if bv[i] < bv[j] {
                                        (bv[i], bv[j])
                                    } else {
                                        (bv[j], bv[i])
                                    };
                                    if skip_pairs.contains(&pair_key) {
                                        continue;
                                    }
                                    let pj = &bv_pos[j];
                                    let d = (pi - pj).norm();
                                    let is_better = match best {
                                        Some((_, _, best_dist)) => d < best_dist,
                                        None => true,
                                    };
                                    if d < tol && is_better {
                                        best = Some((bv[i], bv[j], d));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Phase 3: if no boundary-boundary pair found, try boundary-to-interior
        // using a spatial hash grid with cell size = per_vertex_tol.
        if best.is_none() {
            let per_vertex_tol = tol * 0.5;
            let inv_pvt = 1.0 / per_vertex_tol;
            // Build grid over interior vertices only.
            let all_vids: Vec<VertexId> = mesh.vertices.iter().map(|(id, _)| id).collect();
            let mut igrid: hashbrown::HashMap<(i64, i64, i64), Vec<VertexId>> =
                hashbrown::HashMap::with_capacity(
                    all_vids.len().saturating_sub(boundary_verts.len()),
                );
            for &ivid in &all_vids {
                if boundary_verts.contains(&ivid) {
                    continue;
                }
                let ip = mesh.vertices.position(ivid);
                let cx = (ip.x * inv_pvt).floor() as i64;
                let cy = (ip.y * inv_pvt).floor() as i64;
                let cz = (ip.z * inv_pvt).floor() as i64;
                igrid.entry((cx, cy, cz)).or_default().push(ivid);
            }
            for &bvid in &bv {
                let bp = mesh.vertices.position(bvid);
                let cx = (bp.x * inv_pvt).floor() as i64;
                let cy = (bp.y * inv_pvt).floor() as i64;
                let cz = (bp.z * inv_pvt).floor() as i64;
                for dx in -1..=1_i64 {
                    for dy in -1..=1_i64 {
                        for dz in -1..=1_i64 {
                            if let Some(cell) = igrid.get(&(cx + dx, cy + dy, cz + dz)) {
                                for &ivid in cell {
                                    let pair_key = if bvid < ivid {
                                        (bvid, ivid)
                                    } else {
                                        (ivid, bvid)
                                    };
                                    if skip_pairs.contains(&pair_key) {
                                        continue;
                                    }
                                    let ip = mesh.vertices.position(ivid);
                                    let d = (bp - ip).norm();
                                    let is_better = match best {
                                        Some((_, _, best_dist)) => d < best_dist,
                                        None => true,
                                    };
                                    if d < per_vertex_tol && is_better {
                                        best = Some((ivid, bvid, d));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let (keep, remove, _dist) = match best {
            Some(b) => b,
            None => break,
        };

        // --- Euler-preserving guard ---
        // Save face-store snapshot before merge so we can revert if χ
        // decreases.  A decrease means the merge created a topological
        // handle (common at dense N-way junctions where two boundary
        // loops should not be connected).
        let faces_snapshot: Vec<crate::infrastructure::storage::face_store::FaceData> =
            mesh.faces.iter().copied().collect();

        // Compute χ using referenced vertices only — delegate to the canonical
        // SSOT implementation in `watertight::check`.
        #[inline]
        fn quick_euler_referenced(mesh: &IndexedMesh) -> i64 {
            let edge_store =
                crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(&mesh.faces);
            crate::application::watertight::check::euler_chi_from_stores(&mesh.faces, &edge_store)
        }

        let chi_before = quick_euler_referenced(mesh);

        // Merge: replace all references to `remove` with `keep`.
        let mut changed = false;
        for face in mesh.faces.iter_mut() {
            for v in &mut face.vertices {
                if *v == remove {
                    *v = keep;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }

        collapse_degenerate_faces(mesh);
        mesh.rebuild_edges();

        let chi_after = quick_euler_referenced(mesh);

        // If χ decreased, this merge created a topological handle.
        // Revert and skip this pair.
        if chi_after < chi_before {
            mesh.faces.clear();
            for face_data in faces_snapshot {
                mesh.faces.push(face_data);
            }
            mesh.rebuild_edges();
            let pair_key = if keep < remove {
                (keep, remove)
            } else {
                (remove, keep)
            };
            skip_pairs.insert(pair_key);
            continue; // Try next pair instead of breaking.
        }

        split_non_manifold_edges(mesh);
        collapse_degenerate_faces(mesh);
        mesh.rebuild_edges();

        if !mesh.is_watertight() {
            let es =
                crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(&mesh.faces);
            let _ = crate::application::watertight::seal::seal_boundary_loops(
                &mut mesh.vertices,
                &mut mesh.faces,
                &es,
                crate::domain::core::index::RegionId::INVALID,
            );
            collapse_degenerate_faces(mesh);
            mesh.rebuild_edges();
        }

        if mesh.is_watertight() {
            break;
        }
    }
}

/// Merge coincident vertices and compact the vertex pool.
///
/// 1. **Dedup**: merge vertices with ‖p_i − p_j‖ < ε (union-find).
/// 2. **Compact**: remove unreferenced vertices, re-index face references.
///
/// # Theorem — Vertex Pool Compaction Preserves Euler–Poincaré
///
/// `check_watertight` computes V as `vertex_pool.len()`.  CSG operations
/// leave dead vertices (from input meshes and merged duplicates) which
/// inflate V.  Compaction removes these, yielding V = |referenced vertices|
/// and restoring V − E + F = 2(1−g).  ∎
pub(super) fn merge_coincident_vertices(mesh: &mut IndexedMesh) {
    let n = mesh.vertices.len();
    if n == 0 {
        return;
    }

    // Phase 1: merge coincident vertices (dedup) via spatial hash grid.
    //
    // # Algorithm — Grid-Accelerated Vertex Deduplication
    //
    // Partition R³ into axis-aligned cells of side ε.  Each vertex is
    // assigned to cell (⌊x/ε⌋, ⌊y/ε⌋, ⌊z/ε⌋).  Two vertices can be
    // within distance ε only if their cells differ by at most 1 on every
    // axis (the 3×3×3 = 27-cell neighbourhood).
    //
    // # Theorem — Grid Neighbourhood Soundness
    //
    // If ‖p−q‖ < ε then |⌊p_k/ε⌋ − ⌊q_k/ε⌋| ≤ 1 for k∈{x,y,z}.
    //
    // *Proof.* |p_k − q_k| ≤ ‖p−q‖ < ε.  The floor of two reals that
    // differ by less than ε can differ by at most 1.  ∎
    //
    // Complexity drops from O(V²) to O(V) expected for uniformly
    // distributed vertices (each cell has O(1) occupants on average).
    //
    // Scale-relative tolerance: ε = 1e-4 × mean_edge_length.  This is
    // 3 orders of magnitude below the edge scale, safely catching
    // near-coincident vertices from independent CSG intersection
    // computations while never fusing distinct geometry.  Using a
    // relative epsilon (instead of absolute 1e-6) makes the function
    // scale-invariant for micro- and macro-scale meshes.
    let mean_edge = {
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for face in mesh.faces.iter() {
            for k in 0..3 {
                let a = mesh.vertices.position(face.vertices[k]);
                let b = mesh.vertices.position(face.vertices[(k + 1) % 3]);
                sum += (a - b).norm();
                cnt += 1;
            }
        }
        if cnt == 0 {
            return;
        }
        sum / cnt as f64
    };
    let eps = (mean_edge * 1e-4).max(1e-15);
    let eps_sq = eps * eps;
    let inv_eps = 1.0 / eps;
    let mut parent: Vec<u32> = (0..n as u32).collect();

    // Build spatial hash: cell → list of vertex indices.
    let mut grid: hashbrown::HashMap<(i64, i64, i64), Vec<usize>> =
        hashbrown::HashMap::with_capacity(n);
    let positions: Vec<leto::geometry::Point3<f64>> = (0..n)
        .map(|i| *mesh.vertices.position(VertexId(i as u32)))
        .collect();
    for i in 0..n {
        let p = &positions[i];
        let cx = (p.x * inv_eps).floor() as i64;
        let cy = (p.y * inv_eps).floor() as i64;
        let cz = (p.z * inv_eps).floor() as i64;
        grid.entry((cx, cy, cz)).or_default().push(i);
    }

    // For each vertex, check the 27-cell neighbourhood for coincident vertices.
    for i in 0..n {
        let pi = &positions[i];
        let cx = (pi.x * inv_eps).floor() as i64;
        let cy = (pi.y * inv_eps).floor() as i64;
        let cz = (pi.z * inv_eps).floor() as i64;
        for dx in -1..=1_i64 {
            for dy in -1..=1_i64 {
                for dz in -1..=1_i64 {
                    if let Some(cell) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in cell {
                            if j <= i {
                                continue;
                            }
                            let pj = &positions[j];
                            if (pi - pj).norm_squared() < eps_sq {
                                let ci = uf_find(&mut parent, i as u32);
                                let cj = uf_find(&mut parent, j as u32);
                                if ci != cj {
                                    let (lo, hi) = if ci < cj { (ci, cj) } else { (cj, ci) };
                                    parent[hi as usize] = lo;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Flatten union-find: old_id → canonical_id.
    let dedup: Vec<u32> = (0..n).map(|i| uf_find(&mut parent, i as u32)).collect();

    // Phase 2: remap face references through dedup mapping.
    let face_list: Vec<FaceData> = mesh.faces.iter().copied().collect();
    let mut remapped_faces: Vec<FaceData> = Vec::with_capacity(face_list.len());
    for mut face in face_list {
        for v in &mut face.vertices {
            *v = VertexId(dedup[v.0 as usize]);
        }
        if face.vertices[0] != face.vertices[1]
            && face.vertices[1] != face.vertices[2]
            && face.vertices[2] != face.vertices[0]
        {
            remapped_faces.push(face);
        }
    }

    // Phase 3: compact — collect referenced vertex IDs and build new pool.
    let mut referenced = hashbrown::HashSet::with_capacity(mesh.vertices.len());
    for face in &remapped_faces {
        for &v in &face.vertices {
            referenced.insert(v.0);
        }
    }

    // Sort referenced IDs for deterministic new-index assignment.
    let mut ref_ids: Vec<u32> = referenced.into_iter().collect();
    ref_ids.sort_unstable();

    // Build old → new index mapping.
    let mut old_to_new = vec![u32::MAX; n];
    let mut new_pool = mesh.vertices.empty_clone();
    for &old_id in &ref_ids {
        let vid = VertexId(old_id);
        let pos = *mesh.vertices.position(vid);
        let normal = *mesh.vertices.normal(vid);
        let new_id = new_pool.insert_unique(pos, normal);
        old_to_new[old_id as usize] = new_id.0;
    }

    // Re-index face references.
    mesh.faces = crate::infrastructure::storage::face_store::FaceStore::new();
    for mut face in remapped_faces {
        for v in &mut face.vertices {
            *v = VertexId(old_to_new[v.0 as usize]);
        }
        mesh.faces.push(face);
    }
    mesh.vertices = new_pool;
    mesh.rebuild_edges();
}

// ── Non-manifold edge splitting ──────────────────────────────────────────────
