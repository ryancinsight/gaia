//! Repair passes: boundary-vertex and coincident-vertex merging.

use super::collapse::collapse_degenerate_faces;
use super::edges::split_non_manifold_edges;
use super::uf_find;
use crate::application::welding::GridCell;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Scalar;
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
    #[inline]
    fn quick_euler_referenced(mesh: &IndexedMesh) -> i64 {
        let edge_store =
            crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(&mesh.faces);
        crate::application::watertight::check::euler_chi_from_stores(&mesh.faces, &edge_store)
    }

    // Adaptive tolerance: `merge_mult` fraction of the mean edge length,
    // clamped to [0.01, 0.2] mm.  The escalating repair pipeline calls this
    // with progressively wider multipliers (0.05 → 0.40).
    let mean_edge_len = {
        mesh.rebuild_edges();
        let Some(edges) = mesh.edges_ref() else {
            return;
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
        sum / f64::from_usize(count)
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
        let Some(edges_ref) = mesh.edges_ref() else {
            break;
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
            let mut grid: hashbrown::HashMap<GridCell, Vec<usize>> =
                hashbrown::HashMap::with_capacity(bv.len());
            let bv_pos: Vec<leto::geometry::Point3<f64>> =
                bv.iter().map(|&v| *mesh.vertices.position(v)).collect();
            for (i, p) in bv_pos.iter().enumerate().take(bv.len()) {
                grid.entry(GridCell::from_point(p, inv_tol))
                    .or_default()
                    .push(i);
            }
            for (i, pi) in bv_pos.iter().enumerate().take(bv.len()) {
                let cell = GridCell::from_point(pi, inv_tol);
                for nb_cell in cell.neighborhood_27() {
                    let Some(cell_verts) = grid.get(&nb_cell) else {
                        continue;
                    };
                    for &j in cell_verts {
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

        // Phase 3: if no boundary-boundary pair found, try boundary-to-interior
        // using a spatial hash grid with cell size = per_vertex_tol.
        if best.is_none() {
            let per_vertex_tol = tol * 0.5;
            let inv_pvt = 1.0 / per_vertex_tol;
            // Build grid over interior vertices only.
            let all_vids: Vec<VertexId> = mesh.vertices.iter().map(|(id, _)| id).collect();
            let mut igrid: hashbrown::HashMap<GridCell, Vec<VertexId>> =
                hashbrown::HashMap::with_capacity(
                    all_vids.len().saturating_sub(boundary_verts.len()),
                );
            for &ivid in &all_vids {
                if boundary_verts.contains(&ivid) {
                    continue;
                }
                let ip = mesh.vertices.position(ivid);
                igrid
                    .entry(GridCell::from_point(ip, inv_pvt))
                    .or_default()
                    .push(ivid);
            }
            for &bvid in &bv {
                let bp = mesh.vertices.position(bvid);
                let cell = GridCell::from_point(bp, inv_pvt);
                for nb_cell in cell.neighborhood_27() {
                    let Some(cell_verts) = igrid.get(&nb_cell) else {
                        continue;
                    };
                    for &ivid in cell_verts {
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

        let Some((keep, remove, _dist)) = best else {
            break;
        };

        // --- Euler-preserving guard ---
        // Save face-store snapshot before merge so we can revert if χ
        // decreases.  A decrease means the merge created a topological
        // handle (common at dense N-way junctions where two boundary
        // loops should not be connected).
        let faces_snapshot: Vec<crate::infrastructure::storage::face_store::FaceData> =
            mesh.faces.iter().copied().collect();

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

/// Find union-find classes for coincident vertices using a spatial hash grid.
fn find_coincident_vertex_classes(
    positions: &[leto::geometry::Point3<f64>],
    eps_sq: f64,
    inv_eps: f64,
) -> Vec<u32> {
    let n = positions.len();
    let mut parent: Vec<u32> = (0..u32::try_from(n).expect("vertex count fits in u32")).collect();
    let mut grid: hashbrown::HashMap<GridCell, Vec<usize>> = hashbrown::HashMap::with_capacity(n);

    for (i, position) in positions.iter().enumerate() {
        grid.entry(GridCell::from_point(position, inv_eps))
            .or_default()
            .push(i);
    }

    for i in 0..n {
        let pi = &positions[i];
        let cell = GridCell::from_point(pi, inv_eps);
        for nb_cell in cell.neighborhood_27() {
            let Some(cell_vertices) = grid.get(&nb_cell) else {
                continue;
            };
            for &j in cell_vertices {
                if j <= i {
                    continue;
                }
                let pj = &positions[j];
                if (pi - pj).norm_squared() < eps_sq {
                    let ci = uf_find(
                        &mut parent,
                        u32::try_from(i).expect("vertex index fits in u32"),
                    );
                    let cj = uf_find(
                        &mut parent,
                        u32::try_from(j).expect("vertex index fits in u32"),
                    );
                    if ci != cj {
                        let (lo, hi) = if ci < cj { (ci, cj) } else { (cj, ci) };
                        parent[usize::try_from(hi).expect("union-find index fits in usize")] = lo;
                    }
                }
            }
        }
    }

    (0..n)
        .map(|i| {
            uf_find(
                &mut parent,
                u32::try_from(i).expect("vertex index fits in u32"),
            )
        })
        .collect()
}

/// Remap face references through the dedup classes and compact the vertex pool.
fn assemble_merged_mesh(mesh: &mut IndexedMesh, dedup: &[u32]) {
    let face_list: Vec<FaceData> = mesh.faces.iter().copied().collect();
    let mut remapped_faces: Vec<FaceData> = Vec::with_capacity(face_list.len());
    for mut face in face_list {
        for v in &mut face.vertices {
            *v = VertexId(dedup[v.as_usize()]);
        }
        if face.vertices[0] != face.vertices[1]
            && face.vertices[1] != face.vertices[2]
            && face.vertices[2] != face.vertices[0]
        {
            remapped_faces.push(face);
        }
    }

    let mut referenced = hashbrown::HashSet::with_capacity(mesh.vertices.len());
    for face in &remapped_faces {
        for &vertex in &face.vertices {
            referenced.insert(vertex.0);
        }
    }

    let mut referenced_ids: Vec<u32> = referenced.into_iter().collect();
    referenced_ids.sort_unstable();

    let mut old_to_new = vec![u32::MAX; dedup.len()];
    let mut new_pool = mesh.vertices.empty_clone();
    for &old_id in &referenced_ids {
        let vertex_id = VertexId(old_id);
        let position = *mesh.vertices.position(vertex_id);
        let normal = *mesh.vertices.normal(vertex_id);
        let new_id = new_pool.insert_unique(position, normal);
        old_to_new[usize::try_from(old_id).expect("vertex id fits in usize")] = new_id.0;
    }

    mesh.faces = crate::infrastructure::storage::face_store::FaceStore::new();
    for mut face in remapped_faces {
        for v in &mut face.vertices {
            *v = VertexId(old_to_new[v.as_usize()]);
        }
        mesh.faces.push(face);
    }
    mesh.vertices = new_pool;
    mesh.rebuild_edges();
}

/// Merge coincident vertices and compact the vertex pool.
///
/// 1. **Dedup**: merge vertices with ‖`p_i` − `p_j`‖ < ε (union-find).
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
        sum / f64::from_usize(cnt)
    };
    let eps = (mean_edge * 1e-4).max(1e-15);
    let eps_sq = eps * eps;
    let inv_eps = 1.0 / eps;
    let positions: Vec<leto::geometry::Point3<f64>> = (0..n)
        .map(|i| *mesh.vertices.position(VertexId::from_usize(i)))
        .collect();
    let dedup = find_coincident_vertex_classes(&positions, eps_sq, inv_eps);
    assemble_merged_mesh(mesh, &dedup);
}

// ── Non-manifold edge splitting ──────────────────────────────────────────────
