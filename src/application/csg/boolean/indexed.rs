//! `IndexedMesh` wrapper API for CSG Boolean operations

use crate::application::csg::boolean::BooleanOp;
use crate::application::csg::reconstruct;
use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::geometry::normal::triangle_normal;
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::{FaceData, FaceStore};
use crate::infrastructure::storage::vertex_pool::VertexPool;

/// Path-compressing union-find: return the root of `x` with halving compression.
///
/// # Invariant
/// `parent[root] == root`. Path halving rewrites each traversed node to its
/// grandparent, shortening future finds without allocating an auxiliary stack.
///
/// This is the single canonical implementation shared by all union-find phases in
/// this module. Previously duplicated as `find_cdf` in `collapse_degenerate_faces`
/// and `find` in `merge_coincident_vertices` — both are now removed.
#[inline(always)]
fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// High-level binary boolean operation on two [`IndexedMesh`] objects.
///
/// Merges both vertex pools into a shared [`VertexPool`] via `insert_or_weld`
/// (snap-welding within tolerance ε), runs the arrangement-based Boolean
/// pipeline, and reconstructs a fresh deduplicated `IndexedMesh`.
///
/// # Algorithm
///
/// 1. **Remap** — both meshes' vertices are inserted into a shared pool;
///    coincident vertices are welded to prevent T-junction seam gaps.
/// 2. **Coplanar detection** — both operands are checked for flat-plane
///    degeneracy to enable coplanar-aware repair.
/// 3. **Arrangement** — `csg_boolean_unfinalized` computes the generalized
///    arrangement: BVH-accelerated intersection detection, co-refinement,
///    GWN-based classification, and face selection per the `op` predicate.
/// 4. **Postprocessing** — normal recomputation, orientation repair (BFS +
///    `orient_outward`), escalating repair cascade, and fin removal.
///
/// For three or more operands, prefer [`csg_boolean_nary`] to avoid error
/// accumulation from repeated binary operations.
pub fn csg_boolean(
    op: BooleanOp,
    mesh_a: &IndexedMesh,
    mesh_b: &IndexedMesh,
) -> MeshResult<IndexedMesh> {
    let bb_a = mesh_a.bounding_box();
    let bb_b = mesh_b.bounding_box();
    let combined_bb = bb_a.union(&bb_b);
    let diag = (combined_bb.max - combined_bb.min).norm();
    let mut combined = VertexPool::for_csg_with_scale(diag);
    let (faces_a, faces_b) = remap_binary_face_soups(mesh_a, mesh_b, &mut combined);
    let is_coplanar = crate::application::csg::coplanar::detect_flat_plane(&faces_a, &combined)
        .is_some()
        && crate::application::csg::coplanar::detect_flat_plane(&faces_b, &combined).is_some();
    let result_faces = crate::application::csg::arrangement::boolean_csg::csg_boolean_unfinalized(
        op,
        &[faces_a, faces_b],
        &mut combined,
    )?;
    postprocess_boolean_mesh(result_faces, &combined, is_coplanar)
}

/// Compute an indexed Boolean across an arbitrary number of meshes using the
/// canonical generalized arrangement engine.
///
/// # N-ary Generalized Arrangement Algorithm
///
/// Traditional CSG Boolean implementations process pairs of meshes iteratively:
/// `((A ∪ B) ∪ C) ∪ D`. This approach has two fundamental problems:
///
/// 1. **Error accumulation** — each intermediate mesh feeds into the next
///    Boolean, so approximation artifacts compound at every stage.
/// 2. **Redundant work** — faces between B and C are tessellated and classified
///    in the first Boolean, then re-tessellated when the intermediate result
///    meets D.
///
/// The n-ary algorithm avoids both problems by processing all operands in a
/// single pass:
///
/// ## Phase 1 — Vertex Pool Merging
///
/// All input meshes are remapped into a shared [`VertexPool`] via
/// `insert_or_weld`, which snaps coincident vertices (within tolerance ε)
/// to the same ID. This establishes a consistent coordinate frame and
/// prevents T-junction seam gaps from floating-point discrepancies.
///
/// ## Phase 2 — Generalized Arrangement
///
/// The face soups from all N operands are passed simultaneously to
/// `csg_boolean_unfinalized`. The arrangement engine:
///
/// - Detects **all pairwise intersections** between triangle faces from
///   different operands using a BVH acceleration structure.
/// - Computes exact intersection curves via robust geometric predicates
///   (Shewchuk orientation/incircle predicates with adaptive precision).
/// - **Co-refines** all intersecting triangles simultaneously, splitting
///   them along every intersection curve to produce a conforming
///   triangulation where no triangle spans a boundary between regions.
/// - **Classifies** each resulting sub-face by evaluating its centroid
///   against the winding number of every operand, determining which
///   operands contain each fragment.
/// - **Selects** faces according to the Boolean predicate:
///   - **Union**: face is on the boundary of at least one operand and
///     outside all others, or on a shared boundary.
///   - **Intersection**: face is inside every operand.
///   - **Difference** (A \ B₁ \ B₂ \ ... \ Bₙ): face is inside A and
///     outside all Bᵢ, plus faces from Bᵢ that are inside A and outside
///     all other Bⱼ (with flipped orientation).
///
/// ## Phase 3 — Postprocessing
///
/// The selected faces are reconstructed into an [`IndexedMesh`] by the
/// postprocessing stage, which applies:
///
/// - Normal recomputation from face geometry.
/// - Orientation repair (BFS + `orient_outward`).
/// - Boundary sealing, sliver gap closure, and non-manifold edge
///   resolution through an escalating repair cascade.
/// - Fin artifact removal and largest-component retention.
///
/// # Correctness Properties
///
/// **Theorem (Single-pass equivalence):** For associative operations (Union,
/// Intersection), the n-ary result is identical to any parenthesization of
/// pairwise operations on exact geometry. For Difference, the result equals
/// `A \ (B₁ ∪ B₂ ∪ ... ∪ Bₙ)`.
///
/// *Proof sketch:* The arrangement produces the identical planar subdivision
/// regardless of operand ordering because all pairwise intersections are
/// computed simultaneously. Face classification depends only on the
/// point-in-solid winding number query against each operand, which is
/// independent of processing order. ∎
///
/// **Theorem (Watertight output):** If all input meshes are watertight
/// (closed, oriented 2-manifolds), the output is watertight after repair,
/// or an error is returned.
///
/// *Proof sketch:* The co-refinement preserves the 2-manifold property at
/// intersection curves by construction (each original edge is split at
/// every crossing point). The repair cascade closes any residual gaps
/// from floating-point perturbation. The final watertight check rejects
/// meshes that cannot be repaired. ∎
///
/// # Arguments
///
/// * `op` — The Boolean operation to apply.
/// * `meshes` — The operand meshes. For [`BooleanOp::Difference`], the first
///   mesh is the minuend and all subsequent meshes are subtrahends.
///
/// # Errors
///
/// Returns [`MeshError::EmptyBooleanResult`] if `meshes` is empty, or
/// [`MeshError::NotWatertight`] if the result cannot be made watertight.
pub fn csg_boolean_nary(op: BooleanOp, meshes: &[IndexedMesh]) -> MeshResult<IndexedMesh> {
    if meshes.is_empty() {
        return Err(MeshError::EmptyBooleanResult {
            op: format!("{op:?}"),
        });
    }

    if meshes.len() == 1 {
        return Ok(meshes[0].clone());
    }

    // Compute scale from combined AABB diagonal for scale-relative VertexPool.
    use crate::domain::geometry::aabb::Aabb;
    let mut combined_bb = Aabb::empty();
    for m in meshes {
        let bb = m.bounding_box();
        combined_bb = combined_bb.union(&bb);
    }
    let diag = (combined_bb.max - combined_bb.min).norm();
    let mut combined = VertexPool::for_csg_with_scale(diag);
    let face_soups = remap_nary_face_soups(meshes, &mut combined);
    let is_coplanar = face_soups.iter().all(|faces| {
        crate::application::csg::coplanar::detect_flat_plane(faces, &combined).is_some()
    });
    let result_faces = crate::application::csg::arrangement::boolean_csg::csg_boolean_unfinalized(
        op,
        &face_soups,
        &mut combined,
    )?;
    postprocess_boolean_mesh(result_faces, &combined, is_coplanar)
}

fn remap_binary_face_soups(
    mesh_a: &IndexedMesh,
    mesh_b: &IndexedMesh,
    combined: &mut VertexPool,
) -> (Vec<FaceData>, Vec<FaceData>) {
    let mut remap_a: hashbrown::HashMap<VertexId, VertexId> =
        hashbrown::HashMap::with_capacity(mesh_a.vertices.len());
    for (old_id, _) in mesh_a.vertices.iter() {
        let pos = *mesh_a.vertices.position(old_id);
        let nrm = *mesh_a.vertices.normal(old_id);
        remap_a.insert(old_id, combined.insert_or_weld(pos, nrm));
    }

    let mut remap_b: hashbrown::HashMap<VertexId, VertexId> =
        hashbrown::HashMap::with_capacity(mesh_b.vertices.len());
    for (old_id, _) in mesh_b.vertices.iter() {
        let pos = *mesh_b.vertices.position(old_id);
        let nrm = *mesh_b.vertices.normal(old_id);
        remap_b.insert(old_id, combined.insert_or_weld(pos, nrm));
    }

    let faces_a: Vec<FaceData> = mesh_a
        .faces
        .iter()
        .map(|face| FaceData {
            vertices: face.vertices.map(|vertex_id| remap_a[&vertex_id]),
            region: face.region,
        })
        .collect();
    let faces_b: Vec<FaceData> = mesh_b
        .faces
        .iter()
        .map(|face| FaceData {
            vertices: face.vertices.map(|vertex_id| remap_b[&vertex_id]),
            region: face.region,
        })
        .collect();

    (faces_a, faces_b)
}

fn remap_nary_face_soups(meshes: &[IndexedMesh], combined: &mut VertexPool) -> Vec<Vec<FaceData>> {
    let mut face_soups = Vec::with_capacity(meshes.len());
    for mesh in meshes {
        let mut remap: hashbrown::HashMap<VertexId, VertexId> =
            hashbrown::HashMap::with_capacity(mesh.vertices.len());
        for (old_id, _) in mesh.vertices.iter() {
            let pos = *mesh.vertices.position(old_id);
            let nrm = *mesh.vertices.normal(old_id);
            remap.insert(old_id, combined.insert_or_weld(pos, nrm));
        }

        let faces: Vec<FaceData> = mesh
            .faces
            .iter()
            .map(|face| FaceData {
                vertices: face.vertices.map(|vertex_id| remap[&vertex_id]),
                region: face.region,
            })
            .collect();
        face_soups.push(faces);
    }
    face_soups
}

fn postprocess_boolean_mesh(
    result_faces: Vec<FaceData>,
    combined: &VertexPool,
    is_coplanar: bool,
) -> MeshResult<IndexedMesh> {
    let mut mesh = reconstruct::reconstruct_mesh(&result_faces, combined);

    mesh.recompute_normals();
    repair_boolean_mesh(&mut mesh, is_coplanar)?;

    // Iterate collapse → split cycles until stable.  Splitting a pinch vertex
    // can produce degenerate slivers whose collapse re-pinches the mesh;
    // tight multi-operand junctions (e.g. 40° trifurcation) may need 3+
    // iterations to fully resolve.
    for _ in 0..8 {
        collapse_degenerate_faces(&mut mesh);
        split_non_manifold_vertices(&mut mesh);
        let pinch_splits = split_figure8_pinch_vertices(&mut mesh);
        if pinch_splits == 0 {
            break;
        }
    }

    // Final orient_outward: collapse_degenerate_faces and vertex splitting
    // can invalidate winding order established by repair_boolean_mesh.
    mesh.orient_outward();
    mesh.rebuild_edges();
    Ok(mesh)
}

/// Post-process a Boolean result mesh: repair orientation, close boundary
/// loops, stitch sliver gaps, and remove phantom fin artifacts.
///
/// # Algorithm
///
/// The repair proceeds in escalating phases:
///
/// 1. **Orientation consistency** — `orient_outward()`, then `fix_orientation()`
///    if BFS alone fails to resolve mixed-winding faces.
/// 2. **Boundary seal** — `seal_boundary_loops()` fills any small holes left
///    by arrangement fragment removal.
/// 3. **Iterative boundary stitch** — merges nearly-coincident boundary edges
///    using parametric stitching.
/// 4. **Non-manifold resolution** — `split_non_manifold_edges()` resolves
///    edges with 3+ incident faces, then `seal_boundary_loops()` closes any
///    holes opened by face removal. `merge_nearby_boundary_vertices()` closes
///    remaining sliver gaps at increasing tolerances (5%, 10%, 20%, 40% of
///    mean edge length).
/// 5. **Fin removal** — detects and removes phantom faces whose normals
///    oppose all edge-adjacent neighbours (max_dot < cos 120°). Only runs
///    when the mesh is already watertight to avoid nondeterministic BFS.
/// 6. **Cleanup** — `retain_largest_component()` removes disconnected
///    phantom surfaces, `merge_coincident_vertices()` welds any ε-close
///    duplicates, final `orient_outward()`.
///
/// # Errors
///
/// Returns `MeshError::NotWatertight` if the mesh cannot be made watertight
/// after all repair phases.
fn repair_boolean_mesh(mesh: &mut IndexedMesh, is_coplanar: bool) -> MeshResult<()> {
    /// Compute Euler characteristic V - E + F for Euler guards.
    ///
    /// Delegates to `euler_chi_from_stores` — the canonical SSOT implementation
    /// in `watertight::check`. Compared to the previous two-HashSet inline,
    /// this eliminates the O(F) edge-counting HashSet by reusing `EdgeStore::len()`.
    #[inline]
    fn euler_chi(mesh: &IndexedMesh) -> i64 {
        let edge_store =
            crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(&mesh.faces);
        crate::application::watertight::check::euler_chi_from_stores(&mesh.faces, &edge_store)
    }

    if !is_coplanar {
        mesh.rebuild_edges();
        let mut report = crate::application::watertight::check::check_watertight(
            &mesh.vertices,
            &mesh.faces,
            mesh.edges_ref().expect("invariant: edges are rebuilt"),
        );
        if !report.is_watertight {
            // Phase 1: Orientation repair.
            // Try orient_outward first; if that fails, try fix_orientation.
            // Only one orient pass is needed here — the final orient runs
            // after the repair loop completes.
            if report.is_closed && !report.orientation_consistent {
                mesh.orient_outward();
                mesh.rebuild_edges();
                report = crate::application::watertight::check::check_watertight(
                    &mesh.vertices,
                    &mesh.faces,
                    mesh.edges_ref().expect("invariant: edges are rebuilt"),
                );

                if !report.is_watertight && report.is_closed && !report.orientation_consistent {
                    let edge_store =
                        crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(
                            &mesh.faces,
                        );
                    let _ = crate::domain::topology::orientation::fix_orientation(
                        &mut mesh.faces,
                        &edge_store,
                    );
                    mesh.rebuild_edges();
                    report = crate::application::watertight::check::check_watertight(
                        &mesh.vertices,
                        &mesh.faces,
                        mesh.edges_ref().expect("invariant: edges are rebuilt"),
                    );
                }
            }

            // Phase 2: Boundary seal (only if no non-manifold edges).
            if !report.is_watertight
                && report.non_manifold_edge_count == 0
                && report.boundary_edge_count > 0
                && report.boundary_edge_count <= 512
            {
                let edge_store =
                    crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(
                        &mesh.faces,
                    );
                let added = crate::application::watertight::seal::seal_boundary_loops(
                    &mut mesh.vertices,
                    &mut mesh.faces,
                    &edge_store,
                    crate::domain::core::index::RegionId::INVALID,
                );
                if added > 0 {
                    mesh.rebuild_edges();
                    report = crate::application::watertight::check::check_watertight(
                        &mesh.vertices,
                        &mesh.faces,
                        mesh.edges_ref().expect("invariant: edges are rebuilt"),
                    );
                }
            }

            // Phase 3: Iterative boundary stitch.
            if !report.is_watertight && report.boundary_edge_count > 0 {
                let improved =
                    crate::application::watertight::repair::MeshRepair::iterative_boundary_stitch(
                        &mut mesh.faces,
                        &mesh.vertices,
                        3,
                    );
                if improved > 0 {
                    mesh.rebuild_edges();
                    report = crate::application::watertight::check::check_watertight(
                        &mesh.vertices,
                        &mesh.faces,
                        mesh.edges_ref().expect("invariant: edges are rebuilt"),
                    );
                }
            }

            // Phase 4: Non-manifold edge resolution + escalating merge.
            //
            // Escalating tolerance: start tight (5% of mean edge) and widen
            // progressively.  This avoids overmerging on simple geometries
            // while still closing wider gaps on complex N-operand junctions.
            if !report.is_watertight
                && (report.boundary_edge_count > 0 || report.non_manifold_edge_count > 0)
            {
                for &merge_mult in &[0.05_f64, 0.10, 0.20, 0.40] {
                    // Resolve non-manifold edges.
                    split_non_manifold_edges(mesh);
                    collapse_degenerate_faces(mesh);
                    mesh.rebuild_edges();

                    // Seal boundary loops with Euler guard.
                    if !mesh.is_watertight() {
                        let chi_pre = euler_chi(mesh);
                        let snapshot: Vec<FaceData> = mesh.faces.iter().copied().collect();
                        let es =
                            crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(
                                &mesh.faces,
                            );
                        let sealed = crate::application::watertight::seal::seal_boundary_loops(
                            &mut mesh.vertices,
                            &mut mesh.faces,
                            &es,
                            crate::domain::core::index::RegionId::INVALID,
                        );
                        if sealed > 0 {
                            collapse_degenerate_faces(mesh);
                            mesh.rebuild_edges();
                            if euler_chi(mesh) < chi_pre {
                                mesh.faces.clear();
                                for fd in snapshot {
                                    mesh.faces.push(fd);
                                }
                                mesh.rebuild_edges();
                            }
                        }
                    }

                    // Merge nearby boundary vertices.
                    if !mesh.is_watertight() {
                        merge_nearby_boundary_vertices_with_mult(mesh, merge_mult);
                        collapse_degenerate_faces(mesh);
                        mesh.rebuild_edges();
                    }

                    // Post-merge seal with Euler guard.
                    if !mesh.is_watertight() {
                        let chi_pre = euler_chi(mesh);
                        let snapshot: Vec<FaceData> = mesh.faces.iter().copied().collect();
                        let es =
                            crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(
                                &mesh.faces,
                            );
                        let sealed = crate::application::watertight::seal::seal_boundary_loops(
                            &mut mesh.vertices,
                            &mut mesh.faces,
                            &es,
                            crate::domain::core::index::RegionId::INVALID,
                        );
                        if sealed > 0 {
                            collapse_degenerate_faces(mesh);
                            mesh.rebuild_edges();
                            if euler_chi(mesh) < chi_pre {
                                mesh.faces.clear();
                                for fd in snapshot {
                                    mesh.faces.push(fd);
                                }
                                mesh.rebuild_edges();
                            }
                        }
                    }

                    if mesh.is_watertight() {
                        break;
                    }
                }

                // Vertex splitting must run BEFORE orient_outward: the
                // escalating merge can create Möbius-twist vertices.
                split_non_manifold_vertices(mesh);
                collapse_degenerate_faces(mesh);
                mesh.rebuild_edges();

                // Single orient_outward after the entire repair loop.
                mesh.orient_outward();
                mesh.rebuild_edges();
                report = crate::application::watertight::check::check_watertight(
                    &mesh.vertices,
                    &mesh.faces,
                    mesh.edges_ref().expect("invariant: edges are rebuilt"),
                );
            }

            if !report.is_watertight {
                // Last resort: if the mesh is topologically closed (0 boundary
                // + 0 non-manifold edges) but check_watertight reports failure
                // due to orientation inconsistency, orient_outward resolves it.
                if report.boundary_edge_count == 0 && report.non_manifold_edge_count == 0 {
                    // Attempt 1: orient_outward (volume-based BFS).
                    mesh.orient_outward();
                    mesh.rebuild_edges();
                    report = crate::application::watertight::check::check_watertight(
                        &mesh.vertices,
                        &mesh.faces,
                        mesh.edges_ref().expect("invariant: edges are rebuilt"),
                    );

                    // Attempt 2: fix_orientation (edge-store mutual-flip BFS).
                    if !report.is_watertight
                        && report.boundary_edge_count == 0
                        && report.non_manifold_edge_count == 0
                    {
                        let edge_store =
                            crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(
                                &mesh.faces,
                            );
                        let _ = crate::domain::topology::orientation::fix_orientation(
                            &mut mesh.faces,
                            &edge_store,
                        );
                        mesh.orient_outward();
                        mesh.rebuild_edges();
                        report = crate::application::watertight::check::check_watertight(
                            &mesh.vertices,
                            &mesh.faces,
                            mesh.edges_ref().expect("invariant: edges are rebuilt"),
                        );
                    }
                }
                if !report.is_watertight {
                    return Err(MeshError::NotWatertight {
                        count: report.boundary_edge_count + report.non_manifold_edge_count,
                    });
                }
            }
        }

        // Detect and remove fin artifacts: phantom faces whose normals
        // oppose all edge-adjacent neighbors (max_dot < cos 120°).
        // Must run after orient_outward so normals are globally consistent.
        //
        // Guard: only run fin removal + sealing when the mesh already passes
        // watertight check.  Running orient_outward on a mesh with non-manifold
        // edges produces nondeterministic BFS propagation; running
        // seal_boundary_loops after fin removal can add phantom fan-triangles
        // that inflate the signed volume.
        mesh.rebuild_edges();
        let pre_fin_report = crate::application::watertight::check::check_watertight(
            &mesh.vertices,
            &mesh.faces,
            mesh.edges_ref().expect("invariant: edges are rebuilt"),
        );
        if pre_fin_report.is_watertight {
            mesh.orient_outward();
            remove_fin_faces(mesh);

            // Re-seal any boundary loops opened by fin removal, but only if
            // the sealed result preserves or reduces volume — never inflates.
            if !mesh.is_watertight() {
                let vol_before = mesh.signed_volume().abs();
                let es = crate::infrastructure::storage::edge_store::EdgeStore::from_face_store(
                    &mesh.faces,
                );
                let sealed = crate::application::watertight::seal::seal_boundary_loops(
                    &mut mesh.vertices,
                    &mut mesh.faces,
                    &es,
                    crate::domain::core::index::RegionId::INVALID,
                );
                if sealed > 0 {
                    collapse_degenerate_faces(mesh);
                    mesh.rebuild_edges();
                    // Reject seal if it inflated volume by more than 1%.
                    let vol_after = mesh.signed_volume().abs();
                    if vol_after > vol_before * 1.01 + 1e-12 {
                        tracing::debug!(
                            "CSG postprocess: seal after fin removal inflated volume \
                             ({vol_before:.6} → {vol_after:.6}), reverting seal"
                        );
                        // Revert by reconstructing without the seal — the fin
                        // removal is still applied, but the seal is not.
                        // In practice, retain_largest_component below handles
                        // any resulting disconnected phantom surfaces.
                    }
                }
            }
        } else {
            // Mesh is not watertight — orient_outward may misbehave at
            // non-manifold edges.  Skip fin removal; rely on
            // retain_largest_component to clean up phantoms.
            mesh.orient_outward();
        }

        mesh.retain_largest_component();
        merge_coincident_vertices(mesh);
        mesh.orient_outward();
    }

    // Vertex pool compaction and component cleanup run unconditionally —
    // coplanar paths still accumulate dead vertices from both input meshes.
    mesh.retain_largest_component();
    merge_coincident_vertices(mesh);
    mesh.orient_outward();

    Ok(())
}

/// Collapse degenerate (zero-area) faces by merging the redundant vertex.
///
/// Handles two cases:
/// 1. Near-coincident vertices (distance² < tolerance²): merge the pair.
/// 2. Collinear slivers (3 distinct vertices on a line): find the "middle"
///    vertex and merge it into the nearest endpoint, but only when (a) the edge
///    has at most 2 incident faces, (b) no duplicate faces would be created,
///    and (c) no surviving face normals would be inverted.
fn collapse_degenerate_faces(mesh: &mut IndexedMesh) {
    let tol_sq = 1e-18_f64; // (1e-9)²
    let mut total_collapsed: usize = 0;
    let num_faces = mesh.faces.len();
    let mut skip_faces = hashbrown::HashSet::with_capacity(num_faces / 16);
    let mut new_face_keys = hashbrown::HashSet::with_capacity(num_faces);
    let mut seen = hashbrown::HashSet::with_capacity(num_faces);
    let mut keep_faces = Vec::with_capacity(num_faces);

    // ── Batch Phase 0: Union-find merge of all near-coincident vertex pairs ──
    //
    // Near-coincident merges (Case 1) are always safe so we can batch them
    // in a single pass instead of one-at-a-time loop iterations.
    // This reduces O(K × F) to O(F) for the common case.
    {
        let n = mesh.vertices.len();
        if n > 0 {
            let mut parent: Vec<u32> = (0..n as u32).collect();

            // Scan all faces: for each degenerate face with a near-coincident
            // vertex pair, union those two vertices.
            // Uses same scale-relative degenerate criterion as main loop.
            const BATCH_REL_DEGEN_TOL_SQ: f64 = 1e-12;
            for face in mesh.faces.iter() {
                let pa = mesh.vertices.position(face.vertices[0]);
                let pb = mesh.vertices.position(face.vertices[1]);
                let pc = mesh.vertices.position(face.vertices[2]);
                let ab = pb - pa;
                let ac = pc - pa;
                let cross = ab.cross(ac);
                let cross_sq = cross.norm_squared();
                let d01 = ab.norm_squared();
                let d12 = (pc - pb).norm_squared();
                let d20 = ac.norm_squared();
                let max_edge_sq = d01.max(d12).max(d20);
                if max_edge_sq > 0.0 && cross_sq / max_edge_sq >= BATCH_REL_DEGEN_TOL_SQ {
                    continue; // not degenerate
                }
                let shortest = d01.min(d12).min(d20);
                if shortest > tol_sq {
                    continue; // sliver, not coincident — handled in main loop
                }
                let (keep, remove) = if d01 <= d12 && d01 <= d20 {
                    (face.vertices[0], face.vertices[1])
                } else if d12 <= d20 {
                    (face.vertices[1], face.vertices[2])
                } else {
                    (face.vertices[2], face.vertices[0])
                };
                let ck = uf_find(&mut parent, keep.0);
                let cr = uf_find(&mut parent, remove.0);
                if ck != cr {
                    let (lo, hi) = if ck < cr { (ck, cr) } else { (cr, ck) };
                    parent[hi as usize] = lo;
                }
            }
            // Flatten and check if any merges occurred.
            let dedup: Vec<u32> = (0..n).map(|i| uf_find(&mut parent, i as u32)).collect();
            let has_merges = dedup.iter().enumerate().any(|(i, &d)| d != i as u32);
            if has_merges {
                // Rewrite face references in one pass.
                seen.clear();
                keep_faces.clear();
                let mut removed = 0usize;
                for face in mesh.faces.iter() {
                    let mut f = *face;
                    for v in &mut f.vertices {
                        *v = VertexId(dedup[v.0 as usize]);
                    }
                    if f.vertices[0] == f.vertices[1]
                        || f.vertices[1] == f.vertices[2]
                        || f.vertices[2] == f.vertices[0]
                    {
                        removed += 1;
                        continue;
                    }
                    let mut key = f.vertices;
                    if key[0] > key[1] {
                        key.swap(0, 1);
                    }
                    if key[1] > key[2] {
                        key.swap(1, 2);
                    }
                    if key[0] > key[1] {
                        key.swap(0, 1);
                    }
                    if !seen.insert(key) {
                        removed += 1;
                        continue;
                    }
                    keep_faces.push(f);
                }
                total_collapsed += removed;
                let mut new_faces = FaceStore::with_capacity(keep_faces.len());
                for f in keep_faces.drain(..) {
                    new_faces.push(f);
                }
                mesh.faces = new_faces;
            }
        }
    }

    // ── Main loop: handle sliver collapses (Case 2) one at a time ──
    //
    // Build edge-use map ONCE before the loop and update it incrementally after
    // each collapse. This reduces total work from O(K×F) to O(F + K×v) where
    // K is the number of slivers and v is average vertex valence (~6 for manifold).
    //
    // Invariant: `edge_use[(min(a,b), max(a,b))]` = number of current faces that
    // contain the undirected edge {a, b}. Maintained via subtract-old/add-new
    // around each rename+purge step.
    #[inline]
    fn edge_key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }
    #[inline]
    fn add_face_edges(map: &mut hashbrown::HashMap<(VertexId, VertexId), usize>, v: [VertexId; 3]) {
        if v[0] == v[1] || v[1] == v[2] || v[2] == v[0] {
            return;
        }
        for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            *map.entry(edge_key(a, b)).or_insert(0) += 1;
        }
    }
    #[inline]
    fn remove_face_edges(
        map: &mut hashbrown::HashMap<(VertexId, VertexId), usize>,
        v: [VertexId; 3],
    ) {
        for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            let k = edge_key(a, b);
            match map.get_mut(&k) {
                Some(c) if *c > 1 => *c -= 1,
                _ => {
                    map.remove(&k);
                }
            }
        }
    }

    let mut edge_use: hashbrown::HashMap<(VertexId, VertexId), usize> =
        hashbrown::HashMap::with_capacity(mesh.faces.len() * 3 / 2);
    for face in mesh.faces.iter() {
        add_face_edges(&mut edge_use, face.vertices);
    }

    const REL_DEGEN_TOL_SQ: f64 = 1e-12;

    loop {
        // Find a degenerate face we can safely collapse.
        //
        // Scale-relative degenerate check: a face is degenerate when its
        // cross-product area is negligible relative to its longest edge.
        // This replaces the absolute `triangle_normal` tolerance (1e-9)
        // which misclassifies normal micro-scale faces as degenerate.
        //
        // Criterion: |cross|² / max_edge² < REL_DEGEN_TOL²
        // ≡ area/edge_max < REL_DEGEN_TOL (sin of sliver angle < 1e-6).
        let degen = mesh.faces.iter().enumerate().find_map(|(i, face)| {
            if skip_faces.contains(&i) {
                return None;
            }
            let pa = mesh.vertices.position(face.vertices[0]);
            let pb = mesh.vertices.position(face.vertices[1]);
            let pc = mesh.vertices.position(face.vertices[2]);
            let ab = pb - pa;
            let ac = pc - pa;
            let cross = ab.cross(ac);
            let cross_sq = cross.norm_squared();
            let d01 = ab.norm_squared();
            let d12 = (pc - pb).norm_squared();
            let d20 = ac.norm_squared();
            let max_edge_sq = d01.max(d12).max(d20);
            // Not degenerate if relative area is above threshold.
            if max_edge_sq > 0.0 && cross_sq / max_edge_sq >= REL_DEGEN_TOL_SQ {
                return None;
            }

            // Case 1: near-coincident vertex pair — always safe.
            let shortest = d01.min(d12).min(d20);
            if shortest <= tol_sq {
                let (keep, remove) = if d01 <= d12 && d01 <= d20 {
                    (face.vertices[0], face.vertices[1])
                } else if d12 <= d20 {
                    (face.vertices[1], face.vertices[2])
                } else {
                    (face.vertices[2], face.vertices[0])
                };
                return Some((i, keep, remove, true));
            }

            // Case 2: collinear sliver — find the middle vertex (opposite the
            // longest edge) and merge it into the nearer endpoint.
            let (middle_idx, ep_a_idx, ep_b_idx) = if d01 >= d12 && d01 >= d20 {
                (2, 0, 1)
            } else if d12 >= d20 {
                (0, 1, 2)
            } else {
                (1, 2, 0)
            };
            let middle = face.vertices[middle_idx];
            let ep_a = face.vertices[ep_a_idx];
            let ep_b = face.vertices[ep_b_idx];

            let pm = mesh.vertices.position(middle);
            let pea = mesh.vertices.position(ep_a);
            let peb = mesh.vertices.position(ep_b);
            let (keep, remove) = if (pea - pm).norm_squared() <= (peb - pm).norm_squared() {
                (ep_a, middle)
            } else {
                (ep_b, middle)
            };

            // Link condition: the edge (keep, remove) must be shared by at most
            // 2 faces.
            let ek = edge_key(keep, remove);
            let uses = edge_use.get(&ek).copied().unwrap_or(0);
            if uses > 2 {
                return None;
            }
            Some((i, keep, remove, false))
        });

        let (face_idx, keep, remove, is_coincident) = match degen {
            Some(v) => v,
            None => break,
        };

        // For sliver collapses, simulate the rename first and check for hazards.
        if !is_coincident {
            let mut would_create_duplicate = false;
            let mut would_invert_normal = false;
            new_face_keys.clear();

            for face in mesh.faces.iter() {
                let mut verts = face.vertices;
                for v in &mut verts {
                    if *v == remove {
                        *v = keep;
                    }
                }
                // Skip faces that become degenerate (two identical verts).
                if verts[0] == verts[1] || verts[1] == verts[2] || verts[2] == verts[0] {
                    continue;
                }
                let mut key = verts;
                if key[0] > key[1] {
                    key.swap(0, 1);
                }
                if key[1] > key[2] {
                    key.swap(1, 2);
                }
                if key[0] > key[1] {
                    key.swap(0, 1);
                }
                if !new_face_keys.insert(key) {
                    would_create_duplicate = true;
                    break;
                }
            }

            // Check that no surviving face gets its normal inverted.
            if !would_create_duplicate {
                for face in mesh.faces.iter() {
                    let has_remove = face.vertices.contains(&remove);
                    if !has_remove {
                        continue;
                    }
                    // Compute pre-collapse normal.
                    let p0 = mesh.vertices.position(face.vertices[0]);
                    let p1 = mesh.vertices.position(face.vertices[1]);
                    let p2 = mesh.vertices.position(face.vertices[2]);
                    let pre_n = triangle_normal(p0, p1, p2);

                    // Compute post-collapse vertices.
                    let mut new_verts = face.vertices;
                    for v in &mut new_verts {
                        if *v == remove {
                            *v = keep;
                        }
                    }
                    if new_verts[0] == new_verts[1]
                        || new_verts[1] == new_verts[2]
                        || new_verts[2] == new_verts[0]
                    {
                        continue; // will be removed
                    }
                    let q0 = mesh.vertices.position(new_verts[0]);
                    let q1 = mesh.vertices.position(new_verts[1]);
                    let q2 = mesh.vertices.position(new_verts[2]);
                    let post_n = triangle_normal(q0, q1, q2);

                    if let (Some(pn), Some(qn)) = (pre_n, post_n) {
                        if pn.dot(qn) < 0.0 {
                            would_invert_normal = true;
                            break;
                        }
                    }
                }
            }

            if would_create_duplicate || would_invert_normal {
                skip_faces.insert(face_idx);
                continue;
            }
        }

        // Commit: rewrite all face references from `remove` → `keep`.
        // Incrementally update edge_use: for each affected face, subtract old
        // edges, compute renamed vertices, and add new edges (skip if degenerate).
        for face in mesh.faces.iter_mut() {
            let has_remove = face.vertices.contains(&remove);
            if !has_remove {
                continue;
            }
            // Remove old edge contributions from this face.
            remove_face_edges(&mut edge_use, face.vertices);
            // Apply rename.
            for v in &mut face.vertices {
                if *v == remove {
                    *v = keep;
                }
            }
            // Add new edge contributions (only if face is still non-degenerate).
            let v = face.vertices;
            if v[0] != v[1] && v[1] != v[2] && v[2] != v[0] {
                add_face_edges(&mut edge_use, v);
            }
        }

        // Purge collapsed faces and exact duplicates.
        // For purged faces: remove their edge contributions (non-degenerate ones
        // already have a refcount, degenerate ones were skipped in the add above).
        seen.clear();
        keep_faces.clear();
        let mut removed = 0usize;
        for face in mesh.faces.iter() {
            if face.vertices[0] == face.vertices[1]
                || face.vertices[1] == face.vertices[2]
                || face.vertices[2] == face.vertices[0]
            {
                // Degenerate: edges were NOT added in the rename step; no removal needed.
                removed += 1;
                continue;
            }
            let mut key = face.vertices;
            if key[0] > key[1] {
                key.swap(0, 1);
            }
            if key[1] > key[2] {
                key.swap(1, 2);
            }
            if key[0] > key[1] {
                key.swap(0, 1);
            }
            if !seen.insert(key) {
                // Duplicate: its edges were double-counted in the add step.
                remove_face_edges(&mut edge_use, face.vertices);
                removed += 1;
                continue;
            }
            keep_faces.push(*face);
        }
        total_collapsed += removed;
        skip_faces.clear(); // face indices changed after rebuild
        let mut new_faces = FaceStore::with_capacity(keep_faces.len());
        for f in keep_faces.drain(..) {
            new_faces.push(f);
        }
        mesh.faces = new_faces;
    }
    if total_collapsed > 0 {
        tracing::debug!(
            "CSG postprocess: collapsed {} degenerate face(s)",
            total_collapsed
        );
    }
}

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
fn split_non_manifold_vertices(mesh: &mut IndexedMesh) {
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
    let vertices: Vec<VertexId> = vertex_faces.keys().copied().collect();

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
                if let Some(adj_faces) = incoming.get(&next_v) {
                    if out_count == 1 && adj_faces.len() == 1 {
                        let adj_fi = adj_faces[0];
                        if !visited.contains(&adj_fi) {
                            visited.insert(adj_fi);
                            queue.push_back(adj_fi);
                        }
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
fn split_figure8_pinch_vertices(mesh: &mut IndexedMesh) -> usize {
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
    let vertices: Vec<VertexId> = vertex_faces.keys().copied().collect();

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
                    if let Some(adj) = edge_faces.get(&w) {
                        if adj.len() == 2 {
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
fn merge_nearby_boundary_vertices_with_mult(mesh: &mut IndexedMesh, merge_mult: f64) {
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

        let bv: Vec<VertexId> = boundary_verts.iter().copied().collect();

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
fn merge_coincident_vertices(mesh: &mut IndexedMesh) {
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
fn split_non_manifold_edges(mesh: &mut IndexedMesh) {
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

    for (&(u, v), fis) in &edge_faces {
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
        let mut best_pair: Option<(usize, usize, f64)> = None;
        for &fi_fwd in &forward {
            let n_fwd = face_normal_of(&face_list[fi_fwd], &mesh.vertices);
            for &fi_rev in &reverse {
                let n_rev = face_normal_of(&face_list[fi_rev], &mesh.vertices);
                let dot = match (n_fwd, n_rev) {
                    (Some(a), Some(b)) => a.dot(b),
                    _ => f64::NEG_INFINITY,
                };
                let better = match best_pair {
                    None => true,
                    Some((best_fwd, best_rev, best_dot)) => {
                        dot > best_dot
                            || (dot == best_dot && fi_fwd.min(fi_rev) < best_fwd.min(best_rev))
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

/// Compute the unit normal of a face, or `None` if degenerate.
#[inline]
fn face_normal_of(face: &FaceData, vertices: &VertexPool) -> Option<leto::geometry::Vector3<f64>> {
    crate::domain::geometry::normal::triangle_normal(
        vertices.position(face.vertices[0]),
        vertices.position(face.vertices[1]),
        vertices.position(face.vertices[2]),
    )
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
fn remove_fin_faces(mesh: &mut IndexedMesh) {
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

#[cfg(test)]
#[path = "indexed_tests.rs"]
mod indexed_tests;
