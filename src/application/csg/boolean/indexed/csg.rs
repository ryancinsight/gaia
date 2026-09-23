//! Indexed-mesh CSG entry points and operand preparation.

use super::repair::postprocess_boolean_mesh;
use crate::application::csg::boolean::normalization::{
    denormalize_result, normalization_transform, normalize_operand, NormalizedOperand,
};
use crate::application::csg::boolean::BooleanOp;
use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::VertexId;
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

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
    if op == BooleanOp::Union
        && let Some(union) =
            super::super::rectangular_prism::rectangular_prism_union(mesh_a, mesh_b)
    {
        return Ok(union);
    }

    let transform = normalization_transform([mesh_a, mesh_b]);
    let normalized_a = normalize_operand(mesh_a, transform);
    let normalized_b = normalize_operand(mesh_b, transform);
    let mut combined = VertexPool::for_csg_with_scale(1.0);
    let (faces_a, faces_b) = remap_binary_face_soups(
        normalized_a.as_mesh(),
        normalized_b.as_mesh(),
        &mut combined,
    );
    let is_coplanar = crate::application::csg::coplanar::detect_flat_plane(&faces_a, &combined)
        .is_some()
        && crate::application::csg::coplanar::detect_flat_plane(&faces_b, &combined).is_some();
    let result_faces = crate::application::csg::arrangement::boolean_csg::csg_boolean_unfinalized(
        op,
        &[faces_a, faces_b],
        &mut combined,
    )?;
    postprocess_boolean_mesh(result_faces, &combined, is_coplanar)
        .map(|mesh| denormalize_result(mesh, transform))
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

    let transform = normalization_transform(meshes.iter());
    let normalized = meshes
        .iter()
        .map(|mesh| normalize_operand(mesh, transform))
        .collect::<Vec<_>>();
    let mut combined = VertexPool::for_csg_with_scale(1.0);
    let face_soups = remap_nary_face_soups(&normalized, &mut combined);
    let is_coplanar = face_soups.iter().all(|faces| {
        crate::application::csg::coplanar::detect_flat_plane(faces, &combined).is_some()
    });
    let result_faces = crate::application::csg::arrangement::boolean_csg::csg_boolean_unfinalized(
        op,
        &face_soups,
        &mut combined,
    )?;
    postprocess_boolean_mesh(result_faces, &combined, is_coplanar)
        .map(|mesh| denormalize_result(mesh, transform))
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

fn remap_nary_face_soups(
    meshes: &[NormalizedOperand<'_>],
    combined: &mut VertexPool,
) -> Vec<Vec<FaceData>> {
    let mut face_soups = Vec::with_capacity(meshes.len());
    for operand in meshes {
        let mesh = operand.as_mesh();
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
