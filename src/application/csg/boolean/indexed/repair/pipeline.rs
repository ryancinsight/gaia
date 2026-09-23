//! Post-Boolean repair pipeline orchestration.

use super::collapse::collapse_degenerate_faces;
use super::edges::{remove_fin_faces, split_non_manifold_edges};
use super::merge::{merge_coincident_vertices, merge_nearby_boundary_vertices_with_mult};
use super::vertices::{split_figure8_pinch_vertices, split_non_manifold_vertices};
use crate::application::csg::reconstruct;
use crate::application::watertight::check::{check_watertight, WatertightReport};
use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::RegionId;
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

fn euler_characteristic(mesh: &IndexedMesh) -> i64 {
    let edges = EdgeStore::from_face_store(&mesh.faces);
    crate::application::watertight::check::euler_chi_from_stores(&mesh.faces, &edges)
}

fn watertight_report(mesh: &IndexedMesh) -> WatertightReport {
    check_watertight(
        &mesh.vertices,
        &mesh.faces,
        mesh.edges_ref().expect("invariant: edges are rebuilt"),
    )
}

fn repair_orientation(mesh: &mut IndexedMesh, report: &mut WatertightReport) {
    if report.is_closed && !report.orientation_consistent {
        mesh.orient_outward();
        mesh.rebuild_edges();
        *report = watertight_report(mesh);

        if !report.is_watertight && report.is_closed && !report.orientation_consistent {
            let edges = EdgeStore::from_face_store(&mesh.faces);
            let _ = crate::domain::topology::orientation::fix_orientation(&mut mesh.faces, &edges);
            mesh.rebuild_edges();
            *report = watertight_report(mesh);
        }
    }
}

fn seal_small_boundary(mesh: &mut IndexedMesh, report: &mut WatertightReport) {
    if !report.is_watertight
        && report.non_manifold_edge_count == 0
        && report.boundary_edge_count > 0
        && report.boundary_edge_count <= 512
    {
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let added = crate::application::watertight::seal::seal_boundary_loops(
            &mut mesh.vertices,
            &mut mesh.faces,
            &edges,
            RegionId::INVALID,
        );
        if added > 0 {
            mesh.rebuild_edges();
            *report = watertight_report(mesh);
        }
    }
}

fn stitch_boundary(mesh: &mut IndexedMesh, report: &mut WatertightReport) {
    if !report.is_watertight && report.boundary_edge_count > 0 {
        let improved =
            crate::application::watertight::repair::MeshRepair::iterative_boundary_stitch(
                &mut mesh.faces,
                &mesh.vertices,
                3,
            );
        if improved > 0 {
            mesh.rebuild_edges();
            *report = watertight_report(mesh);
        }
    }
}

fn seal_boundary_preserving_euler(mesh: &mut IndexedMesh) {
    if !mesh.is_watertight() {
        let characteristic_before = euler_characteristic(mesh);
        let original_faces: Vec<FaceData> = mesh.faces.iter().copied().collect();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let sealed = crate::application::watertight::seal::seal_boundary_loops(
            &mut mesh.vertices,
            &mut mesh.faces,
            &edges,
            RegionId::INVALID,
        );
        if sealed > 0 {
            collapse_degenerate_faces(mesh);
            mesh.rebuild_edges();
            if euler_characteristic(mesh) < characteristic_before {
                mesh.faces.clear();
                for face in original_faces {
                    mesh.faces.push(face);
                }
                mesh.rebuild_edges();
            }
        }
    }
}

fn escalate_boundary_repair(mesh: &mut IndexedMesh) {
    for merge_mult in [0.05_f64, 0.10, 0.20, 0.40] {
        split_non_manifold_edges(mesh);
        collapse_degenerate_faces(mesh);
        mesh.rebuild_edges();

        seal_boundary_preserving_euler(mesh);

        if !mesh.is_watertight() {
            merge_nearby_boundary_vertices_with_mult(mesh, merge_mult);
            collapse_degenerate_faces(mesh);
            mesh.rebuild_edges();
        }

        seal_boundary_preserving_euler(mesh);

        if mesh.is_watertight() {
            break;
        }
    }

    split_non_manifold_vertices(mesh);
    collapse_degenerate_faces(mesh);
    mesh.rebuild_edges();
    mesh.orient_outward();
    mesh.rebuild_edges();
}

fn repair_closed_orientation(
    mesh: &mut IndexedMesh,
    mut report: WatertightReport,
) -> WatertightReport {
    if report.is_watertight
        || report.boundary_edge_count != 0
        || report.non_manifold_edge_count != 0
    {
        return report;
    }

    mesh.orient_outward();
    mesh.rebuild_edges();
    report = watertight_report(mesh);

    if !report.is_watertight
        && report.boundary_edge_count == 0
        && report.non_manifold_edge_count == 0
    {
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let _ = crate::domain::topology::orientation::fix_orientation(&mut mesh.faces, &edges);
        mesh.orient_outward();
        mesh.rebuild_edges();
        report = watertight_report(mesh);
    }

    report
}

fn repair_topology(mesh: &mut IndexedMesh) -> MeshResult<()> {
    mesh.rebuild_edges();
    let mut report = watertight_report(mesh);
    if report.is_watertight {
        return Ok(());
    }

    repair_orientation(mesh, &mut report);
    seal_small_boundary(mesh, &mut report);
    stitch_boundary(mesh, &mut report);

    if !report.is_watertight
        && (report.boundary_edge_count > 0 || report.non_manifold_edge_count > 0)
    {
        escalate_boundary_repair(mesh);
        report = watertight_report(mesh);
    }

    report = repair_closed_orientation(mesh, report);
    if !report.is_watertight {
        return Err(MeshError::NotWatertight {
            count: report.boundary_edge_count + report.non_manifold_edge_count,
        });
    }

    Ok(())
}

fn reject_inflated_seal(
    mesh: &mut IndexedMesh,
    original_faces: Vec<FaceData>,
    volume_before: f64,
) -> bool {
    let volume_after = mesh.signed_volume().abs();
    if volume_after <= volume_before * 1.01 + 1e-12 {
        return false;
    }

    mesh.faces.clear();
    for face in original_faces {
        mesh.faces.push(face);
    }
    mesh.rebuild_edges();
    tracing::debug!(
        "CSG postprocess: seal after fin removal inflated volume \
         ({volume_before:.6} → {volume_after:.6}); discarded the seal"
    );
    true
}

fn remove_fin_artifacts(mesh: &mut IndexedMesh) {
    mesh.rebuild_edges();
    let report = watertight_report(mesh);
    if report.is_watertight {
        mesh.orient_outward();
        remove_fin_faces(mesh);

        if !mesh.is_watertight() {
            let volume_before = mesh.signed_volume().abs();
            let original_faces: Vec<FaceData> = mesh.faces.iter().copied().collect();
            let edges = EdgeStore::from_face_store(&mesh.faces);
            let sealed = crate::application::watertight::seal::seal_boundary_loops(
                &mut mesh.vertices,
                &mut mesh.faces,
                &edges,
                RegionId::INVALID,
            );
            if sealed > 0 {
                collapse_degenerate_faces(mesh);
                mesh.rebuild_edges();
                reject_inflated_seal(mesh, original_faces, volume_before);
            }
        }
    } else {
        mesh.orient_outward();
    }
}

fn cleanup_mesh(mesh: &mut IndexedMesh) {
    mesh.retain_largest_component();
    merge_coincident_vertices(mesh);
    mesh.orient_outward();
}

/// Repair Boolean mesh topology and remove phantom fin artifacts.
///
/// Non-coplanar results receive orientation, boundary, and non-manifold
/// repair before fin removal. Both paths then compact connected geometry and
/// weld coincident vertices.
///
/// # Errors
///
/// Returns [`MeshError::NotWatertight`] when a non-coplanar result cannot be
/// repaired into a watertight mesh.
pub(super) fn repair_boolean_mesh(mesh: &mut IndexedMesh, is_coplanar: bool) -> MeshResult<()> {
    if !is_coplanar {
        repair_topology(mesh)?;
        remove_fin_artifacts(mesh);
        cleanup_mesh(mesh);
    }

    // Coplanar paths also accumulate dead vertices during operand remapping.
    cleanup_mesh(mesh);
    Ok(())
}

pub(in crate::application::csg::boolean::indexed) fn postprocess_boolean_mesh(
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

#[cfg(test)]
mod tests {
    use super::reject_inflated_seal;
    use crate::domain::geometry::primitives::{Cube, PrimitiveMesh};
    use crate::domain::mesh::IndexedMesh;
    use crate::infrastructure::storage::face_store::FaceData;

    fn cube() -> IndexedMesh {
        Cube::unit().build().expect("unit cube builds")
    }

    #[test]
    fn inflated_boundary_repair_restores_original_faces() {
        let mut mesh = cube();
        let original_faces: Vec<FaceData> = mesh.faces.iter().copied().collect();
        let original_volume = mesh.signed_volume().abs();

        for face in original_faces.iter().copied() {
            mesh.faces.push(face);
        }
        assert!(mesh.signed_volume().abs() > original_volume * 1.01 + 1e-12);

        assert!(reject_inflated_seal(
            &mut mesh,
            original_faces.clone(),
            original_volume,
        ));

        let restored_faces: Vec<FaceData> = mesh.faces.iter().copied().collect();
        assert_eq!(restored_faces, original_faces);
        assert_eq!(mesh.signed_volume().abs(), original_volume);
        assert!(mesh.is_watertight());
    }

    #[test]
    fn volume_guard_keeps_a_seal_within_the_bound() {
        let mut mesh = cube();
        let original_faces: Vec<FaceData> = mesh.faces.iter().copied().collect();
        let original_volume = mesh.signed_volume().abs();

        assert!(!reject_inflated_seal(
            &mut mesh,
            original_faces,
            original_volume,
        ));
        assert_eq!(mesh.signed_volume().abs(), original_volume);
        assert!(mesh.is_watertight());
    }
}
