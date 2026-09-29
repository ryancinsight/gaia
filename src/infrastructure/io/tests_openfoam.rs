//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::scalar::Point3r;
use crate::domain::mesh::MeshBuilder;
use crate::test_support::assert_rejects;

/// Helper: build a tiny tetrahedron IndexedMesh.
fn tet_mesh() -> IndexedMesh {
    let mut b = MeshBuilder::new();
    let a = b.vertex(Point3r::new(0.0, 0.0, 0.0));
    let bv = b.vertex(Point3r::new(1.0, 0.0, 0.0));
    let c = b.vertex(Point3r::new(0.0, 1.0, 0.0));
    let d = b.vertex(Point3r::new(0.0, 0.0, 1.0));
    b.triangle(a, bv, c);
    b.triangle(a, c, d);
    b.triangle(a, d, bv);
    b.triangle(bv, d, c);
    b.build()
}

#[test]
fn write_openfoam_creates_all_five_files() {
    let mesh = tet_mesh();
    let dir = std::env::temp_dir().join("gaia_of_test_five");
    write_openfoam_polymesh(&mesh, &dir, &[]).expect("write should succeed");

    for name in ["points", "faces", "owner", "neighbour", "boundary"] {
        assert!(
            dir.join(name).exists(),
            "expected file {name} to exist in output directory"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn points_file_has_correct_vertex_count() {
    let mesh = tet_mesh();
    let dir = std::env::temp_dir().join("gaia_of_test_vcount");
    write_openfoam_polymesh(&mesh, &dir, &[]).unwrap();
    let content = std::fs::read_to_string(dir.join("points")).unwrap();
    let count_line = content
        .lines()
        .find(|l| l.trim().parse::<usize>().is_ok())
        .unwrap_or("");
    assert_eq!(
        count_line.trim(),
        "4",
        "expected vertex count 4 in points file"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn faces_file_has_correct_face_count() {
    let mesh = tet_mesh();
    let dir = std::env::temp_dir().join("gaia_of_test_fcount");
    write_openfoam_polymesh(&mesh, &dir, &[]).unwrap();
    let content = std::fs::read_to_string(dir.join("faces")).unwrap();
    let count_line = content
        .lines()
        .find(|l| l.trim().parse::<usize>().is_ok())
        .unwrap_or("");
    assert_eq!(count_line.trim(), "4", "tetrahedron has 4 faces");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn boundary_file_default_patch_when_no_regions() {
    let mesh = tet_mesh();
    let dir = std::env::temp_dir().join("gaia_of_test_boundary");
    write_openfoam_polymesh(&mesh, &dir, &[]).unwrap();
    let content = std::fs::read_to_string(dir.join("boundary")).unwrap();
    assert!(
        content.contains("defaultFaces"),
        "boundary should contain 'defaultFaces' when no regions specified"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn boundary_file_named_patches_respected() {
    use crate::domain::mesh::IndexedMesh;

    // Build a mesh with two regions
    let mut mesh = IndexedMesh::new();
    let v0 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
    let v1 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
    let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
    let v3 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 1.0));
    let r0 = RegionId::from_usize(0);
    let r1 = RegionId::from_usize(1);
    mesh.add_face_with_region(v0, v1, v2, r0);
    mesh.add_face_with_region(v0, v2, v3, r1);

    let dir = std::env::temp_dir().join("gaia_of_test_named");
    write_openfoam_polymesh(
        &mesh,
        &dir,
        &[
            (r0, "inlet", PatchType::Inlet),
            (r1, "outlet", PatchType::Outlet),
        ],
    )
    .unwrap();

    let content = std::fs::read_to_string(dir.join("boundary")).unwrap();
    assert!(content.contains("inlet"), "should contain 'inlet'");
    assert!(content.contains("outlet"), "should contain 'outlet'");
    assert!(
        content.contains("physicalType    inlet"),
        "inlet physicalType"
    );
    assert!(
        content.contains("physicalType    outlet"),
        "outlet physicalType"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn empty_mesh_returns_error() {
    let mesh = IndexedMesh::new();
    let dir = std::env::temp_dir().join("gaia_of_test_empty");
    let result = write_openfoam_polymesh(&mesh, &dir, &[]);
    assert_rejects(&result, "cannot write empty mesh to OpenFOAM format");
}
