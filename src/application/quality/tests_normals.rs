//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::primitives::{Cube, PrimitiveMesh, UvSphere};

#[test]
fn sphere_all_outward() {
    let mesh = UvSphere {
        radius: 1.0,
        segments: 32,
        stacks: 16,
        ..Default::default()
    }
    .build()
    .unwrap();
    let r = analyze_normals(&mesh);
    assert_eq!(r.inward_faces, 0, "UV sphere should have zero inward faces");
    assert_eq!(
        r.degenerate_faces, 0,
        "UV sphere should have no degenerate faces"
    );
    assert!(
        r.face_vertex_alignment_mean > 0.9,
        "face-vertex alignment mean should be > 0.9, got {}",
        r.face_vertex_alignment_mean
    );
}

#[test]
fn cube_all_outward() {
    let mesh = Cube {
        origin: Point3r::origin(),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .unwrap();
    let r = analyze_normals(&mesh);
    assert_eq!(r.inward_faces, 0, "cube should have zero inward faces");
}

#[test]
fn empty_mesh_returns_zeros() {
    let mesh = IndexedMesh::new();
    let r = analyze_normals(&mesh);
    assert_eq!(r.outward_faces, 0);
    assert_eq!(r.inward_faces, 0);
    assert_eq!(r.degenerate_faces, 0);
    assert_eq!(r.face_vertex_alignment_mean.to_bits(), 0.0_f64.to_bits());
    assert_eq!(r.face_vertex_alignment_min.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn inward_fraction_zero_on_clean_mesh() {
    let mesh = UvSphere {
        radius: 1.0,
        segments: 16,
        stacks: 8,
        ..Default::default()
    }
    .build()
    .unwrap();
    let r = analyze_normals(&mesh);
    assert_eq!(r.inward_fraction().to_bits(), 0.0_f64.to_bits());
    assert!(r.all_outward());
}

#[test]
fn total_faces_matches_mesh() {
    let mesh = UvSphere {
        radius: 1.0,
        segments: 16,
        stacks: 8,
        ..Default::default()
    }
    .build()
    .unwrap();
    let r = analyze_normals(&mesh);
    assert_eq!(r.total_faces(), mesh.face_count());
}

// ── Adversarial BFS analysis tests ────────────────────────────────────

/// # Theorem — Signed-Volume BFS Correction for Inward Meshes
///
/// **Statement**: When `analyze_normals` BFS labels a majority of
/// faces as "outward" but the signed-volume integral is negative,
/// the seed heuristic was wrong.  The outward/inward counts must
/// be swapped so that `inward_faces` reflects the true orientation
/// inconsistency count.
///
/// **Proof**: The BFS seed heuristic (max-X face with $n_x \geq 0$)
/// assumes the extreme face points outward.  For a fully inward-wound
/// mesh, the seed labels all faces "outward" (consistent BFS), but
/// the signed volume is negative.  Swapping the counts corrects the
/// analysis without re-running BFS.
#[test]
fn analyze_normals_all_inward_tet() {
    use crate::domain::mesh::IndexedMesh;

    let mut mesh = IndexedMesh::with_cell_size(0.01);
    let v0 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
    let v1 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
    let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 1.0));
    let v3 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
    // CW winding (inward)
    mesh.add_face(v0, v2, v1);
    mesh.add_face(v0, v1, v3);
    mesh.add_face(v0, v3, v2);
    mesh.add_face(v1, v2, v3);

    let r = analyze_normals(&mesh);
    assert_eq!(r.total_faces(), 4);
    // All faces have the same (inward) winding, so BFS labels them
    // consistently.  The signed-volume swap means all 4 are reported
    // as "inward" after correction.
    assert_eq!(
        r.inward_faces, 4,
        "all-inward tet should report 4 inward faces, got {}",
        r.inward_faces
    );
    assert_eq!(r.outward_faces, 0);
}

/// # Theorem — BFS Multi-Component Completeness
///
/// **Statement**: `analyze_normals` correctly handles meshes with
/// multiple disconnected connected components by re-seeding BFS
/// for each unvisited component.  The total face count must equal
/// the sum across all components.
///
/// **Proof**: The outer `loop` in `analyze_normals` iterates until
/// `find_seed` returns `None`, which only happens when every
/// non-degenerate face has been assigned an orientation.  Each
/// iteration seeds and floods one component.
#[test]
fn analyze_normals_two_disjoint_cubes() {
    // Two separate cubes — both outward-wound.
    let cube1 = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();
    let cube2 = Cube {
        origin: Point3r::new(10.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();

    // Merge into one mesh.
    let mut combined = IndexedMesh::with_cell_size(1e-4);
    for fi in 0..cube1.face_count() {
        let fid = crate::domain::core::index::FaceId::from_usize(fi);
        let face = cube1.faces.get(fid);
        let a = combined.add_vertex_pos(*cube1.vertices.position(face.vertices[0]));
        let b = combined.add_vertex_pos(*cube1.vertices.position(face.vertices[1]));
        let c = combined.add_vertex_pos(*cube1.vertices.position(face.vertices[2]));
        combined.add_face(a, b, c);
    }
    for fi in 0..cube2.face_count() {
        let fid = crate::domain::core::index::FaceId::from_usize(fi);
        let face = cube2.faces.get(fid);
        let a = combined.add_vertex_pos(*cube2.vertices.position(face.vertices[0]));
        let b = combined.add_vertex_pos(*cube2.vertices.position(face.vertices[1]));
        let c = combined.add_vertex_pos(*cube2.vertices.position(face.vertices[2]));
        combined.add_face(a, b, c);
    }

    let r = analyze_normals(&combined);
    assert_eq!(
        r.total_faces(),
        cube1.face_count() + cube2.face_count(),
        "total faces must cover both components"
    );
    assert_eq!(
        r.inward_faces, 0,
        "two correctly-wound cubes must have zero inward faces"
    );
    assert!(r.all_outward());
}
