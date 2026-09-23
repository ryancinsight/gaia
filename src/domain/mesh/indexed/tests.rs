//! Tests for mesh construction, component pruning, and orientation repair.

use super::*;

#[test]
fn empty_clone_preserves_custom_tolerance() {
    // Create a mesh with a very tight custom tolerance
    let mut mesh: IndexedMesh<f64> = IndexedMesh::with_cell_size(1e-8);
    mesh.add_vertex_pos(Point3::new(1.0, 1.0, 1.0));
    assert_eq!(mesh.vertex_count(), 1);

    // Clone it
    let mut clone = mesh.empty_clone();
    assert_eq!(clone.vertex_count(), 0);

    // Check behavior: two points 1e-6 apart should NOT be welded under 1e-8 tolerance.
    let v1 = clone.add_vertex_pos(Point3::new(0.0, 0.0, 0.0));
    let v2 = clone.add_vertex_pos(Point3::new(1e-6, 0.0, 0.0));
    assert_ne!(
        v1, v2,
        "Vertices should not weld under preserved 1e-8 tolerance"
    );
    assert_eq!(clone.vertex_count(), 2);
}

#[test]
fn retain_largest_component_preserves_tolerance() {
    let mut mesh: IndexedMesh<f64> = IndexedMesh::with_cell_size(1e-8);
    // Component 1 (Largest) - 4 faces
    let v0 = mesh.add_vertex_pos(Point3::new(0.0, 0.0, 0.0));
    let v1 = mesh.add_vertex_pos(Point3::new(1.0, 0.0, 0.0));
    let v2 = mesh.add_vertex_pos(Point3::new(0.0, 1.0, 0.0));
    let v3 = mesh.add_vertex_pos(Point3::new(0.0, 0.0, 1.0));
    mesh.add_face(v0, v1, v2);
    mesh.add_face(v0, v2, v3);
    mesh.add_face(v0, v3, v1);
    mesh.add_face(v1, v3, v2); // closed tet
    mesh.attributes
        .set("temperature", FaceId::from_usize(0), 12.5);
    mesh.mark_boundary(FaceId::from_usize(0), "kept");

    // Component 2 (Phantom island, 1 face)
    let v4 = mesh.add_vertex_pos(Point3::new(10.0, 0.0, 0.0));
    let v5 = mesh.add_vertex_pos(Point3::new(11.0, 0.0, 0.0));
    let v6 = mesh.add_vertex_pos(Point3::new(10.0, 1.0, 0.0));
    let phantom_face = mesh.add_face(v4, v5, v6);
    mesh.attributes.set("temperature", phantom_face, 99.5);
    mesh.mark_boundary(phantom_face, "discarded");

    // Run filter
    let discarded = mesh.retain_largest_component();
    assert_eq!(discarded, 1);
    assert_eq!(mesh.face_count(), 4);
    assert_eq!(
        mesh.attributes.get("temperature", FaceId::from_usize(0)),
        Some(12.5)
    );
    assert_eq!(mesh.boundary_label(FaceId::from_usize(0)), Some("kept"));
    assert_eq!(mesh.boundary_label(FaceId::from_usize(4)), None);

    // Add points 1e-6 apart; under the default 1e-4 tolerance they would weld.
    // Under the preserved 1e-8 tolerance they should remain distinct.
    let n1 = mesh.add_vertex_pos(Point3::new(20.0, 0.0, 0.0));
    let n2 = mesh.add_vertex_pos(Point3::new(20.0 + 1e-6, 0.0, 0.0));
    assert_ne!(
        n1, n2,
        "Tolerance must be preserved after retain_largest_component"
    );
}

#[test]
fn retain_largest_component_skips_reconstruction_when_all_components_are_kept() {
    let mut mesh: IndexedMesh<f32> = IndexedMesh::with_cell_size(1e-6);
    for offset in [0.0_f32, 10.0] {
        let v0 = mesh.add_vertex_pos(Point3::new(offset, 0.0, 0.0));
        let v1 = mesh.add_vertex_pos(Point3::new(offset + 1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex_pos(Point3::new(offset, 1.0, 0.0));
        let v3 = mesh.add_vertex_pos(Point3::new(offset, 0.0, 1.0));
        mesh.add_face(v0, v1, v2);
        mesh.add_face(v0, v2, v3);
        mesh.add_face(v0, v3, v1);
        mesh.add_face(v1, v3, v2);
    }
    mesh.mark_boundary(FaceId::from_usize(0), "first");

    let before_faces = mesh.face_count();
    let before_vertices = mesh.vertex_count();
    let discarded = mesh.retain_largest_component();

    assert_eq!(discarded, 0);
    assert_eq!(mesh.face_count(), before_faces);
    assert_eq!(mesh.vertex_count(), before_vertices);
    assert_eq!(mesh.boundary_label(FaceId::from_usize(0)), Some("first"));
}

/// `orient_outward` seeds every component from its own extremal face.
///
/// The seed rule is "the unvisited non-degenerate face with the maximum
/// centroid X", re-evaluated for each component.  The three cubes below are
/// added in the order `x = 10`, `x = 5`, `x = 20`, and each is wound
/// inward, so every component needs repair and the seed order
/// (`20`, `10`, `5`) is not the face-index order.  A cube is used rather
/// than a tetrahedron because a tetrahedron's three apex faces tie for the
/// maximum centroid X, which leaves the seed's sign heuristic free to pick
/// either winding; a cube's `+X` face is the unique maximum.
///
/// Each cube has volume 1, so three repaired components sum to 3.  A cursor
/// that fails to advance past a visited entry — or a single global ordering
/// — leaves one component inward, which reads 1 instead.
#[test]
fn orient_outward_seeds_each_component_from_its_own_extremum() {
    let mut mesh: IndexedMesh<f64> = IndexedMesh::with_cell_size(1.0e-6);

    for offset_x in [10.0_f64, 5.0, 20.0] {
        let corner = {
            let mut v =
                |dx: f64, dy: f64, dz: f64| mesh.add_vertex_pos(Point3::new(offset_x + dx, dy, dz));
            [
                v(-0.5, -0.5, -0.5),
                v(0.5, -0.5, -0.5),
                v(-0.5, 0.5, -0.5),
                v(0.5, 0.5, -0.5),
                v(-0.5, -0.5, 0.5),
                v(0.5, -0.5, 0.5),
                v(-0.5, 0.5, 0.5),
                v(0.5, 0.5, 0.5),
            ]
        };
        let [c000, c100, c010, c110, c001, c101, c011, c111] = corner;

        // The outward winding of a unit cube, with each triangle reversed,
        // so every component starts consistently inward.
        for (a, b, c) in [
            (c000, c110, c010),
            (c000, c100, c110),
            (c001, c111, c101),
            (c001, c011, c111),
            (c000, c011, c001),
            (c000, c010, c011),
            (c100, c111, c110),
            (c100, c101, c111),
            (c000, c101, c100),
            (c000, c001, c101),
            (c010, c111, c011),
            (c010, c110, c111),
        ] {
            mesh.add_face(a, b, c);
        }
    }

    mesh.orient_outward();

    let volume = crate::domain::geometry::measure::total_signed_volume(
        mesh.faces.iter_enumerated().map(|(_, face)| {
            (
                mesh.vertices.position(face.vertices[0]),
                mesh.vertices.position(face.vertices[1]),
                mesh.vertices.position(face.vertices[2]),
            )
        }),
    );
    assert!(
        (volume - 3.0).abs() < 1.0e-9,
        "three outward unit cubes must sum to 3; got {volume}, which means \
             a component was seeded from the wrong extremum"
    );
}

#[test]
fn mesh_builder_vertex_array_welds_duplicate_coordinates() {
    let mut builder = MeshBuilder::<f64>::new();
    let a = builder.vertex_array([0.0, 0.0, 0.0]);
    let b = builder.vertex_array([0.0, 0.0, 0.0]);
    let c = builder.vertex_xyz(1.0, 0.0, 0.0);
    let d = builder.vertex_xyz(0.0, 1.0, 0.0);
    builder.triangle(a, c, d);

    let mesh = builder.build();

    assert_eq!(a, b);
    assert_eq!(mesh.vertex_count(), 3);
    assert_eq!(mesh.face_count(), 1);
}

#[test]
fn mesh_builder_triangle_soup_arrays_adds_faces() {
    let mut builder = MeshBuilder::<f64>::new();
    builder.add_triangle_soup_arrays(&[
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ]);

    let mesh = builder.build();

    assert_eq!(mesh.vertex_count(), 4);
    assert_eq!(mesh.face_count(), 2);
}

// ── orient_outward adversarial tests ──────────────────────────────────

/// Build an outward-oriented closed tetrahedron.
///
/// Vertices: (1,0,0), (0,1,0), (0,0,1), (0,0,0).
/// Winding: each face normal points away from the centroid.
fn outward_tet() -> IndexedMesh<f64> {
    let mut m = IndexedMesh::with_cell_size(0.01);
    let v0 = m.add_vertex_pos(Point3::new(1.0, 0.0, 0.0));
    let v1 = m.add_vertex_pos(Point3::new(0.0, 1.0, 0.0));
    let v2 = m.add_vertex_pos(Point3::new(0.0, 0.0, 1.0));
    let v3 = m.add_vertex_pos(Point3::new(0.0, 0.0, 0.0));
    // CCW winding viewed from outside
    m.add_face(v0, v1, v2);
    m.add_face(v0, v3, v1);
    m.add_face(v0, v2, v3);
    m.add_face(v1, v3, v2);
    m
}

/// Build an inward-oriented closed tetrahedron (all faces reversed).
fn inward_tet() -> IndexedMesh<f64> {
    let mut m = IndexedMesh::with_cell_size(0.01);
    let v0 = m.add_vertex_pos(Point3::new(1.0, 0.0, 0.0));
    let v1 = m.add_vertex_pos(Point3::new(0.0, 1.0, 0.0));
    let v2 = m.add_vertex_pos(Point3::new(0.0, 0.0, 1.0));
    let v3 = m.add_vertex_pos(Point3::new(0.0, 0.0, 0.0));
    // CW winding (inward) — swap v1 ↔ v2 relative to outward_tet
    m.add_face(v0, v2, v1);
    m.add_face(v0, v1, v3);
    m.add_face(v0, v3, v2);
    m.add_face(v1, v2, v3);
    m
}

/// # Theorem — Signed-Volume Orientation Correction
///
/// **Statement**: For a closed, orientable triangulated manifold,
/// the divergence-theorem signed volume is positive iff all face
/// normals point outward.  `orient_outward` must correct a
/// fully-inward mesh to positive signed volume via the global
/// flip fallback.
///
/// **Proof**: The signed volume integral
/// $V = \frac{1}{6} \sum_f \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})$
/// changes sign under face reversal (swapping two vertices negates
/// the cross product).  `orient_outward`'s signed-volume check
/// detects $V < 0$ and flips every face, yielding $V > 0$.
#[test]
fn orient_outward_corrects_all_inward_tet() {
    let mut mesh = inward_tet();

    // Before: signed volume should be negative.
    let vol_before = crate::domain::geometry::measure::total_signed_volume(
        mesh.faces.iter_enumerated().map(|(_, f)| {
            (
                mesh.vertices.position(f.vertices[0]),
                mesh.vertices.position(f.vertices[1]),
                mesh.vertices.position(f.vertices[2]),
            )
        }),
    );
    assert!(
        vol_before < 0.0,
        "inward tet should have negative signed vol"
    );

    mesh.orient_outward();

    // After: signed volume should be positive.
    let vol_after = crate::domain::geometry::measure::total_signed_volume(
        mesh.faces.iter_enumerated().map(|(_, f)| {
            (
                mesh.vertices.position(f.vertices[0]),
                mesh.vertices.position(f.vertices[1]),
                mesh.vertices.position(f.vertices[2]),
            )
        }),
    );
    assert!(
        vol_after > 0.0,
        "orient_outward must produce positive signed volume, got {vol_after}"
    );
}

/// Already-outward mesh must remain unchanged.
#[test]
fn orient_outward_preserves_correct_winding() {
    let mut mesh = outward_tet();
    let vol_before = crate::domain::geometry::measure::total_signed_volume(
        mesh.faces.iter_enumerated().map(|(_, f)| {
            (
                mesh.vertices.position(f.vertices[0]),
                mesh.vertices.position(f.vertices[1]),
                mesh.vertices.position(f.vertices[2]),
            )
        }),
    );
    assert!(vol_before > 0.0, "outward tet must have positive vol");

    mesh.orient_outward();

    let vol_after = crate::domain::geometry::measure::total_signed_volume(
        mesh.faces.iter_enumerated().map(|(_, f)| {
            (
                mesh.vertices.position(f.vertices[0]),
                mesh.vertices.position(f.vertices[1]),
                mesh.vertices.position(f.vertices[2]),
            )
        }),
    );
    assert!(
        vol_after > 0.0,
        "orient_outward must not break already-outward mesh, got {vol_after}"
    );
}

/// # Theorem — BFS Disconnected-Component Completeness
///
/// **Statement**: The outer loop in `orient_outward` re-seeds BFS
/// for every connected component.  Two disjoint tetrahedra must
/// both be oriented outward, with total positive signed volume
/// equal to the sum of their individual volumes.
///
/// **Proof**: After the first component's BFS exhausts its connected
/// faces, the seed-search finds the next unvisited non-degenerate
/// face and starts a fresh BFS.  Inductive application shows all
/// components are covered.
#[test]
fn orient_outward_two_disjoint_tets() {
    let mut mesh = IndexedMesh::with_cell_size(0.01);

    // Component 1: tet at origin (inward winding)
    let a0 = mesh.add_vertex_pos(Point3::new(1.0, 0.0, 0.0));
    let a1 = mesh.add_vertex_pos(Point3::new(0.0, 1.0, 0.0));
    let a2 = mesh.add_vertex_pos(Point3::new(0.0, 0.0, 1.0));
    let a3 = mesh.add_vertex_pos(Point3::new(0.0, 0.0, 0.0));
    mesh.add_face(a0, a2, a1); // inward
    mesh.add_face(a0, a1, a3);
    mesh.add_face(a0, a3, a2);
    mesh.add_face(a1, a2, a3);

    // Component 2: tet at (10,0,0) (outward winding)
    let b0 = mesh.add_vertex_pos(Point3::new(11.0, 0.0, 0.0));
    let b1 = mesh.add_vertex_pos(Point3::new(10.0, 1.0, 0.0));
    let b2 = mesh.add_vertex_pos(Point3::new(10.0, 0.0, 1.0));
    let b3 = mesh.add_vertex_pos(Point3::new(10.0, 0.0, 0.0));
    mesh.add_face(b0, b1, b2);
    mesh.add_face(b0, b3, b1);
    mesh.add_face(b0, b2, b3);
    mesh.add_face(b1, b3, b2);

    assert_eq!(mesh.face_count(), 8, "should have 8 faces total");

    mesh.orient_outward();

    let vol = crate::domain::geometry::measure::total_signed_volume(
        mesh.faces.iter_enumerated().map(|(_, f)| {
            (
                mesh.vertices.position(f.vertices[0]),
                mesh.vertices.position(f.vertices[1]),
                mesh.vertices.position(f.vertices[2]),
            )
        }),
    );
    assert!(
        vol > 0.0,
        "two disjoint tets must both orient outward (positive vol), got {vol}"
    );
}

/// Empty mesh must not panic in orient_outward.
#[test]
fn orient_outward_empty_mesh_no_panic() {
    let mut mesh: IndexedMesh<f64> = IndexedMesh::new();
    mesh.orient_outward();
    assert_eq!(mesh.face_count(), 0);
}
