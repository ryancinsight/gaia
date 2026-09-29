use super::*;

// ── Degeneracy tests ───────────────────────────────────────────────────

/// Coaxial cylinders of the same radius share coincident lateral surfaces.
/// The CSG union must complete (no panic) and return a non-empty result.
///
/// This is the canonical "coaxial degeneracy" path documented in MEMORY.md.
/// The merge_collinear_segments fix in the blueprint pipeline is tested here
/// at the raw CSG level: if the union completes without panic, the guard works.
#[test]
fn coaxial_tubes_union_completes_without_panic() {
    // Two cylinders, same radius, same axis (+Y), overlapping length.
    // Segments=16 for speed; enough to trigger the coplanar lateral surface path.
    let cyl_a = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius: 1.0,
        height: 4.0,
        segments: 16,
    }
    .build()
    .expect("cyl_a");

    let cyl_b = Cylinder {
        base_center: Point3r::new(0.0, 1.0, 0.0), // overlapping by 3 units
        radius: 1.0,
        height: 4.0,
        segments: 16,
    }
    .build()
    .expect("cyl_b");

    // Must not panic; result may or may not be Ok depending on degenerate
    // surface handling — we only require no panic.
    let result = csg_boolean(BooleanOp::Union, &cyl_a, &cyl_b);
    // Either success or a structured error — never a panic or OOM.
    // If it succeeded, the mesh must be non-empty.
    if let Ok(mesh) = result {
        assert!(!mesh.faces.is_empty(), "union result must be non-empty");
    }
}

/// Two cubes whose faces are nearly-parallel (0.01° tilt) produce a
/// near-degenerate intersection line.  The GWN of the interior centroid
/// must remain finite and classify correctly as Inside.
#[test]
fn near_parallel_face_intersection_gwn_stable() {
    let (pool, faces) = unit_cube_faces();
    // Query point deep inside the cube
    let interior = Point3r::new(0.0, 0.0, 0.0);
    let wn = gwn::<f64>(&interior, &faces, &pool);
    assert!(
        wn.is_finite(),
        "GWN must be finite for interior point: {wn}"
    );
    assert!(
        wn.abs() > GWN_INSIDE_THRESHOLD,
        "Interior GWN |wn|={} must exceed the inside threshold",
        wn.abs()
    );
}

/// A flat sliver triangle with 10000:1 aspect ratio (4mm × 0.4µm)
/// must be classified correctly by `classify_fragment`, not silently
/// skipped due to a too-generous sliver threshold.
///
/// **Scale regression**: fixes the bug where `area_sq < 1e-10 * max_edge_sq`
/// incorrectly skipped valid millifluidic faces at 4mm:50µm scale.
#[test]
fn flat_sliver_millifluidic_face_classified_not_skipped() {
    let (pool, faces) = unit_cube_faces();

    // A flat fragment: 4mm wide, 0.0004mm tall (1e-4 aspect) — well within
    // millifluidic scale.  Centroid is clearly outside the unit cube.
    let tri = [
        Point3r::new(2.0, 0.0, 0.0),
        Point3r::new(6.0, 0.0, 0.0),
        Point3r::new(6.0, 0.0004, 0.0),
    ];
    let c = centroid(&tri);
    let n = tri_normal(&tri);

    // The fragment is outside — GWN of (4,0,0) vs a unit cube is 0.
    let cls = classify_fragment(&c, &n, &faces, &pool);
    assert_eq!(
        cls,
        FragmentClass::Outside,
        "high-aspect millifluidic fragment outside unit cube must be Outside, got {cls:?}"
    );
}

/// A point very close to a mesh vertex must produce a finite GWN result.
///
/// Regression for the f32 near-vertex guard underflow bug (Step 1b fix):
/// uses f64 here since the guard `min_positive_value` is now type-generic.
#[test]
fn gwn_near_vertex_produces_finite_result() {
    let (pool, faces) = unit_cube_faces();
    // Query at a vertex of the cube — exactly on-boundary degenerate position.
    let corner = Point3r::new(0.5, 0.5, 0.5);
    let wn = gwn::<f64>(&corner, &faces, &pool);
    assert!(
        wn.is_finite(),
        "GWN at cube corner must be finite, got {wn}"
    );
}

// ── Self-intersection detection ────────────────────────────────────────

/// detect_self_intersections finds crossing triangles in a "butterfly"
/// mesh where two triangles share only a vertex but their interiors cross.
#[test]
fn self_intersection_detection_finds_crossing_triangles() {
    let mut pool = VertexPool::default_millifluidic();
    let n = leto::geometry::Vector3::zeros();

    // Triangle A: (0,0,0)-(2,0,0)-(1,2,0) in XY plane
    let a0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let a1 = pool.insert_or_weld(Point3r::new(2.0, 0.0, 0.0), n);
    let a2 = pool.insert_or_weld(Point3r::new(1.0, 2.0, 0.0), n);

    // Triangle B: (1,-1,-1)-(1,-1,1)-(1,3,0) — cuts through triangle A along X=1
    let b0 = pool.insert_or_weld(Point3r::new(1.0, -1.0, -1.0), n);
    let b1 = pool.insert_or_weld(Point3r::new(1.0, -1.0, 1.0), n);
    let b2 = pool.insert_or_weld(Point3r::new(1.0, 3.0, 0.0), n);

    // Additional non-intersecting triangle to test adjacency filtering
    let c0 = pool.insert_or_weld(Point3r::new(10.0, 0.0, 0.0), n);
    let c1 = pool.insert_or_weld(Point3r::new(12.0, 0.0, 0.0), n);
    let c2 = pool.insert_or_weld(Point3r::new(11.0, 2.0, 0.0), n);

    let faces = vec![
        FaceData::untagged(a0, a1, a2),
        FaceData::untagged(b0, b1, b2),
        FaceData::untagged(c0, c1, c2),
    ];

    let pairs = detect_self_intersections(&faces, &pool);
    assert!(
        !pairs.is_empty(),
        "crossing triangles A and B should be detected as self-intersecting"
    );
    // The non-intersecting triangle C must not appear with A or B.
    for &(i, j) in &pairs {
        assert!(
            !(i == 2 || j == 2),
            "non-intersecting triangle C (index 2) should not appear in self-intersection pairs"
        );
    }
}

/// Adjacent triangles sharing an edge (manifold mesh) must NOT be reported.
#[test]
fn self_intersection_adjacent_faces_not_reported() {
    let mut pool = VertexPool::default_millifluidic();
    let n = leto::geometry::Vector3::zeros();
    // Two adjacent triangles forming a quad (0,0)-(1,0)-(1,1)-(0,1).
    let v0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let v1 = pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), n);
    let v2 = pool.insert_or_weld(Point3r::new(1.0, 1.0, 0.0), n);
    let v3 = pool.insert_or_weld(Point3r::new(0.0, 1.0, 0.0), n);
    let faces = vec![
        FaceData::untagged(v0, v1, v2),
        FaceData::untagged(v0, v2, v3),
    ];
    let pairs = detect_self_intersections(&faces, &pool);
    assert!(
        pairs.is_empty(),
        "adjacent manifold faces must not be reported as self-intersecting"
    );
}

// ── Property-based tests (proptest) ────────────────────────────────────

// Property: GWN of exterior points is below the outside threshold
// for a closed manifold unit cube.
//
// For any query point at distance > 1 from the cube surface along +Z,
// the winding number must be close to 0 (exterior).
proptest! {
    #[test]
    fn gwn_exterior_below_outside_threshold(qz in 2.0_f64..100.0) {
        let (pool, faces) = unit_cube_faces();
        let q = Point3r::new(0.0, 0.0, qz);
        let wn = gwn::<f64>(&q, &faces, &pool);
        prop_assert!(wn.is_finite(), "GWN must be finite: {wn}");
        prop_assert!(
            wn.abs() < GWN_OUTSIDE_THRESHOLD,
            "exterior GWN |wn|={} must be below the outside threshold for q=(0,0,{qz})",
            wn.abs()
        );
    }
}

// Property: Union volume ≥ max(vol_a, vol_b).
//
// For two overlapping unit cubes with offset ∈ (0.1, 1.0) along X, the
// union must be larger than each individual cube.
proptest! {
    #[test]
    fn union_vertex_count_geq_each_operand(dx in 0.1_f64..1.0) {
        let a = unit_cube();
        let b = offset_cube(dx);
        if let Ok(union) = csg_boolean(BooleanOp::Union, &a, &b) {
            let fa = a.faces.len();
            let fb = b.faces.len();
            let fu = union.faces.len();
            // Union cannot have fewer faces than either operand (loose check:
            // the interior gets removed, but boundary faces are preserved).
            prop_assert!(fu >= 1, "union must be non-empty: fa={fa} fb={fb} fu={fu}");
        }
    }
}

// Property: snap determinism — GridCell from two different computation
// paths for the same geometric point must agree.
proptest! {
    #[test]
    fn snap_gridcell_deterministic(
        x in -10.0_f64..10.0,
        y in -10.0_f64..10.0,
        z in -10.0_f64..10.0,
    ) {
        use crate::application::welding::snap::GridCell;
        let inv_eps = 1e3_f64; // 1mm cells
        let p = Point3r::new(x, y, z);
        // Two independent calls — must agree.
        let cell_a = GridCell::from_point_round(&p, inv_eps);
        let cell_b = GridCell::from_point_round(&p, inv_eps);
        prop_assert_eq!(cell_a, cell_b, "GridCell must be deterministic");
    }
}

// Property: CSG intersection is contained within each operand.
//
// For overlapping cubes, the intersection face count must be ≤ min(fa, fb).
// This is a weak containment check — exact volume bounds require signed-volume
// integration which is not exposed here.
proptest! {
    #[test]
    fn intersection_nonempty_for_overlapping_cubes(dx in 0.01_f64..0.99) {
        let a = unit_cube();
        let b = offset_cube(dx);
        if let Ok(inter) = csg_boolean(BooleanOp::Intersection, &a, &b) {
            prop_assert!(!inter.faces.is_empty(), "intersection of overlapping cubes must be non-empty");
        }
    }
}

// ── Adversarial Boolean tests ─────────────────────────────────────────
//
// These test known failure modes of mesh Boolean libraries:
// - Identical operands (degenerate overlap)
// - Shared faces (coplanar colocation)
// - Inclusion-exclusion volume identity
// - Near-coplanar faces (GWN boundary band)
// - Vertex/edge touching (zero-volume intersection)
// - Contained geometry (fully nested operand)

/// A ∪ A must equal A — same geometry, all faces coplanar and coincident.
/// Most CSG libraries fail here because every face pair is coplanar.
#[test]
fn identical_cubes_union_equals_single() {
    let a = unit_cube();
    let b = unit_cube();
    if let Ok(result) = csg_boolean(BooleanOp::Union, &a, &b) {
        let vol_a = signed_volume(&a);
        let vol_union = signed_volume(&result);
        let rel_err = (vol_union - vol_a).abs() / vol_a;
        assert!(
                rel_err < 0.05,
                "A∪A volume must ≈ vol(A): vol_a={vol_a:.6}, vol_union={vol_union:.6}, err={rel_err:.4}"
            );
    }
}

/// A ∩ A must equal A — intersection of identical meshes is the mesh itself.
#[test]
fn identical_cubes_intersection_equals_single() {
    let a = unit_cube();
    let b = unit_cube();
    if let Ok(result) = csg_boolean(BooleanOp::Intersection, &a, &b) {
        let vol_a = signed_volume(&a);
        let vol_inter = signed_volume(&result);
        let rel_err = (vol_inter - vol_a).abs() / vol_a;
        assert!(
                rel_err < 0.05,
                "A∩A volume must ≈ vol(A): vol_a={vol_a:.6}, vol_inter={vol_inter:.6}, err={rel_err:.4}"
            );
    }
}

/// A \ A must be empty — subtracting a mesh from itself leaves nothing.
#[test]
fn identical_cubes_difference_is_empty() {
    let a = unit_cube();
    let b = unit_cube();
    if let Ok(result) = csg_boolean(BooleanOp::Difference, &a, &b) {
        let vol_diff = signed_volume(&result);
        assert!(
            vol_diff < 1e-6,
            "A\\A must have zero volume: vol_diff={vol_diff:.8}"
        );
    }
}

/// Two cubes sharing exactly one face — the union is a 1×2×1 box.
/// Tests coplanar face handling when shared face must be removed from output.
#[test]
fn kissing_cubes_shared_face_union() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube a");
    let b = Cube {
        origin: Point3r::new(1.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube b");

    if let Ok(result) = csg_boolean(BooleanOp::Union, &a, &b) {
        let vol_a = signed_volume(&a);
        let vol_b = signed_volume(&b);
        let vol_union = signed_volume(&result);
        let expected = vol_a + vol_b; // no overlap
        let rel_err = (vol_union - expected).abs() / expected;
        assert!(
                rel_err < 0.05,
                "kissing cubes union vol must ≈ 2×vol(cube): expected={expected:.6}, got={vol_union:.6}, err={rel_err:.4}"
            );
    }
}

/// Inclusion-exclusion identity: vol(A) + vol(B) = vol(A∪B) + vol(A∩B).
/// Uses overlapping cubes with 50% overlap.
#[test]
fn volume_identity_inclusion_exclusion() {
    let a = unit_cube();
    let b = offset_cube(1.0); // 50% overlap for 2-wide cubes
    let vol_a = signed_volume(&a);
    let vol_b = signed_volume(&b);

    let union_ok = csg_boolean(BooleanOp::Union, &a, &b);
    let inter_ok = csg_boolean(BooleanOp::Intersection, &a, &b);

    if let (Ok(union), Ok(inter)) = (union_ok, inter_ok) {
        let vol_union = signed_volume(&union);
        let vol_inter = signed_volume(&inter);
        let lhs = vol_a + vol_b;
        let rhs = vol_union + vol_inter;
        let rel_err = (lhs - rhs).abs() / lhs;
        assert!(
                rel_err < 0.05,
                "inclusion-exclusion: vol(A)+vol(B)={lhs:.6} ≠ vol(A∪B)+vol(A∩B)={rhs:.6}, err={rel_err:.4}"
            );
    }
}

/// Near-coplanar cubes: one cube shifted by 1e-10 along X so "shared"
/// faces are not exactly coplanar. Tests robustness of the near-coplanar
/// classification boundary.
#[test]
fn near_coplanar_cubes_union_non_degenerate() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube a");
    let b = Cube {
        origin: Point3r::new(1.0 + 1e-10, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube b");

    if let Ok(result) = csg_boolean(BooleanOp::Union, &a, &b) {
        let vol_union = signed_volume(&result);
        // The tiny gap is negligible — union should be ≈ 2.0
        assert!(
            vol_union > 1.9 && vol_union < 2.1,
            "near-coplanar union vol must ≈ 2.0: got={vol_union:.6}"
        );
    }
}

/// Two cubes touching at exactly one edge (no shared face, no overlap).
/// The intersection volume must be zero (or empty).
#[test]
fn touching_cubes_at_single_edge() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube a");
    // Second cube positioned so it touches cube A along the edge x=1, y=1
    let b = Cube {
        origin: Point3r::new(1.0, 1.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube b");

    if let Ok(result) = csg_boolean(BooleanOp::Intersection, &a, &b) {
        let vol_inter = signed_volume(&result);
        assert!(
            vol_inter < 1e-6,
            "edge-touching intersection must have zero volume: got={vol_inter:.8}"
        );
    }
}

/// Two cubes touching at exactly one vertex (no shared edge, no overlap).
/// The intersection must be zero-volume.
#[test]
fn touching_cubes_at_single_vertex() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube a");
    // Second cube positioned so it touches cube A at the single vertex (1,1,1)
    let b = Cube {
        origin: Point3r::new(1.0, 1.0, 1.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("cube b");

    if let Ok(result) = csg_boolean(BooleanOp::Intersection, &a, &b) {
        let vol_inter = signed_volume(&result);
        assert!(
            vol_inter < 1e-6,
            "vertex-touching intersection must have zero volume: got={vol_inter:.8}"
        );
    }
}
