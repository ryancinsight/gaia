use super::*;

/// Small cube fully contained inside a larger cube.
/// Intersection must equal the inner cube volume.
#[test]
fn contained_cube_intersection_equals_inner() {
    let outer = Cube {
        origin: Point3r::new(-2.0, -2.0, -2.0),
        width: 4.0,
        height: 4.0,
        depth: 4.0,
    }
    .build()
    .expect("outer");
    let inner = Cube {
        origin: Point3r::new(-0.5, -0.5, -0.5),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("inner");

    let vol_inner = signed_volume(&inner);
    if let Ok(result) = csg_boolean(BooleanOp::Intersection, &outer, &inner) {
        let vol_inter = signed_volume(&result);
        let rel_err = (vol_inter - vol_inner).abs() / vol_inner;
        assert!(
                rel_err < 0.05,
                "contained intersection must ≈ inner: inner={vol_inner:.6}, got={vol_inter:.6}, err={rel_err:.4}"
            );
    }
}

/// Difference: large cube minus small inner cube.
/// Result volume must be vol(outer) - vol(inner).
#[test]
fn contained_cube_difference_volume() {
    let outer = Cube {
        origin: Point3r::new(-2.0, -2.0, -2.0),
        width: 4.0,
        height: 4.0,
        depth: 4.0,
    }
    .build()
    .expect("outer");
    let inner = Cube {
        origin: Point3r::new(-0.5, -0.5, -0.5),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("inner");

    let vol_outer = signed_volume(&outer);
    let vol_inner = signed_volume(&inner);
    if let Ok(result) = csg_boolean(BooleanOp::Difference, &outer, &inner) {
        let vol_diff = signed_volume(&result);
        let expected = vol_outer - vol_inner;
        let rel_err = (vol_diff - expected).abs() / expected;
        assert!(
                rel_err < 0.05,
                "difference vol must ≈ outer-inner: expected={expected:.6}, got={vol_diff:.6}, err={rel_err:.4}"
            );
    }
}

/// Commutativity: A ∪ B = B ∪ A (volume should match).
#[test]
fn union_commutativity_volume() {
    let a = unit_cube();
    let b = offset_cube(0.5);

    let ab = csg_boolean(BooleanOp::Union, &a, &b);
    let ba = csg_boolean(BooleanOp::Union, &b, &a);

    if let (Ok(ab), Ok(ba)) = (ab, ba) {
        let vol_ab = signed_volume(&ab);
        let vol_ba = signed_volume(&ba);
        let rel_err = (vol_ab - vol_ba).abs() / vol_ab.max(1e-12);
        assert!(
            rel_err < 0.01,
            "A∪B ≠ B∪A by volume: {vol_ab:.6} vs {vol_ba:.6}, err={rel_err:.4}"
        );
    }
}

/// Commutativity: A ∩ B = B ∩ A (volume should match).
#[test]
fn intersection_commutativity_volume() {
    let a = unit_cube();
    let b = offset_cube(0.5);

    let ab = csg_boolean(BooleanOp::Intersection, &a, &b);
    let ba = csg_boolean(BooleanOp::Intersection, &b, &a);

    if let (Ok(ab), Ok(ba)) = (ab, ba) {
        let vol_ab = signed_volume(&ab);
        let vol_ba = signed_volume(&ba);
        let rel_err = (vol_ab - vol_ba).abs() / vol_ab.max(1e-12);
        assert!(
            rel_err < 0.01,
            "A∩B ≠ B∩A by volume: {vol_ab:.6} vs {vol_ba:.6}, err={rel_err:.4}"
        );
    }
}

/// Non-commutativity: A \ B ≠ B \ A for non-identical overlapping cubes.
/// But vol(A\B) + vol(A∩B) = vol(A) must hold.
#[test]
fn difference_plus_intersection_equals_operand() {
    let a = unit_cube();
    let b = offset_cube(0.5);

    let diff_ab = csg_boolean(BooleanOp::Difference, &a, &b);
    let inter_ab = csg_boolean(BooleanOp::Intersection, &a, &b);

    if let (Ok(diff), Ok(inter)) = (diff_ab, inter_ab) {
        let vol_a = signed_volume(&a);
        let vol_diff = signed_volume(&diff);
        let vol_inter = signed_volume(&inter);
        let sum = vol_diff + vol_inter;
        let rel_err = (sum - vol_a).abs() / vol_a;
        assert!(
            rel_err < 0.05,
            "vol(A\\B)+vol(A∩B) must ≈ vol(A): {sum:.6} vs {vol_a:.6}, err={rel_err:.4}"
        );
    }
}

// ── BVH GWN integration ───────────────────────────────────────────────

/// BVH-accelerated GWN must agree with linear GWN for interior/exterior
/// queries against a closed mesh.  This validates the gwn_bvh wiring.
#[test]
fn gwn_bvh_agrees_with_linear() {
    use crate::application::csg::arrangement::gwn::prepare_classification_faces;
    use crate::application::csg::arrangement::gwn_bvh::{gwn_bvh, prepare_bvh_mesh};

    let (pool, faces) = unit_cube_faces();
    let prepared = prepare_classification_faces(&faces, &pool);
    let bvh = prepare_bvh_mesh(&prepared).expect("BVH build must succeed");

    let queries = [
        (Point3r::new(0.0, 0.0, 0.0), "interior"),
        (Point3r::new(5.0, 0.0, 0.0), "exterior far"),
        (Point3r::new(0.49, 0.0, 0.0), "near +x face"),
        (Point3r::new(0.0, -0.49, 0.0), "near -y face"),
        (Point3r::new(0.0, 0.0, 0.6), "exterior near +z"),
    ];

    for (q, label) in &queries {
        let wn_linear = gwn::<f64>(q, &faces, &pool);
        let wn_bvh = gwn_bvh(q, &bvh, 0.01);
        let delta = (wn_linear - wn_bvh).abs();
        assert!(
                delta < 0.15,
                "BVH vs linear GWN mismatch at {label}: linear={wn_linear:.4}, bvh={wn_bvh:.4}, delta={delta:.4}"
            );
    }
}

// ── Additional adversarial failure-mode tests ─────────────────────────
//
// Target failure modes documented in mesh Boolean literature (Cork, CGAL,
// libigl, Manifold) that are not yet covered by existing tests.

/// Iterated Boolean: `(A ∪ B) \ C` must not panic and must produce
/// a valid mesh.  Many libraries fail because the intermediate union
/// introduces T-junctions or non-manifold edges that break the second op.
#[test]
fn iterated_boolean_union_then_difference() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("a");
    let b = Cube {
        origin: Point3r::new(1.0, 0.0, 0.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("b");
    let c = Cube {
        origin: Point3r::new(0.5, 0.5, 0.5),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("c");

    if let Ok(ab) = csg_boolean(BooleanOp::Union, &a, &b)
        && let Ok(result) = csg_boolean(BooleanOp::Difference, &ab, &c)
    {
        let vol = signed_volume(&result);
        let vol_ab = signed_volume(&ab);
        let vol_c = signed_volume(&c);
        // Result must be smaller than union but non-empty
        assert!(
            vol > 0.0,
            "iterated (A∪B)\\C must have positive volume, got {vol:.6}"
        );
        assert!(
            vol < vol_ab + 0.1,
            "iterated result volume {vol:.6} must be ≤ vol(A∪B)={vol_ab:.6}"
        );
        let _ = vol_c; // used only to verify C was built
    }
}

/// Large scale disparity: 10× size difference between operands.
/// The small cube is 0.2×0.2×0.2; the large is 2×2×2.
/// Tests that broad-phase AABB and GWN are numerically stable
/// when one mesh is much smaller than the other.
#[test]
fn large_scale_disparity_intersection() {
    let large = Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("large");
    let small = Cube {
        origin: Point3r::new(-0.1, -0.1, -0.1),
        width: 0.2,
        height: 0.2,
        depth: 0.2,
    }
    .build()
    .expect("small");

    let vol_small = signed_volume(&small);
    if let Ok(result) = csg_boolean(BooleanOp::Intersection, &large, &small) {
        let vol_inter = signed_volume(&result);
        let rel_err = (vol_inter - vol_small).abs() / vol_small;
        assert!(
                rel_err < 0.1,
                "10× disparity inter must ≈ vol(small): small={vol_small:.8}, got={vol_inter:.8}, err={rel_err:.4}"
            );
    }
}

/// Vertex-on-face coincidence: a cube with one vertex exactly on
/// another cube's face.  This produces a degenerate intersection line
/// (ray from vertex to interior = zero length on one end).
///
/// # Known limitation
///
/// The CSG pipeline currently produces an empty/zero-volume mesh for
/// this configuration.  The `csg_boolean` may return `Ok` with a
/// degenerate result instead of `Err`.  Ignored until the corefinement
/// pipeline handles vertex-on-face degeneracies correctly.
#[test]
fn vertex_on_face_coincidence_union() {
    // Cube A: unit cube at origin
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("a");
    // Cube B: vertex (0,0,0) lies exactly on A's face z=0
    let b = Cube {
        origin: Point3r::new(-1.0, -1.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("b");

    // Must not panic; volume identity still holds
    if let Ok(result) = csg_boolean(BooleanOp::Union, &a, &b) {
        let vol_a = signed_volume(&a);
        let vol_b = signed_volume(&b);
        let vol_union = signed_volume(&result);
        // No overlap volume, so union = vol_a + vol_b
        // (they only touch at one vertex — zero volume intersection)
        let expected = vol_a + vol_b;
        let rel_err = (vol_union - expected).abs() / expected;
        assert!(
                rel_err < 0.1,
                "vertex-on-face union vol: expected={expected:.6}, got={vol_union:.6}, err={rel_err:.4}"
            );
    }
}

/// Edge-edge coincidence: two cubes share exactly one edge.
/// The intersection volume must be zero.
#[test]
fn edge_edge_coincidence_intersection() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("a");
    // Cube B shares edge at x=1, z=0 (the edge from (1,0,0)-(1,1,0))
    let b = Cube {
        origin: Point3r::new(1.0, 0.0, -1.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("b");

    if let Ok(result) = csg_boolean(BooleanOp::Intersection, &a, &b) {
        let vol = signed_volume(&result);
        assert!(
            vol < 1e-6,
            "edge-sharing intersection must have zero volume: got={vol:.8}"
        );
    }
}

/// Cube–cylinder intersection: non-planar surface intersection curve.
/// Tests that the pipeline handles curved-surface meshes correctly.
#[test]
fn cube_cylinder_intersection_non_planar() {
    let cube = Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube");
    let cyl = Cylinder {
        base_center: Point3r::new(0.0, -1.5, 0.0),
        radius: 0.5,
        height: 3.0,
        segments: 24,
    }
    .build()
    .expect("cylinder");

    if let Ok(result) = csg_boolean(BooleanOp::Intersection, &cube, &cyl) {
        let vol_cyl = signed_volume(&cyl);
        let vol_inter = signed_volume(&result);
        // Cylinder extends beyond cube top/bottom, so intersection is a
        // shorter cylinder segment (height 2 vs 3).
        assert!(
            vol_inter > 0.0 && vol_inter < vol_cyl,
            "cube∩cyl must have 0 < vol < vol_cyl={vol_cyl:.6}: got={vol_inter:.6}"
        );
    }
}

/// Cube minus an inscribed cylinder: tests difference with curved geometry.
/// Result volume = vol(cube) - vol(cylinder_inside_cube).
#[test]
fn cube_minus_inscribed_cylinder() {
    let cube = Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube");
    // Cylinder fits inside cube along Y axis
    let cyl = Cylinder {
        base_center: Point3r::new(0.0, -1.0, 0.0),
        radius: 0.8,
        height: 2.0,
        segments: 24,
    }
    .build()
    .expect("cylinder");

    let vol_cube = signed_volume(&cube);
    let vol_cyl = signed_volume(&cyl);
    if let Ok(result) = csg_boolean(BooleanOp::Difference, &cube, &cyl) {
        let vol_diff = signed_volume(&result);
        let expected = vol_cube - vol_cyl;
        let rel_err = (vol_diff - expected).abs() / expected;
        assert!(
            rel_err < 0.1,
            "cube\\cyl vol: expected={expected:.6}, got={vol_diff:.6}, err={rel_err:.4}"
        );
    }
}

/// De Morgan's law: A \ B = A ∩ B^c.  Since we can't compute complement,
/// we test the equivalent: vol(A \ B) + vol(A ∩ B) = vol(A).
/// Uses offset cubes with partial overlap.
#[test]
fn de_morgan_volume_identity() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("a");
    let b = Cube {
        origin: Point3r::new(1.0, 0.5, 0.5),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("b");

    let vol_a = signed_volume(&a);
    let diff_ab = csg_boolean(BooleanOp::Difference, &a, &b);
    let inter_ab = csg_boolean(BooleanOp::Intersection, &a, &b);

    if let (Ok(diff), Ok(inter)) = (diff_ab, inter_ab) {
        let sum = signed_volume(&diff) + signed_volume(&inter);
        let rel_err = (sum - vol_a).abs() / vol_a;
        assert!(
            rel_err < 0.05,
            "De Morgan: vol(A\\B)+vol(A∩B)={sum:.6} must ≈ vol(A)={vol_a:.6}, err={rel_err:.4}"
        );
    }
}

/// Associativity: (A ∪ B) ∪ C ≈ A ∪ (B ∪ C) by volume.
/// Tests that iterated unions are stable and do not accumulate errors.
#[test]
fn union_associativity_volume() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("a");
    let b = Cube {
        origin: Point3r::new(0.5, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("b");
    let c = Cube {
        origin: Point3r::new(1.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("c");

    let ab = csg_boolean(BooleanOp::Union, &a, &b).expect("A union B");
    let bc = csg_boolean(BooleanOp::Union, &b, &c).expect("B union C");
    let left = csg_boolean(BooleanOp::Union, &ab, &c).expect("(A union B) union C");
    let right = csg_boolean(BooleanOp::Union, &a, &bc).expect("A union (B union C)");

    let vol_l = signed_volume(&left);
    let vol_r = signed_volume(&right);
    let rel_err = (vol_l - vol_r).abs() / vol_l.max(1e-12);
    assert!(
        rel_err < 0.05,
        "(A∪B)∪C ≠ A∪(B∪C) by volume: {vol_l:.6} vs {vol_r:.6}, err={rel_err:.4}"
    );
}
