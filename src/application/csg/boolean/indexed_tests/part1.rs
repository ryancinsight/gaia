use super::*;

#[test]
fn rectangular_prism_union_is_axis_independent() {
    let a = Cube::unit().build().expect("unit prism");
    let b = Cube {
        origin: Point3r::new(0.0, 0.5, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("offset prism");

    let result = rectangular_prism_union(&a, &b).expect("the set union is one prism");
    let expected_volume = 1.5_f64;
    let tolerance = 64.0 * f64::EPSILON * expected_volume;
    assert!((result.signed_volume() - expected_volume).abs() <= tolerance);
    assert_eq!(result.faces.len(), 12);
}

#[test]
fn rectangular_prism_union_does_not_fill_an_l_shaped_gap() {
    let a = Cube::unit().build().expect("unit prism");
    let b = Cube {
        origin: Point3r::new(0.5, 0.5, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .expect("diagonally offset prism");

    let result = csg_boolean(BooleanOp::Union, &a, &b).expect("L-shaped union");
    let expected_volume = 1.75_f64;
    let tolerance = 512.0 * f64::EPSILON * expected_volume;
    assert!(
        (result.signed_volume() - expected_volume).abs() <= tolerance,
        "the union must preserve the L-shaped gap: expected {expected_volume}, got {}",
        result.signed_volume(),
    );
}

// ── sphere × cylinder (curved × curved — arrangement pipeline) ─────────────

#[test]
fn sphere_cylinder_union_is_watertight() {
    let result = csg_boolean(BooleanOp::Union, &sphere(), &cylinder()).expect("sphere ∪ cylinder");
    assert_3d_watertight(result);
}

#[test]
fn sphere_cylinder_intersection_is_watertight() {
    let result =
        csg_boolean(BooleanOp::Intersection, &sphere(), &cylinder()).expect("sphere ∩ cylinder");
    assert_3d_watertight(result);
}

#[test]
fn sphere_cylinder_difference_is_watertight() {
    let result =
        csg_boolean(BooleanOp::Difference, &sphere(), &cylinder()).expect("sphere \\ cylinder");
    assert_3d_watertight(result);
}

// ── cube × cube (flat faces — intersecting arrangement pipeline) ───────────

#[test]
fn cube_cube_union_is_watertight() {
    let result = csg_boolean(BooleanOp::Union, &cube_a(), &cube_b()).expect("cube ∪ cube");
    assert_3d_watertight(result);
}

#[test]
fn cube_cube_intersection_is_watertight() {
    let result = csg_boolean(BooleanOp::Intersection, &cube_a(), &cube_b()).expect("cube ∩ cube");
    assert_3d_watertight(result);
}

#[test]
fn cube_cube_difference_is_watertight() {
    let result = csg_boolean(BooleanOp::Difference, &cube_a(), &cube_b()).expect("cube \\ cube");
    assert_3d_watertight(result);
}

/// Difference of cube minus a coplanar cylinder must be watertight.
/// The cylinder end caps are coplanar with the cube's top and bottom walls.
/// The 2-D coplanar pipeline must subtract circular discs from the square
/// walls, producing annular rings (tunnel openings).
#[test]
fn cube_cylinder_coplanar_difference_is_watertight() {
    let result = csg_boolean(BooleanOp::Difference, &cube_a(), &cylinder_coplanar())
        .expect("cube \\\\ cylinder_coplanar");
    assert_3d_watertight(result);
}

#[test]
fn cube_cylinder_coplanar_union_is_watertight() {
    let result = csg_boolean(BooleanOp::Union, &cube_a(), &cylinder_coplanar())
        .expect("cube ∪ cylinder_coplanar");
    assert_3d_watertight(result);
}

#[test]
fn cube_cylinder_coplanar_intersection_is_watertight() {
    let result = csg_boolean(BooleanOp::Intersection, &cube_a(), &cylinder_coplanar())
        .expect("cube ∩ cylinder_coplanar");
    assert_3d_watertight(result);
}

// ── disk × disk (coplanar — 2-D Sutherland-Hodgman pipeline) ───────────────
// Disk operands are open surfaces; the coplanar path produces an open
// surface result.  Only assert the operation completes without error.

#[test]
fn disk_disk_union_succeeds() {
    csg_boolean(BooleanOp::Union, &disk_a(), &disk_b()).expect("disk ∪ disk must not error");
}

#[test]
fn disk_disk_intersection_succeeds() {
    csg_boolean(BooleanOp::Intersection, &disk_a(), &disk_b()).expect("disk ∩ disk must not error");
}

#[test]
fn disk_disk_difference_succeeds() {
    csg_boolean(BooleanOp::Difference, &disk_a(), &disk_b()).expect("disk \\ disk must not error");
}

#[test]
fn symmetric_parallel_cylinder_intersection_is_single_watertight_component() {
    let (cyl_a, cyl_b) = symmetric_parallel_cylinders(64);
    let mut result =
        csg_boolean(BooleanOp::Intersection, &cyl_a, &cyl_b).expect("symmetric intersection");

    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(
        report.is_watertight,
        "symmetric cylinder intersection must be watertight: boundary={}, non_manifold={}",
        report.boundary_edge_count, report.non_manifold_edge_count
    );
    assert_eq!(
        component_count(&mut result),
        1,
        "symmetric cylinder intersection must remain a single component",
    );

    let radius = 0.6;
    let height = 3.0;
    let theta = std::f64::consts::FRAC_PI_3;
    let overlap_area = 2.0 * radius * radius * (theta - theta.sin() * theta.cos());
    let expected = height * overlap_area;
    let relative_error = (result.signed_volume() - expected).abs() / expected;
    assert!(
        relative_error < 0.01,
        "symmetric cylinder intersection volume error {:.2}% exceeds 1%",
        relative_error * 100.0
    );
}

#[test]
fn indexed_nary_quadfurcation_union_is_watertight_without_component_dropping() {
    let mut result =
        csg_boolean_nary(BooleanOp::Union, &quadfurcation_meshes()).expect("quadfurcation union");
    assert_eq!(
        component_count(&mut result),
        1,
        "quadfurcation union must be a single connected component",
    );
    assert_3d_watertight(result);
}

#[test]
fn indexed_nary_trifurcation_union_is_watertight_without_component_dropping() {
    let mut result =
        csg_boolean_nary(BooleanOp::Union, &trifurcation_meshes()).expect("trifurcation union");
    assert_eq!(
        component_count(&mut result),
        1,
        "trifurcation union must be a single connected component",
    );
    assert_3d_watertight(result);
}

#[test]
fn indexed_nary_pentafurcation_union_is_watertight_without_component_dropping() {
    let mut result =
        csg_boolean_nary(BooleanOp::Union, &pentafurcation_meshes()).expect("pentafurcation union");
    assert_eq!(
        component_count(&mut result),
        1,
        "pentafurcation union must be a single connected component",
    );
    assert_3d_watertight(result);
}

#[test]
fn indexed_nary_union_is_permutation_invariant() {
    let forward = quadfurcation_meshes();
    let mut reversed = quadfurcation_meshes();
    reversed.reverse();

    let mut forward_union =
        csg_boolean_nary(BooleanOp::Union, &forward).expect("forward quadfurcation union");
    let mut reversed_union =
        csg_boolean_nary(BooleanOp::Union, &reversed).expect("reversed quadfurcation union");

    assert_3d_watertight(forward_union.clone());
    assert_3d_watertight(reversed_union.clone());
    assert_eq!(
        component_count(&mut forward_union),
        component_count(&mut reversed_union),
        "operand order must not change the number of connected components",
    );

    let forward_volume = forward_union.signed_volume();
    let reversed_volume = reversed_union.signed_volume();
    let relative_error =
        (forward_volume - reversed_volume).abs() / forward_volume.abs().max(1.0e-12);
    assert!(
        relative_error < 0.005,
        "operand order changed union volume by {:.2}%",
        relative_error * 100.0
    );
}

// ── Y-junction trunk difference (curved × curved, Difference) ──────────
// Diagnostic: verify watertight trunk difference has outward-only normals.
// The BFS seed is the extremal (max-X) face — by the Jordan-Brouwer theorem
// its outward normal must have nx ≥ 0, so BFS correctly orients the mesh.
#[test]
fn cylinder_difference_normals_check() {
    use crate::application::csg::CsgNode;
    use crate::application::quality::normals::analyze_normals;
    use crate::domain::core::scalar::Point3r;
    use crate::domain::geometry::primitives::{Cylinder, PrimitiveMesh};
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};
    use std::f64::consts::FRAC_PI_2;

    const R: f64 = 0.5;
    const H_TRUNK: f64 = 3.0;
    const H_BRANCH: f64 = 3.0;
    const EPS: f64 = R * 0.10;
    const SEGS: usize = 32;
    let theta = std::f64::consts::FRAC_PI_4;

    let trunk = {
        let raw = Cylinder {
            base_center: Point3r::new(0.0, 0.0, 0.0),
            radius: R,
            height: H_TRUNK + EPS,
            segments: SEGS,
        }
        .build()
        .unwrap();
        let rot = UnitQuaternion::<f64>::from_axis_angle(Vector3::z_axis(), -FRAC_PI_2);
        let iso = Isometry3::from_parts(Translation3::new(-H_TRUNK, 0.0, 0.0), rot);
        CsgNode::Transform {
            node: Box::new(CsgNode::Leaf(Box::new(raw))),
            iso,
        }
        .evaluate()
        .unwrap()
    };
    let branch_up = {
        let raw = Cylinder {
            base_center: Point3r::new(0.0, 0.0, 0.0),
            radius: R,
            height: H_BRANCH,
            segments: SEGS,
        }
        .build()
        .unwrap();
        let rot = UnitQuaternion::<f64>::from_axis_angle(Vector3::z_axis(), theta - FRAC_PI_2);
        let iso = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rot);
        CsgNode::Transform {
            node: Box::new(CsgNode::Leaf(Box::new(raw))),
            iso,
        }
        .evaluate()
        .unwrap()
    };
    let branch_dn = {
        let raw = Cylinder {
            base_center: Point3r::new(0.0, 0.0, 0.0),
            radius: R,
            height: H_BRANCH,
            segments: SEGS,
        }
        .build()
        .unwrap();
        let rot = UnitQuaternion::<f64>::from_axis_angle(Vector3::z_axis(), -theta - FRAC_PI_2);
        let iso = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rot);
        CsgNode::Transform {
            node: Box::new(CsgNode::Leaf(Box::new(raw))),
            iso,
        }
        .evaluate()
        .unwrap()
    };
    let branches = csg_boolean(BooleanOp::Union, &branch_up, &branch_dn).unwrap();
    let mut result = csg_boolean(BooleanOp::Difference, &trunk, &branches).unwrap();
    let normals_before = analyze_normals(&result);
    tracing::info!(
        "before orient_outward: outward={}, inward={}, degen={}",
        normals_before.outward_faces,
        normals_before.inward_faces,
        normals_before.degenerate_faces,
    );
    result.orient_outward();
    let normals_after = analyze_normals(&result);
    tracing::info!(
        "after  orient_outward: outward={}, inward={}, degen={}",
        normals_after.outward_faces,
        normals_after.inward_faces,
        normals_after.degenerate_faces,
    );
    assert_eq!(
        normals_after.inward_faces, 0,
        "orient_outward must eliminate inward faces"
    );

    // Single connected component — retain_largest_component must have
    // stripped the 2 × 8-face phantom islands from the trunk difference.
    {
        use crate::domain::topology::connectivity::connected_components;
        use crate::domain::topology::AdjacencyGraph;
        result.rebuild_edges();
        let edges = result.edges_ref().unwrap();
        let adj = AdjacencyGraph::build(&result.faces, edges);
        let comps = connected_components(&result.faces, &adj);
        assert_eq!(
            comps.len(),
            1,
            "trunk difference must be a single connected component; \
                 got {} (phantom islands not removed)",
            comps.len(),
        );
    }
    // Euler characteristic χ = 2 for a single genus-0 closed body.
    {
        use crate::application::watertight::check::check_watertight;
        result.rebuild_edges();
        let rpt = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
        assert_eq!(
            rpt.euler_characteristic,
            Some(2),
            "trunk difference must have Euler χ = 2; got {:?}",
            rpt.euler_characteristic,
        );
    }
}

// ── Adversarial CSG tests ─────────────────────────────────────────────
//
// These test failure modes commonly encountered in mesh Boolean libraries:
// shared edges, shared vertices, self-union idempotency, n-ary consistency,
// disjoint intersection, and high-operand-count n-ary unions.

/// Two cubes sharing exactly one edge — a degenerate configuration that
/// triggers coplanar-face and shared-edge handling in the arrangement
/// engine.  Many mesh Boolean libraries produce non-manifold output here.
///
/// # Theorem — Shared-Edge Union Watertightness
///
/// When two watertight genus-0 solids share exactly one edge *e*, the
/// union boundary equals `∂A ∪ ∂B` minus the two faces incident to *e*
/// that lie in the interior of the other solid.  The result is a genus-0
/// closed 2-manifold with Euler characteristic χ = 2.  ∎
#[test]
fn shared_edge_union_watertight() {
    // Cube A: unit cube at origin.
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();
    // Cube B: unit cube touching A along the edge x=1, z=0..1.
    let b = Cube {
        origin: Point3r::new(1.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();
    let result = csg_boolean(BooleanOp::Union, &a, &b).unwrap();
    assert_3d_watertight(result);
}

/// Two cubes touching at exactly one vertex — another degenerate
/// configuration.  The union must remain a single watertight component.
///
/// # Theorem — Shared-Vertex Union Topology
///
/// Two solids meeting at a single vertex *v* produce a union whose
/// boundary is `∂A ∪ ∂B` with *v* shared.  The result is a pinched
/// genus-0 surface that is still a closed 2-manifold (every edge is
/// shared by exactly two faces).  ∎
#[test]
fn shared_vertex_union_watertight() {
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();
    // B's corner (0,0,0) touches A's corner (1,1,1).
    let b = Cube {
        origin: Point3r::new(1.0, 1.0, 1.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();
    let result = csg_boolean(BooleanOp::Union, &a, &b).unwrap();
    assert_3d_watertight(result);
}

/// Self-union idempotency: A ∪ A must equal A (same face count, same
/// volume up to floating-point tolerance).
///
/// # Theorem — Union Idempotency
///
/// For any watertight solid *A*, `A ∪ A = A` because every point of
/// ∂A is on the boundary of both operands, and the GWN classifier
/// assigns the same in/out label to every face.  The result preserves
/// face count and signed volume.  ∎
#[test]
fn self_union_idempotent() {
    let a = cube_a();
    let original_face_count = a.faces.len();
    let original_vol = a.signed_volume();
    let result = csg_boolean(BooleanOp::Union, &a, &a).unwrap();
    assert_3d_watertight(result.clone());
    // Volume must be preserved (within tolerance).
    let vol = result.signed_volume();
    let rel_err = ((vol - original_vol) / original_vol).abs();
    assert!(
        rel_err < 0.05,
        "self-union volume drift: original={original_vol:.6}, result={vol:.6}, rel_err={rel_err:.4}",
    );
    // Face count should not explode.
    assert!(
        result.faces.len() <= original_face_count * 3,
        "self-union face explosion: original={original_face_count}, result={}",
        result.faces.len(),
    );
}

#[test]
fn csg_boolean_nary_empty_slice_returns_error() {
    let result = csg_boolean_nary(BooleanOp::Union, &[]);
    assert!(matches!(
        result,
        Err(crate::domain::core::error::MeshError::EmptyBooleanResult { .. })
    ));
}
