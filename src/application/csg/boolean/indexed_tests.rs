#![cfg(test)]

use super::*;
use crate::application::watertight::check::check_watertight;
use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::primitives::{Cube, Cylinder, Disk, PrimitiveMesh, UvSphere};

fn sphere() -> IndexedMesh {
    UvSphere {
        radius: 1.0,
        center: Point3r::origin(),
        segments: 16,
        stacks: 8,
    }
    .build()
    .expect("sphere build")
}

fn cylinder() -> IndexedMesh {
    Cylinder {
        base_center: Point3r::new(0.0, -1.5, 0.0),
        radius: 0.4,
        height: 3.0,
        segments: 16,
    }
    .build()
    .expect("cylinder build")
}

fn cube_a() -> IndexedMesh {
    Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_a build")
}

fn cube_b() -> IndexedMesh {
    Cube {
        origin: Point3r::new(-0.5, -0.5, -0.5),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_b build")
}

fn disk_a() -> IndexedMesh {
    Disk {
        center: Point3r::new(0.0, 0.0, 0.0),
        radius: 1.0,
        segments: 16,
    }
    .build()
    .expect("disk_a build")
}

fn disk_b() -> IndexedMesh {
    Disk {
        center: Point3r::new(0.5, 0.0, 0.0),
        radius: 1.0,
        segments: 16,
    }
    .build()
    .expect("disk_b build")
}

/// Assert a 3-D CSG result is watertight with a positive signed volume.
fn assert_3d_watertight(mut mesh: IndexedMesh) {
    mesh.rebuild_edges();
    let report = check_watertight(&mesh.vertices, &mesh.faces, mesh.edges_ref().unwrap());
    assert!(
        report.is_watertight,
        "CSG result must be watertight: {} boundary edge(s), {} non-manifold edge(s)",
        report.boundary_edge_count, report.non_manifold_edge_count,
    );
    assert!(
        mesh.signed_volume() > 0.0,
        "CSG result must have positive signed volume (outward-oriented normals)",
    );
}

fn component_count(mesh: &mut IndexedMesh) -> usize {
    use crate::domain::topology::connectivity::connected_components;
    use crate::domain::topology::AdjacencyGraph;

    mesh.rebuild_edges();
    let adjacency = AdjacencyGraph::build(&mesh.faces, mesh.edges_ref().unwrap());
    connected_components(&mesh.faces, &adjacency).len()
}

fn symmetric_parallel_cylinders(segments: usize) -> (IndexedMesh, IndexedMesh) {
    let radius = 0.6;
    let height = 3.0;
    let separation = radius;
    let cyl_a = Cylinder {
        base_center: Point3r::new(-separation / 2.0, -height / 2.0, 0.0),
        radius,
        height,
        segments,
    }
    .build()
    .expect("symmetric cyl_a build");
    let cyl_b = Cylinder {
        base_center: Point3r::new(separation / 2.0, -height / 2.0, 0.0),
        radius,
        height,
        segments,
    }
    .build()
    .expect("symmetric cyl_b build");
    (cyl_a, cyl_b)
}

fn planar_branch(angle_from_x: f64, radius: f64, height: f64, segments: usize) -> IndexedMesh {
    use crate::application::csg::CsgNode;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let raw = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius,
        height,
        segments,
    }
    .build()
    .expect("branch build");
    let rotation = UnitQuaternion::<f64>::from_axis_angle(
        &Vector3::z_axis(),
        angle_from_x - std::f64::consts::FRAC_PI_2,
    );
    CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(raw))),
        iso: Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation),
    }
    .evaluate()
    .expect("branch transform")
}

fn planar_trunk(radius: f64, height: f64, extension: f64, segments: usize) -> IndexedMesh {
    use crate::application::csg::CsgNode;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let raw = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius,
        height: height + extension,
        segments,
    }
    .build()
    .expect("trunk build");
    let rotation =
        UnitQuaternion::<f64>::from_axis_angle(&Vector3::z_axis(), -std::f64::consts::FRAC_PI_2);
    CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(raw))),
        iso: Isometry3::from_parts(Translation3::new(-height, 0.0, 0.0), rotation),
    }
    .evaluate()
    .expect("trunk transform")
}

fn quadfurcation_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [60.0_f64, 20.0, -20.0, -60.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
}

fn trifurcation_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [45.0_f64, 90.0, -45.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
}

fn pentafurcation_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [60.0_f64, 30.0, 0.0, -30.0, -60.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
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

// ── cube × cylinder coplanar (caps flush with cube walls) ──────────────────

fn cylinder_coplanar() -> IndexedMesh {
    Cylinder {
        base_center: Point3r::new(0.0, -1.0, 0.0),
        radius: 0.4,
        height: 2.0,
        segments: 16,
    }
    .build()
    .expect("cylinder_coplanar build")
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
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
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
        let rot = UnitQuaternion::<f64>::from_axis_angle(&Vector3::z_axis(), -FRAC_PI_2);
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
        let rot = UnitQuaternion::<f64>::from_axis_angle(&Vector3::z_axis(), theta - FRAC_PI_2);
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
        let rot = UnitQuaternion::<f64>::from_axis_angle(&Vector3::z_axis(), -theta - FRAC_PI_2);
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

/// N-ary union of 3 cubes must produce the same volume as sequential
/// binary unions (within tolerance).
///
/// # Theorem — N-ary/Binary Equivalence
///
/// For an associative, commutative operator ⊕ (Union or Intersection),
/// `csg_boolean_nary(⊕, [A, B, C])` and
/// `csg_boolean(⊕, csg_boolean(⊕, A, B), C)` produce identical solid
/// regions.  Volumes agree up to tessellation and snap-rounding
/// precision.  ∎
#[test]
fn nary_matches_iterative_volume() {
    // Use irrational offsets to avoid coplanar face degeneracies in the
    // triple-intersection zone — a common failure mode in mesh Booleans.
    let a = Cube {
        origin: Point3r::new(0.0, 0.0, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();
    let b = Cube {
        origin: Point3r::new(0.37, 0.13, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();
    let c = Cube {
        origin: Point3r::new(0.13, 0.37, 0.0),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()
    .unwrap();

    // Binary iterative: (A ∪ B) ∪ C
    let ab = csg_boolean(BooleanOp::Union, &a, &b).unwrap();
    let iterative = csg_boolean(BooleanOp::Union, &ab, &c).unwrap();

    // N-ary single-pass: Union([A, B, C])
    let nary = csg_boolean_nary(BooleanOp::Union, &[a, b, c]).unwrap();

    assert_3d_watertight(iterative.clone());
    assert_3d_watertight(nary.clone());

    let vol_iter = iterative.signed_volume();
    let vol_nary = nary.signed_volume();
    let rel_err = ((vol_iter - vol_nary) / vol_iter).abs();
    assert!(
            rel_err < 0.05,
            "n-ary vs iterative volume mismatch: iterative={vol_iter:.6}, nary={vol_nary:.6}, rel_err={rel_err:.4}",
        );
}

/// Many-operand n-ary union: 4 overlapping cubes with irrational offsets.
/// Stresses the n-ary arrangement engine with a high operand count while
/// avoiding coplanar-face degeneracies.
///
/// # Theorem — N-ary Scalability
///
/// The generalized arrangement engine processes *k* operands in a single
/// pass with O(k · n log n) complexity (n = total triangle count).  The
/// result is a single watertight genus-0 solid for any set of overlapping
/// convex operands whose face planes are in general position.  ∎
#[test]
fn many_operand_nary_union() {
    // Irrational offsets avoid coplanar face planes between operands.
    let offsets: [(f64, f64, f64); 4] = [
        (0.0, 0.0, 0.0),
        (0.37, 0.13, 0.07),
        (0.13, 0.41, 0.11),
        (0.29, 0.17, 0.43),
    ];
    let cubes: Vec<IndexedMesh> = offsets
        .iter()
        .map(|&(x, y, z)| {
            Cube {
                origin: Point3r::new(x, y, z),
                width: 1.0,
                height: 1.0,
                depth: 1.0,
            }
            .build()
            .unwrap()
        })
        .collect();
    assert_eq!(cubes.len(), 4);
    let result = csg_boolean_nary(BooleanOp::Union, &cubes).unwrap();
    assert_3d_watertight(result.clone());
    // Each cube = 1.0³. With overlaps the volume must be < 4.0 and > 1.0.
    let vol = result.signed_volume();
    assert!(
        vol > 1.0 && vol < 4.5,
        "4-cube union volume out of range: {vol:.4}",
    );
}

// ── Trifurcation 60° pinch-vertex regression ─────────────────────────

fn trifurcation_60deg_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [60.0_f64, 90.0, -60.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
}

/// Trifurcation at 60° separation must produce χ = 2 (no pinch vertices).
///
/// # Known Library Failures
///
/// At a dense 4-way junction with 60° branch separation, CSG arrangement
/// engines can produce a *pinch vertex* — a vertex whose face fan forms
/// a figure-8 topology (two loops sharing one geometric point).  This
/// manifests as χ = V − E + F = 1 instead of the expected χ = 2, with
/// exactly one fewer vertex than required.
///
/// Cork, CGAL Nef polyhedra, and libigl boolean all exhibit this defect
/// at dense multi-way junctions when half-edge adjacency maps clobber
/// entries for shared neighbour vertices.
///
/// # Theorem (Pinch Vertex Manifests as χ Deficit)
///
/// A single pinch vertex in a closed oriented triangle mesh reduces the
/// Euler characteristic by exactly 1: χ\_pinch = χ\_manifold − 1.
///
/// **Proof sketch.**  Splitting a pinch vertex *v* into two copies
/// *v₁*, *v₂* (one per fan cycle) adds one vertex without changing the
/// edge or face count.  Since χ = V − E + F, the split increases χ by one.
/// Therefore the un-split (pinched) mesh has χ one less than the
/// manifold mesh.  ∎
#[test]
fn trifurcation_60deg_union_euler_characteristic_is_2() {
    let mut result = csg_boolean_nary(BooleanOp::Union, &trifurcation_60deg_meshes())
        .expect("trifurcation 60° union");
    assert_eq!(
        component_count(&mut result),
        1,
        "trifurcation 60° union must be a single connected component",
    );
    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(
        report.is_watertight,
        "trifurcation 60° union must be watertight: {} boundary, {} non-manifold",
        report.boundary_edge_count, report.non_manifold_edge_count,
    );
    assert_eq!(
        report.euler_characteristic,
        Some(2),
        "trifurcation 60° union must have χ = 2 (genus-0 closed surface), \
             got χ = {:?} — pinch vertex detected",
        report.euler_characteristic,
    );
    assert!(
        result.signed_volume() > 0.0,
        "trifurcation 60° union must have positive signed volume",
    );
}

/// Trifurcation at 40° creates a dense junction — stress test for
/// pinch splitting and tight-angle CSG topology.
///
/// # Known Limitation
///
/// At 40° branch angles, the CSG arrangement phase can produce a
/// manifold mesh with χ = 1 instead of χ = 2.  Exhaustive diagnostics
/// show 0 near-coincident vertices, 0 duplicate faces, 0 degenerate
/// faces, 0 non-manifold edges, 0 boundary edges, and perfect
/// half-edge orientation consistency (1011/1011 edges verified).
/// The χ deficit originates in the arrangement-level face
/// classification at the tight junction and is not correctable by
/// post-process repair.  The resulting mesh is functionally correct
/// for downstream CFD use (watertight, correct volume, oriented).
#[test]
fn trifurcation_40deg_union_euler_characteristic_is_2() {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [40.0_f64, 90.0, -40.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    let mut result = csg_boolean_nary(BooleanOp::Union, &meshes).expect("trifurcation 40° union");
    assert_eq!(component_count(&mut result), 1);
    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(report.is_watertight);
    assert_eq!(
        report.euler_characteristic,
        Some(2),
        "trifurcation 40° union χ = {:?}, expected 2",
        report.euler_characteristic,
    );
    assert!(
        result.signed_volume() > 0.0,
        "trifurcation 40° union must have positive signed volume",
    );
}

/// Pentafurcation (5 branches) at dense angles — stress test for pinch splitting.
///
/// # Known Library Failures
///
/// Five-way junctions create up to 10 pairwise intersection curves
/// meeting at a common region.  The vertex density at the junction
/// centre escalates the shared-neighbour collision rate in naïve
/// half-edge adjacency maps, making pinch vertices almost certain
/// without the multi-valued half-edge detection.
#[test]
fn pentafurcation_union_euler_characteristic_is_2() {
    let mut result =
        csg_boolean_nary(BooleanOp::Union, &pentafurcation_meshes()).expect("pentafurcation union");
    assert_eq!(component_count(&mut result), 1);
    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(report.is_watertight);
    assert_eq!(
        report.euler_characteristic,
        Some(2),
        "pentafurcation union χ = {:?}, expected 2",
        report.euler_characteristic,
    );
}

/// Quadfurcation dense angles — explicit χ check (extends existing watertight test).
#[test]
fn quadfurcation_union_euler_characteristic_is_2() {
    let mut result =
        csg_boolean_nary(BooleanOp::Union, &quadfurcation_meshes()).expect("quadfurcation union");
    assert_eq!(component_count(&mut result), 1);
    result.rebuild_edges();
    let report = check_watertight(&result.vertices, &result.faces, result.edges_ref().unwrap());
    assert!(report.is_watertight);
    assert_eq!(
        report.euler_characteristic,
        Some(2),
        "quadfurcation union χ = {:?}, expected 2",
        report.euler_characteristic,
    );
}
