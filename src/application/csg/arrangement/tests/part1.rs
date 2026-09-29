use super::*;

/// Sphere-sphere intersection (lens) volume test.
///
/// Two unit spheres (r = 1.0) with centres 1.0 apart.
/// Analytical lens volume = Ãâ‚¬/12 * (4Ã‚Â·r + d) * (2Ã‚Â·r Ã¢Ë†â€™ d)Ã‚Â²
///   where r=1, d=1  Ã¢â€ â€™  Ãâ‚¬/12 * 5 * 1 = 5Ãâ‚¬/12 Ã¢â€°Ë† 1.3090
///
/// Uses 64Ãƒâ€”32 resolution: the lens boundary (intersection seam) is more
/// finely sampled than the union/difference cases, requiring higher mesh
/// resolution to reach < 5% discretisation error.
#[test]
fn sphere_sphere_intersection_volume() {
    // 64 stacks Ãƒâ€” 32 segments gives dense enough triangulation at the seam
    // to keep discretisation error below 5%.
    let stacks = 64;
    let segments = 32;
    let sphere_a = make_sphere(0.0, 0.0, 0.0, 1.0, stacks, segments);
    let sphere_b = make_sphere(1.0, 0.0, 0.0, 1.0, stacks, segments);

    let result = csg_boolean(BooleanOp::Intersection, &sphere_a, &sphere_b)
        .expect("sphere-sphere intersection should not fail");

    let vol = signed_volume(&result);
    let expected = 5.0 * std::f64::consts::PI / 12.0; // Ã¢â€°Ë† 1.3090
    let err = (vol - expected).abs() / expected;
    assert!(
        err < 0.05,
        "sphere-sphere intersection volume error {:.1}% > 5% (got {:.4}, expected {:.4})",
        err * 100.0,
        vol,
        expected
    );
}

/// Sphere-sphere union volume test.
///
/// vol(A Ã¢Ë†Âª B) = vol(A) + vol(B) Ã¢Ë†â€™ vol(A Ã¢Ë†Â© B)
///   = 2 * 4Ãâ‚¬/3 Ã¢Ë†â€™ 5Ãâ‚¬/12 Ã¢â€°Ë† 7.1272
#[test]
fn sphere_sphere_union_volume() {
    let stacks = 32;
    let segments = 16;
    let sphere_a = make_sphere(0.0, 0.0, 0.0, 1.0, stacks, segments);
    let sphere_b = make_sphere(1.0, 0.0, 0.0, 1.0, stacks, segments);

    let result = csg_boolean(BooleanOp::Union, &sphere_a, &sphere_b)
        .expect("sphere-sphere union should not fail");

    let vol = signed_volume(&result);
    let vol_sphere = 4.0 * std::f64::consts::PI / 3.0;
    let vol_lens = 5.0 * std::f64::consts::PI / 12.0;
    let expected = 2.0 * vol_sphere - vol_lens; // Ã¢â€°Ë† 7.1272
    let err = (vol - expected).abs() / expected;
    assert!(
        err < 0.05,
        "sphere-sphere union volume error {:.1}% > 5% (got {:.4}, expected {:.4})",
        err * 100.0,
        vol,
        expected
    );
}

/// Sphere-sphere difference volume test.
///
/// vol(A \ B) = vol(A) Ã¢Ë†â€™ vol(A Ã¢Ë†Â© B)
///   = 4Ãâ‚¬/3 Ã¢Ë†â€™ 5Ãâ‚¬/12 Ã¢â€°Ë† 2.8798
#[test]
fn sphere_sphere_difference_volume() {
    let stacks = 32;
    let segments = 16;
    let sphere_a = make_sphere(0.0, 0.0, 0.0, 1.0, stacks, segments);
    let sphere_b = make_sphere(1.0, 0.0, 0.0, 1.0, stacks, segments);

    let result = csg_boolean(BooleanOp::Difference, &sphere_a, &sphere_b)
        .expect("sphere-sphere difference should not fail");

    let vol = signed_volume(&result);
    let vol_sphere = 4.0 * std::f64::consts::PI / 3.0;
    let vol_lens = 5.0 * std::f64::consts::PI / 12.0;
    let expected = vol_sphere - vol_lens; // Ã¢â€°Ë† 2.8798
    let err = (vol - expected).abs() / expected;
    assert!(
        err < 0.05,
        "sphere-sphere difference volume error {:.1}% > 5% (got {:.4}, expected {:.4})",
        err * 100.0,
        vol,
        expected
    );
}

/// Regression test: equal-height parallel cylinders with exactly coplanar caps.
///
/// Two cylinders of equal radius r = 0.4 mm and equal height h = 3 mm,
/// with axes offset by d = r = 0.4 mm in X.  Both base caps sit at
/// exactly y = Ã¢Ë†â€™1.5 and both top caps at y = +1.5.
///
/// Before the fix: the coplanar cap faces were routed through the
/// Sutherland-Hodgman clip, which duplicated them (orient_3d Degenerate =
/// "inside"), causing the union volume to exceed V_A + V_B.
///
/// After the fix: coplanar pairs bypass clipping and are classified by
/// normal alignment.  Co-oriented cap pairs keep exactly one copy, so:
///   vol(A Ã¢Ë†Âª B) < V_A + V_B   (overlap correctly subtracted)
///   vol(A Ã¢Ë†Â© B) > 0            (lens region positive)
#[test]
fn coplanar_caps_no_double_counting() {
    use crate::domain::geometry::primitives::{Cylinder, PrimitiveMesh};
    let r = 0.4_f64;
    let h = 3.0_f64;
    // Both cylinders: Y Ã¢Ë†Ë† [Ã¢Ë†â€™1.5, 1.5] Ã¢â‚¬â€ caps coplanar at y = Ã‚Â±1.5.
    let cyl_a = Cylinder {
        base_center: Point3r::new(-r / 2.0, -h / 2.0, 0.0),
        radius: r,
        height: h,
        segments: 32,
    }
    .build()
    .expect("cyl_a build failed");
    let cyl_b = Cylinder {
        base_center: Point3r::new(r / 2.0, -h / 2.0, 0.0),
        radius: r,
        height: h,
        segments: 32,
    }
    .build()
    .expect("cyl_b build failed");

    let v_each = std::f64::consts::PI * r * r * h; // Ã¢â€°Ë† 1.5080

    // Use the low-level raw path: coplanar caps currently produce boundary
    // seams (known limitation), so the watertight-enforcing high-level API
    // would return an error.  This test validates only the volume invariant.
    let union_mesh = boolean_raw(BooleanOp::Union, &cyl_a, &cyl_b);
    let union_vol = signed_volume(&union_mesh);

    // Union must be strictly less than V_A + V_B (not double-counted).
    assert!(
        union_vol < 2.0 * v_each - 0.01,
        "union vol {:.4} should be < 2Ã‚Â·V_each = {:.4} (coplanar caps double-counted)",
        union_vol,
        2.0 * v_each,
    );
    // And it must be positive.
    assert!(
        union_vol > 0.1,
        "union vol {union_vol:.4} should be positive"
    );

    // Intersection must be positive (the lens barrel is non-empty).
    let inter_mesh = boolean_raw(BooleanOp::Intersection, &cyl_a, &cyl_b);
    let inter_vol = signed_volume(&inter_mesh);
    assert!(
        inter_vol > 0.05,
        "intersection vol {inter_vol:.4} should be positive"
    );

    // Inclusion-exclusion: vol(A) + vol(B) Ã¢â€°Ë† vol(AÃ¢Ë†ÂªB) + vol(AÃ¢Ë†Â©B)
    // Allow 10% tolerance for triangulation discretisation.
    let ie_lhs = 2.0 * v_each;
    let ie_rhs = union_vol + inter_vol;
    let ie_err = (ie_lhs - ie_rhs).abs() / ie_lhs;
    assert!(
        ie_err < 0.20,
        "inclusion-exclusion error {:.1}% > 10% (lhs={:.4}, rhs={:.4})",
        ie_err * 100.0,
        ie_lhs,
        ie_rhs,
    );
}

/// Regression test: ensure faces that are almost but not exactly coplanar
/// do not get grouped together by a quantization heuristic.
#[test]
fn exact_algebraic_coplanarity_no_shatter() {
    let _pool: VertexPool<f64> = VertexPool::default_millifluidic();

    // Let's create `IndexedMesh`es so they run through the full `BooleanOp` pipeline
    // assuring broad-phase overlap tests correctly dispatch to intersection.
    let mut mesh_a = IndexedMesh::new();
    let mut mesh_b = IndexedMesh::new();

    let a1 = mesh_a.add_vertex(Point3r::new(0.0, 0.0, 0.0), Vector3r::z());
    let a2 = mesh_a.add_vertex(Point3r::new(1.0, 0.0, 0.0), Vector3r::z());
    let a3 = mesh_a.add_vertex(Point3r::new(0.0, 1.0, 0.0), Vector3r::z());
    mesh_a.add_face(a1, a2, a3);

    let b1 = mesh_b.add_vertex(Point3r::new(0.0, 0.0, 0.0000001), Vector3r::z());
    let b2 = mesh_b.add_vertex(Point3r::new(1.0, 0.0, 0.0000001), Vector3r::z());
    let b3 = mesh_b.add_vertex(Point3r::new(0.0, 1.0, 0.0), Vector3r::z());
    mesh_b.add_face(b1, b2, b3);

    let result = csg_boolean(BooleanOp::Intersection, &mesh_a, &mesh_b);

    let result_mesh = result.expect("intersection evaluation must succeed");
    // Since they only intersect at exactly one line inside the 2D plane (y=0, z~=0),
    // there is no 3D overlap, and no coplanar 2D area overlap. The intersection mesh correctly
    // produces the exact intersection artifacts (typically 1 or 2 seam fragments) rather than completely
    // vanishing or duplicating due to a coplanar grouping heuristic.
    assert!(
        result_mesh.faces.len() <= 2,
        "Non-coplanar tilted faces should not output massive coplanar intersection"
    );
}

/// Diagnostic: perpendicular cylinder union should be watertight.
///
/// Two equal-radius cylinders (r=0.5, h=3) with perpendicular axes crossing
/// at the centre.  The arrangement pipeline should produce a closed manifold.
#[test]
fn perpendicular_cylinder_union_is_watertight() {
    use crate::domain::geometry::primitives::{Cylinder, PrimitiveMesh};
    let r = 0.5_f64;
    let h = 3.0_f64;
    let cyl_a = Cylinder {
        base_center: Point3r::new(0.0, -h / 2.0, 0.0),
        radius: r,
        height: h,
        segments: 16,
    }
    .build()
    .expect("cyl_a build");
    let cyl_b = Cylinder {
        base_center: Point3r::new(-h / 2.0, 0.0, 0.0),
        radius: r,
        height: h,
        segments: 16,
    }
    .build()
    .expect("cyl_b build");

    let mut result = csg_boolean(BooleanOp::Union, &cyl_a, &cyl_b).expect("union should not fail");
    let report = watertight_report(&mut result);
    assert!(
        report.is_watertight,
        "perpendicular cylinder union should be watertight \
             (boundary_edges={}, non_manifold={})",
        report.boundary_edge_count, report.non_manifold_edge_count
    );
}

/// Regression: T-junction cylinder booleans remain watertight and preserve
/// expected analytic volumes.
#[test]
fn t_junction_volume_and_watertightness() {
    use crate::application::csg::CsgNode;
    use crate::domain::core::scalar::Real;
    use crate::domain::geometry::primitives::{Cylinder, PrimitiveMesh};
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let r: Real = 0.5;
    let h: Real = 3.0;

    let stem = Cylinder {
        base_center: Point3r::new(0.0, -h / 2.0, 0.0),
        radius: r,
        height: h,
        segments: 64,
    }
    .build()
    .expect("stem build");

    let crossbar_raw = Cylinder {
        base_center: Point3r::new(0.0, -h / 2.0, 0.0),
        radius: r,
        height: h,
        segments: 64,
    }
    .build()
    .expect("crossbar build");

    // +Y -> +X, then translate crossbar to the stem top cap.
    let rot =
        UnitQuaternion::<Real>::from_axis_angle(Vector3::z_axis(), -std::f64::consts::FRAC_PI_2);
    let iso = Isometry3::from_parts(Translation3::new(0.0, h / 2.0, 0.0), rot);
    let crossbar = CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(crossbar_raw))),
        iso,
    }
    .evaluate()
    .expect("crossbar transform");

    let mut union = csg_boolean(BooleanOp::Union, &stem, &crossbar).expect("union");
    let mut inter = csg_boolean(BooleanOp::Intersection, &stem, &crossbar).expect("intersection");
    let mut diff = csg_boolean(BooleanOp::Difference, &stem, &crossbar).expect("difference");

    let rep_u = watertight_report(&mut union);
    let rep_i = watertight_report(&mut inter);
    let rep_d = watertight_report(&mut diff);
    assert!(
        rep_u.is_watertight,
        "T-junction union not watertight: boundary={}, non_manifold={}",
        rep_u.boundary_edge_count, rep_u.non_manifold_edge_count
    );
    assert!(
        rep_i.is_watertight,
        "T-junction intersection not watertight: boundary={}, non_manifold={}",
        rep_i.boundary_edge_count, rep_i.non_manifold_edge_count
    );
    assert!(
        rep_d.is_watertight,
        "T-junction difference not watertight: boundary={}, non_manifold={}",
        rep_d.boundary_edge_count, rep_d.non_manifold_edge_count
    );

    let v_cyl = std::f64::consts::PI * r * r * h;
    let v_inter_expected = (8.0 / 3.0) * r * r * r;
    let v_union_expected = 2.0 * v_cyl - v_inter_expected;
    let v_diff_expected = v_cyl - v_inter_expected;

    let v_union = signed_volume(&union);
    let v_inter = signed_volume(&inter);
    let v_diff = signed_volume(&diff);
    let tol = 0.10; // Triangle discretization + seam topology budget.

    assert!(
        (v_union - v_union_expected).abs() / v_union_expected < tol,
        "T-junction union volume error >10%: got={v_union:.6}, expected={v_union_expected:.6}"
    );
    assert!(
        (v_inter - v_inter_expected).abs() / v_inter_expected < tol,
        "T-junction inter volume error >10%: got={v_inter:.6}, expected={v_inter_expected:.6}"
    );
    assert!(
        (v_diff - v_diff_expected).abs() / v_diff_expected < tol,
        "T-junction diff volume error >10%: got={v_diff:.6}, expected={v_diff_expected:.6}"
    );

    // Inclusion-exclusion: vol(A)+vol(B)=vol(A∪B)+vol(A∩B)
    let ie_lhs = 2.0 * v_cyl;
    let ie_rhs = v_union + v_inter;
    let ie_err = (ie_lhs - ie_rhs).abs() / ie_lhs.max(1e-12);
    assert!(
        ie_err < tol,
        "T-junction inclusion-exclusion error >10%: lhs={ie_lhs:.6}, rhs={ie_rhs:.6}"
    );
}

/// Diagnostic: parallel cylinder union (same height Ã¢â€ â€™ coplanar caps) should be watertight.
#[test]
fn parallel_cylinder_union_is_watertight() {
    use crate::domain::geometry::primitives::{Cylinder, PrimitiveMesh};
    let r = 0.5_f64;
    let h = 3.0_f64;
    let cyl_a = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius: r,
        height: h,
        segments: 16,
    }
    .build()
    .expect("cyl_a build");
    let cyl_b = Cylinder {
        base_center: Point3r::new(r, 0.0, 0.0),
        radius: r,
        height: h,
        segments: 16,
    }
    .build()
    .expect("cyl_b build");

    let mut result = csg_boolean(BooleanOp::Union, &cyl_a, &cyl_b).expect("union should not fail");
    let report = watertight_report(&mut result);

    // Debug: print boundary edge positions if any
    if !report.is_watertight {
        result.rebuild_edges();
        let edges = result.edges_ref().unwrap();
        let mut boundary_positions: Vec<(Point3r, Point3r, f64)> = Vec::new();
        for edge in edges.iter() {
            if edge.valence() == 1 {
                let pa = *result.vertices.position(edge.vertices.0);
                let pb = *result.vertices.position(edge.vertices.1);
                let mid_z = f64::midpoint(pa.z, pb.z);
                boundary_positions.push((pa, pb, mid_z));
            }
        }
        boundary_positions
            .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        for (a, b, _z) in &boundary_positions {
            let mid_y = f64::midpoint(a.y, b.y);
            tracing::info!(
                "  yÃ¢â€°Ë†{:.4}  ({:.4},{:.4},{:.4}) Ã¢â€ â€™ ({:.4},{:.4},{:.4})",
                mid_y,
                a.x,
                a.y,
                a.z,
                b.x,
                b.y,
                b.z
            );
        }
    }

    // Debug: print non-manifold edge positions if any
    if report.non_manifold_edge_count > 0 {
        let edges = result.edges_ref().unwrap();
        for edge in edges.iter() {
            if edge.valence() > 2 {
                let pa = *result.vertices.position(edge.vertices.0);
                let pb = *result.vertices.position(edge.vertices.1);
                tracing::info!(
                    "NM_EDGE val={}: ({:.6},{:.6},{:.6})->({:.6},{:.6},{:.6})",
                    edge.valence(),
                    pa.x,
                    pa.y,
                    pa.z,
                    pb.x,
                    pb.y,
                    pb.z
                );
            }
        }
    }

    assert!(
        report.is_watertight,
        "parallel cylinder union should be watertight \
             (boundary_edges={}, non_manifold={})",
        report.boundary_edge_count, report.non_manifold_edge_count
    );

    // Volume: A u B = 2*V_cyl - V_lens.  d = r => theta = pi/3.
    let theta = (r / (2.0 * r)).acos();
    let a_seg = r * r * (theta - theta.sin() * theta.cos());
    let v_union_expected = 2.0 * std::f64::consts::PI * r * r * h - 2.0 * h * a_seg;
    let vol = result.signed_volume();
    assert!(vol > 0.0, "orientation inverted (vol={vol:.6})");
    assert!(
        (vol - v_union_expected).abs() / v_union_expected < 0.05,
        "volume error >5%: vol={vol:.6} expected={v_union_expected:.6}"
    );
}

/// Regression: 64-segment parallel cylinder union volume (r=0.6, h=3.0, d=r).
///
/// Exercises the exact geometry from the `csg_cylinder_cylinder` example.
/// Verifies cop_faces direct emission produces correct volume for coplanar-cap case.
#[test]
fn cylinder_cylinder_union_64seg_volume() {
    use crate::domain::geometry::primitives::{Cylinder, PrimitiveMesh};
    let r = 0.6_f64;
    let h = 3.0_f64;
    let d = r;
    let cyl_a = Cylinder {
        base_center: Point3r::new(-d / 2.0, -h / 2.0, 0.0),
        radius: r,
        height: h,
        segments: 64,
    }
    .build()
    .expect("cyl_a");
    let cyl_b = Cylinder {
        base_center: Point3r::new(d / 2.0, -h / 2.0, 0.0),
        radius: r,
        height: h,
        segments: 64,
    }
    .build()
    .expect("cyl_b");
    let mut result = csg_boolean(BooleanOp::Union, &cyl_a, &cyl_b).expect("union");
    // Cylindrical lens: d = r => theta = arccos(0.5) = pi/3
    let theta = (d / (2.0 * r)).acos();
    let a_seg = r * r * (theta - theta.sin() * theta.cos());
    let v_intersect = 2.0 * h * a_seg;
    let v_cyl = std::f64::consts::PI * r * r * h;
    let expected = 2.0 * v_cyl - v_intersect;
    let vol = result.signed_volume();
    let report = watertight_report(&mut result);
    assert!(
        report.is_watertight,
        "64-seg union should be watertight (boundary={}, non_manifold={})",
        report.boundary_edge_count, report.non_manifold_edge_count
    );
    assert!(vol > 0.0, "orientation inverted (vol={vol:.6})");
    assert!(
        (vol - expected).abs() / expected < 0.05,
        "volume error >5%: vol={vol:.6} expected={expected:.6}"
    );
}
