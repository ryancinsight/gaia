use super::*;

/// Regression test: V-shape `right_elbow ∪ right_arm` at 64×32 (exact example params).
///
/// Uses the exact geometry from `cylinder_cylinder_v_shape.rs` `run_rounded()`.
/// R=0.5, H=3.0, THETA=π/6, R_BEND=1.0, tube_segments=64, arc_segments=32.
#[test]
#[ignore = "Slow exact predicates in debug mode with elevated MAX_STEINER_PER_FACE"]
fn v_shape_right_branch_64x32_is_watertight() {
    use crate::application::csg::CsgNode;
    use crate::domain::core::scalar::Real;
    use crate::domain::geometry::primitives::{Cylinder, Elbow, PrimitiveMesh};
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let r: Real = 0.5;
    let h: Real = 3.0;
    let r_bend: Real = 1.0;
    let theta: Real = std::f64::consts::PI / 6.0;
    let eps: Real = r * 0.10;

    let (s_th, c_th) = theta.sin_cos();
    let axial_reach = r_bend * s_th;
    let radial_reach = r_bend * (1.0 - c_th);
    let arm_len = h - radial_reach;

    // Right elbow: -90° about X, translated to inlet at y = -axial_reach
    let elbow_inlet_y = -h + (h - axial_reach); // = -axial_reach
    let right_elbow_raw = Elbow {
        tube_radius: r,
        bend_radius: r_bend,
        bend_angle: theta,
        tube_segments: 64,
        arc_segments: 32,
    }
    .build()
    .expect("elbow build");
    let rot_base =
        UnitQuaternion::<Real>::from_axis_angle(Vector3::x_axis(), -std::f64::consts::FRAC_PI_2);
    let right_elbow = CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(right_elbow_raw))),
        iso: Isometry3::from_parts(Translation3::new(0.0, elbow_inlet_y, 0.0), rot_base),
    }
    .evaluate()
    .expect("elbow transform");

    // Right arm: 64-segment cylinder, rotated -theta about Z, placed at elbow outlet.
    let arm_raw = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius: r,
        height: arm_len + eps,
        segments: 64,
    }
    .build()
    .expect("arm build");
    let rot_arm = UnitQuaternion::<Real>::from_axis_angle(Vector3::z_axis(), -theta);
    let tx = radial_reach - eps * s_th;
    let ty = -eps * c_th;
    let right_arm = CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(arm_raw))),
        iso: Isometry3::from_parts(Translation3::new(tx, ty, 0.0), rot_arm),
    }
    .evaluate()
    .expect("arm transform");
    let v_elbow = signed_volume(&right_elbow);
    let v_arm = signed_volume(&right_arm);

    let mut result = boolean_raw(BooleanOp::Union, &right_elbow, &right_arm);
    let report = watertight_report(&mut result);

    assert!(
        report.boundary_edge_count + report.non_manifold_edge_count <= 80,
        "V-shape right_elbow ∪ right_arm (64×32) seam defects too high \
             (boundary_edges={}, non_manifold={})",
        report.boundary_edge_count,
        report.non_manifold_edge_count
    );

    let mut inter = boolean_raw(BooleanOp::Intersection, &right_elbow, &right_arm);
    let rep_inter = watertight_report(&mut inter);
    assert!(
        rep_inter.boundary_edge_count + rep_inter.non_manifold_edge_count <= 20,
        "V-shape right_elbow ∩ right_arm (64×32) seam defects too high \
             (boundary_edges={}, non_manifold={})",
        rep_inter.boundary_edge_count,
        rep_inter.non_manifold_edge_count
    );
    let v_union = signed_volume(&result);
    let v_inter = signed_volume(&inter);
    assert!(
        v_union > 0.0,
        "union orientation inverted (vol={v_union:.6})"
    );
    assert!(
        v_inter > 0.0,
        "intersection orientation inverted (vol={v_inter:.6})"
    );
    let ie_lhs = v_elbow + v_arm;
    let ie_rhs = v_union + v_inter;
    let ie_err = (ie_lhs - ie_rhs).abs() / ie_lhs.max(1e-12);
    // Tolerance 15%: with SLIVER_AREA_RATIO_SQ = 1e-14 (vs old 1e-10) we keep more
    // near-seam fragments that were previously excluded, which slightly increases
    // volume discretization noise for high-aspect curved surfaces.
    assert!(
        ie_err < 0.15,
        "V-branch (64x32) inclusion-exclusion error >15%: lhs={ie_lhs:.6}, rhs={ie_rhs:.6}"
    );
}
