use super::*;

/// **Failure mode**: CDT with a hole entirely inside the domain.
/// Tests that hole-seeding and flood-fill removal correctly identify
/// interior vs. exterior regions.
#[test]
fn cdt_hole_inside_domain() {
    let mut pslg = Pslg::new();
    // Outer square.
    let o0 = pslg.add_vertex(0.0, 0.0);
    let o1 = pslg.add_vertex(4.0, 0.0);
    let o2 = pslg.add_vertex(4.0, 4.0);
    let o3 = pslg.add_vertex(0.0, 4.0);
    pslg.add_segment(o0, o1);
    pslg.add_segment(o1, o2);
    pslg.add_segment(o2, o3);
    pslg.add_segment(o3, o0);

    // Inner square (hole).
    let h0 = pslg.add_vertex(1.0, 1.0);
    let h1 = pslg.add_vertex(3.0, 1.0);
    let h2 = pslg.add_vertex(3.0, 3.0);
    let h3 = pslg.add_vertex(1.0, 3.0);
    pslg.add_segment(h0, h1);
    pslg.add_segment(h1, h2);
    pslg.add_segment(h2, h3);
    pslg.add_segment(h3, h0);

    // Seed the hole.
    pslg.add_hole(2.0, 2.0);
    let cdt = Cdt::from_pslg(&pslg);
    let dt = cdt.triangulation();

    // Verify no triangles exist inside the hole.
    for (_, tri) in dt.interior_triangles() {
        let centroid_x: f64 = tri.vertices.iter().map(|v| dt.vertex(*v).x).sum::<f64>() / 3.0;
        let centroid_y: f64 = tri.vertices.iter().map(|v| dt.vertex(*v).y).sum::<f64>() / 3.0;
        let inside_hole =
            centroid_x > 1.1 && centroid_x < 2.9 && centroid_y > 1.1 && centroid_y < 2.9;
        assert!(
            !inside_hole,
            "triangle centroid ({centroid_x}, {centroid_y}) should not be inside hole"
        );
    }
}

// ── Refinement adversarial ────────────────────────────────────────────────

/// **Failure mode**: Ruppert refinement with acute input angles (< 20°).
/// Known to cause infinite Steiner point insertion loops in naive
/// implementations.  Robust implementations must handle encroachment
/// splitting of obtuse Steiner vertices.
///
/// **Literature**: Ruppert (1995), "A Delaunay Refinement Algorithm for
/// Quality 2-Dimensional Mesh Generation." Theorem 4: termination requires
/// minimum input angle ≥ ~20.7°.
#[test]
fn ruppert_acute_input_angle() {
    use crate::application::delaunay::dim2::refinement::ruppert::RuppertRefiner;

    let mut pslg = Pslg::new();
    // Isoceles triangle with very narrow top — ~10° at the apex.
    let v0 = pslg.add_vertex(0.0, 0.0);
    let v1 = pslg.add_vertex(2.0, 0.0);
    // Apex angle ~10°.
    let apex_half_angle = 5.0_f64.to_radians();
    let h = 1.0 / apex_half_angle.tan();
    let v2 = pslg.add_vertex(1.0, h);
    pslg.add_segment(v0, v1);
    pslg.add_segment(v1, v2);
    pslg.add_segment(v2, v0);

    let cdt = Cdt::from_pslg(&pslg);
    let mut refiner = RuppertRefiner::new(cdt);
    refiner.set_max_steiner(500);
    let _n_steiner = refiner.refine();
    // Should terminate (the Steiner limit prevents infinite loops).
    let dt = refiner.cdt().triangulation();
    assert!(dt.triangle_count() > 0);
    assert!(dt.is_delaunay());
}

/// **Failure mode**: Ruppert on a highly elongated rectangle.  Tests that
/// refinement correctly handles high aspect-ratio input domains.
#[test]
fn ruppert_elongated_rectangle() {
    use crate::application::delaunay::dim2::refinement::ruppert::RuppertRefiner;

    let mut pslg = Pslg::new();
    let v0 = pslg.add_vertex(0.0, 0.0);
    let v1 = pslg.add_vertex(10.0, 0.0);
    let v2 = pslg.add_vertex(10.0, 1.0);
    let v3 = pslg.add_vertex(0.0, 1.0);
    pslg.add_segment(v0, v1);
    pslg.add_segment(v1, v2);
    pslg.add_segment(v2, v3);
    pslg.add_segment(v3, v0);

    let cdt = Cdt::from_pslg(&pslg);
    let mut refiner = RuppertRefiner::new(cdt);
    refiner.set_max_area(0.5);
    refiner.set_max_steiner(10000);
    let _n_steiner = refiner.refine();
    let dt = refiner.cdt().triangulation();
    assert!(
        dt.triangle_count() > 10,
        "10×1 rectangle with max_area=0.5 should have many triangles, got {}",
        dt.triangle_count()
    );
    assert!(dt.is_delaunay());
}
