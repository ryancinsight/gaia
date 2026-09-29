use super::*;

/// Vertices that are clearly distinct should pass validation.
#[test]
fn pslg_accepts_distinct_vertices() {
    let mut pslg = Pslg::new();
    let a = pslg.add_vertex(0.0, 0.0);
    let b = pslg.add_vertex(1.0, 0.0);
    let c = pslg.add_vertex(0.5, 1.0);
    pslg.add_segment(a, b);
    pslg.add_segment(b, c);
    pslg.add_segment(c, a);

    assert!(pslg.validate().is_ok(), "distinct vertices should pass");
}

// ── Adversarial near-degenerate constraint ────────────────────────────────

/// A very thin, long triangle with a constraint across its shortest edge.
/// Tests that scale-relative tolerances handle near-degenerate geometry.
#[test]
fn cdt_near_degenerate_thin_triangle() {
    let mut pslg = Pslg::new();
    // Very elongated triangle: base 1000, height ~0.001
    let a = pslg.add_vertex(0.0, 0.0);
    let b = pslg.add_vertex(1000.0, 0.0);
    let c = pslg.add_vertex(500.0, 0.001);
    pslg.add_segment(a, b);
    pslg.add_segment(b, c);
    pslg.add_segment(c, a);

    let cdt = Cdt::from_pslg(&pslg);
    let dt = cdt.triangulation();

    assert!(
        dt.triangle_count() >= 1,
        "Near-degenerate CDT should produce triangles"
    );
    assert!(cdt.is_constrained(a, b));
    assert!(cdt.is_constrained(b, c));
    assert!(cdt.is_constrained(c, a));
}
