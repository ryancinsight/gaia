use super::*;

#[test]
fn min_connectivity_large_random() {
    let mut rng = 123_u64;
    let mut pts = Vec::with_capacity(50);
    for _ in 0..50 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (rng >> 33) as f64 / (1u64 << 31) as f64;
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let y = (rng >> 33) as f64 / (1u64 << 31) as f64;
        pts.push((x, y));
    }
    let dt = DelaunayTriangulation::from_points(&pts);
    let kappa = dt.min_vertex_connectivity();
    assert!(
        kappa >= 2,
        "50-point random triangulation min connectivity should be ≥ 2, got {kappa}"
    );
}

// ── Scale-relative quality metrics ────────────────────────────────────────

/// Test that quality metrics work correctly at micro-scale (1e-6 range).
#[test]
fn quality_micro_scale() {
    use crate::application::delaunay::dim2::pslg::vertex::PslgVertex;
    use crate::application::delaunay::dim2::refinement::quality::TriangleQuality;

    // Equilateral triangle at micro-scale.
    let s = 1e-6;
    let a = PslgVertex::new(0.0, 0.0);
    let b = PslgVertex::new(s, 0.0);
    let c = PslgVertex::new(s * 0.5, s * 0.866_025_403_784);
    let q = TriangleQuality::compute(&a, &b, &c);
    // Equilateral triangle: ratio ≈ 0.577..
    assert!(
        q.radius_edge_ratio < 0.6,
        "Micro-scale equilateral ratio {} should be < 0.6",
        q.radius_edge_ratio
    );
    assert!(
        q.radius_edge_ratio > 0.5,
        "Ratio too small: {}",
        q.radius_edge_ratio
    );
}

/// Degenerate sliver triangle should have very high ratio.
#[test]
fn quality_sliver_degenerate() {
    use crate::application::delaunay::dim2::pslg::vertex::PslgVertex;
    use crate::application::delaunay::dim2::refinement::quality::TriangleQuality;

    let a = PslgVertex::new(0.0, 0.0);
    let b = PslgVertex::new(10.0, 0.0);
    let c = PslgVertex::new(5.0, 1e-8);
    let q = TriangleQuality::compute(&a, &b, &c);
    // Sliver: very high ratio.
    assert!(
        q.radius_edge_ratio > 10.0,
        "Sliver should have high ratio, got {}",
        q.radius_edge_ratio
    );
}
