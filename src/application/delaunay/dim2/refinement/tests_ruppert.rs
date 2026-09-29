//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::application::delaunay::dim2::constraint::enforce::Cdt;
use crate::application::delaunay::dim2::pslg::graph::Pslg;
use crate::application::delaunay::dim2::refinement::metric::MetricTensor;

/// Build a square PSLG (0,0)-(1,0)-(1,1)-(0,1) with constrained boundary.
fn square_cdt() -> Cdt {
    let mut p = Pslg::new();
    let v0 = p.add_vertex(0.0, 0.0);
    let v1 = p.add_vertex(1.0, 0.0);
    let v2 = p.add_vertex(1.0, 1.0);
    let v3 = p.add_vertex(0.0, 1.0);
    p.add_segment(v0, v1);
    p.add_segment(v1, v2);
    p.add_segment(v2, v3);
    p.add_segment(v3, v0);
    Cdt::from_pslg(&p)
}

/// Isotropic refiner terminates and produces ≥ 2 interior triangles.
#[test]
fn isotropic_refiner_terminates() {
    let cdt = square_cdt();
    let mut refiner = RuppertRefiner::new(cdt);
    refiner.set_max_ratio(1.5);
    let n = refiner.refine();
    assert!(n <= 100_000, "Steiner count should be bounded");
    assert!(
        refiner.cdt().triangulation().interior_triangles().count() >= 2,
        "should have interior triangles"
    );
}

/// Identity metric gives same result as no metric (backward compat).
#[test]
fn identity_metric_matches_no_metric() {
    let cdt_a = square_cdt();
    let cdt_b = square_cdt();

    let mut r_iso = RuppertRefiner::new(cdt_a);
    r_iso.set_max_ratio(1.5);
    r_iso.set_max_steiner(50);
    r_iso.refine();

    let mut r_id = RuppertRefiner::new(cdt_b).with_metric(MetricTensor::identity());
    r_id.set_max_ratio(1.5);
    r_id.set_max_steiner(50);
    r_id.refine();

    // Both should produce the same number of interior triangles
    // (identity metric ≡ Euclidean quality).
    assert_eq!(
        r_iso.cdt().triangulation().interior_triangles().count(),
        r_id.cdt().triangulation().interior_triangles().count(),
        "identity metric should give same result as no metric"
    );
}

/// Anisotropic metric terminates within the safety limit.
///
/// A 2:1 anisotropic metric on the unit square is used (rather than 10:1)
/// because very high aspect ratios cause O(α²) insertions on a square domain
/// where all edges are comparable in Euclidean space.  The test checks that
/// the refiner honours `max_steiner` and produces a valid CDT.
#[test]
fn anisotropic_refiner_terminates() {
    let cdt = square_cdt();
    // 2:1 anisotropy along x: triangles may be 2× longer in x than in y.
    let metric = MetricTensor::anisotropic(0.0, 2.0);
    let mut refiner = RuppertRefiner::new(cdt).with_metric(metric);
    refiner.set_max_ratio(1.5);
    refiner.set_max_steiner(2_000); // explicit safety cap for test
    let n = refiner.refine();
    assert!(
        n <= 2_000,
        "anisotropic refiner Steiner count should be bounded: {n}"
    );
    assert!(
        refiner.cdt().triangulation().interior_triangles().count() >= 2,
        "CDT must contain triangles after anisotropic refinement"
    );
}

/// set_metric / clear_metric mutate metric field correctly.
#[test]
fn set_and_clear_metric() {
    let mut refiner = RuppertRefiner::new(square_cdt());
    assert!(refiner.metric.is_none());
    let metric = MetricTensor::anisotropic(0.5, 5.0);
    refiner.set_metric(metric);
    assert_eq!(refiner.metric, Some(metric));
    refiner.clear_metric();
    assert!(refiner.metric.is_none());
}

/// with_metric builder sets metric; final CDT is valid (≥ 1 alive triangle).
///
/// Uses a moderate 3:1 anisotropy to keep Steiner point count manageable.
#[test]
fn with_metric_builder_produces_valid_cdt() {
    let cdt = square_cdt();
    let mut refiner = RuppertRefiner::new(cdt).with_metric(MetricTensor::anisotropic(0.0, 3.0));
    refiner.set_max_steiner(2_000);
    refiner.refine();
    let tri_count = refiner.cdt().triangulation().interior_triangles().count();
    assert!(
        tri_count >= 1,
        "CDT must contain triangles after anisotropic refinement"
    );
}

#[test]
fn bad_triangle_ordering_is_total() {
    let nan_a = BadTriangle {
        tid: TriangleId::new(0),
        ratio: Real::NAN,
    };
    let nan_b = BadTriangle {
        tid: TriangleId::new(1),
        ratio: Real::NAN,
    };
    assert_eq!(nan_a.cmp(&nan_b), Ordering::Equal);
    assert_eq!(nan_a, nan_b);

    let negative_zero = BadTriangle {
        tid: TriangleId::new(2),
        ratio: -0.0,
    };
    let positive_zero = BadTriangle {
        tid: TriangleId::new(3),
        ratio: 0.0,
    };
    assert_eq!(negative_zero.cmp(&positive_zero), Ordering::Less);
    assert_ne!(negative_zero, positive_zero);
}
