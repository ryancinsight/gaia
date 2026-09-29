//! Tests for the parent module, extracted from the module body.

use super::*;
use leto::geometry::Point2;

// ── orient_2d ──────────────────────────────────────────────────────────

#[test]
fn orient_2d_ccw() {
    // Standard CCW triangle
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(1.0, 0.0);
    let c = Point2::new(0.0, 1.0);
    assert_eq!(orient_2d(&a, &b, &c), Orientation::Positive);
}

#[test]
fn orient_2d_cw() {
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(0.0, 1.0);
    let c = Point2::new(1.0, 0.0);
    assert_eq!(orient_2d(&a, &b, &c), Orientation::Negative);
}

#[test]
fn orient_2d_collinear() {
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(1.0, 0.0);
    let c = Point2::new(2.0, 0.0);
    assert_eq!(orient_2d(&a, &b, &c), Orientation::Degenerate);
}

// ── orient_3d ──────────────────────────────────────────────────────────

#[test]
fn orient_3d_positive_tet() {
    let o = [0.0, 0.0, 0.0];
    let ex = [1.0, 0.0, 0.0];
    let ey = [0.0, 1.0, 0.0];
    let ez = [0.0, 0.0, 1.0];
    assert_eq!(orient_3d(o, ex, ey, ez), Orientation::Positive);
}

#[test]
fn orient_3d_negative_tet() {
    let o = [0.0, 0.0, 0.0];
    let ex = [1.0, 0.0, 0.0];
    let ey = [0.0, 1.0, 0.0];
    // Swap ex and ey to flip sign
    assert_eq!(orient_3d(o, ey, ex, [0.0, 0.0, 1.0]), Orientation::Negative);
}

#[test]
fn orient_3d_coplanar() {
    // All four points in z=0 plane
    assert_eq!(
        orient_3d(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ),
        Orientation::Degenerate
    );
}

// ── incircle ───────────────────────────────────────────────────────────

#[test]
fn incircle_inside() {
    // Unit circle: points on the circle, test point at origin (inside)
    let a = Point2::new(1.0, 0.0);
    let b = Point2::new(0.0, 1.0);
    let c = Point2::new(-1.0, 0.0);
    let d = Point2::new(0.0, 0.0); // origin — strictly inside
    assert_eq!(incircle(&a, &b, &c, &d), Orientation::Positive);
}

#[test]
fn incircle_outside() {
    let a = Point2::new(1.0, 0.0);
    let b = Point2::new(0.0, 1.0);
    let c = Point2::new(-1.0, 0.0);
    let d = Point2::new(0.0, 2.0); // above the circle — outside
    assert_eq!(incircle(&a, &b, &c, &d), Orientation::Negative);
}

// ── insphere ───────────────────────────────────────────────────────────

#[test]
fn insphere_inside() {
    // Unit sphere vertices, test point at origin
    let a = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    let c = [0.0, 0.0, 1.0];
    let d = [-1.0, 0.0, 0.0];
    let e = [0.0, 0.0, 0.0]; // origin — inside unit sphere
    assert_eq!(insphere(a, b, c, d, e), Orientation::Positive);
}

#[test]
fn insphere_outside() {
    let a = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    let c = [0.0, 0.0, 1.0];
    let d = [-1.0, 0.0, 0.0];
    let e = [2.0, 0.0, 0.0]; // outside unit sphere
    assert_eq!(insphere(a, b, c, d, e), Orientation::Negative);
}

// ── Generic instantiation: exact promotion, identity at f64 ─────────────

/// Dyadic coordinates store identically at `f32` and `f64`, and the
/// promotion into the `f64` predicate arithmetic is lossless, so both
/// instantiations must return the same sign for the same stored values
/// (ADR 0005). `f64::from` performs the same exact promotion the wrapper
/// performs internally.
#[test]
fn f32_instantiation_matches_f64_on_dyadic_inputs() {
    let triple_2d = [
        ([0.0_f32, 0.0], [1.0, 0.0], [0.0, 1.0]),
        ([0.0_f32, 0.0], [0.0, 1.0], [1.0, 0.0]),
        ([0.25_f32, 0.125], [0.5, 0.25], [0.125, 0.5]),
    ];
    for (a, b, c) in triple_2d {
        let promote = |v: [f32; 2]| Point2::new(f64::from(v[0]), f64::from(v[1]));
        assert_eq!(
            orient_2d(
                &Point2::new(a[0], a[1]),
                &Point2::new(b[0], b[1]),
                &Point2::new(c[0], c[1])
            ),
            orient_2d(&promote(a), &promote(b), &promote(c)),
            "orient_2d must agree across precisions on {a:?} {b:?} {c:?}"
        );
    }

    let quad_3d = [
        (
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ),
        (
            [0.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ),
    ];
    for (a, b, c, d) in quad_3d {
        let promote = |v: [f32; 3]| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])];
        assert_eq!(
            orient_3d(a, b, c, d),
            orient_3d(promote(a), promote(b), promote(c), promote(d)),
            "orient_3d must agree across precisions on {a:?} {b:?} {c:?} {d:?}"
        );
    }

    let sphere = [
        [1.0_f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let promoted: [[f64; 3]; 5] =
        sphere.map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]);
    assert_eq!(
        insphere(sphere[0], sphere[1], sphere[2], sphere[3], sphere[4]),
        insphere(
            promoted[0],
            promoted[1],
            promoted[2],
            promoted[3],
            promoted[4]
        ),
        "insphere must agree across precisions on the unit-sphere fixture"
    );
}

/// An exactly-collinear `f32` configuration is collinear after promotion:
/// lossless promotion must preserve exact degeneracy, not approximate it.
#[test]
fn promotion_preserves_exactly_degenerate_f32_configuration() {
    let a = [0.0_f32, 0.0, 0.0];
    let b = [0.5_f32, 0.5, 0.5];
    let c = [1.0_f32, 1.0, 1.0];
    let d = [2.0_f32, 2.0, 2.0];
    let promote = |v: [f32; 3]| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])];
    assert_eq!(
        orient_3d(a, b, c, d),
        orient_3d(promote(a), promote(b), promote(c), promote(d))
    );
    assert_eq!(orient_3d(a, b, c, d), Orientation::Degenerate);
}

// ── Orientation helpers ────────────────────────────────────────────────

#[test]
fn orientation_helpers() {
    assert!(Orientation::Positive.is_positive());
    assert!(!Orientation::Positive.is_negative());
    assert!(!Orientation::Positive.is_degenerate());

    assert!(Orientation::Negative.is_negative());
    assert!(Orientation::Degenerate.is_degenerate());
}

// ── GAIA-004: naive-oracle differentials on constructed failures ─────────

/// Naive f64 evaluation of the 2-D orientation determinant — the
/// non-robust oracle the exact wrappers must beat.
fn naive_orient_2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// A pair of f64-exact points whose orientation determinant is provably
/// positive but of magnitude 2⁻¹⁰⁴: with `t = 1 + 2⁻⁵²` and
/// `u = 1 + 2⁻⁵¹`, the exact determinant of `b = (t, 1)`, `c = (u, t)`
/// against the origin is `t² − u = 2⁻¹⁰⁴ > 0`. Naive f64 evaluation
/// rounds `t²` to `u`, cancelling to exactly `0.0` — collinear.
/// The disagreement proves the exact wrapper is live (GAIA-004).
#[test]
fn naive_orient_2d_cancellation_reports_collinear_but_exact_is_positive() {
    let t = 1.0 + f64::EPSILON;
    let u = 1.0 + 2.0 * f64::EPSILON;
    let (ax, ay) = (0.0, 0.0);
    let (bx, by) = (t, 1.0);
    let (cx, cy) = (u, t);

    let naive = naive_orient_2d(ax, ay, bx, by, cx, cy);
    assert_eq!(
        naive, 0.0,
        "construction invariant: naive f64 cancels to zero"
    );
    assert_eq!(
        orient_2d(
            &Point2::new(ax, ay),
            &Point2::new(bx, by),
            &Point2::new(cx, cy)
        ),
        Orientation::Positive,
        "exact evaluation must resolve the 2^-104 determinant"
    );
    // Mirrored configuration: same cancellation, opposite exact sign.
    let naive_mirrored = naive_orient_2d(ax, ay, cx, cy, bx, by);
    assert_eq!(naive_mirrored, 0.0);
    assert_eq!(
        orient_2d(
            &Point2::new(ax, ay),
            &Point2::new(cx, cy),
            &Point2::new(bx, by)
        ),
        Orientation::Negative
    );
}

/// The 2-D cancellation embedded as a tetrahedron: `b`, `c` carry the
/// constructed pair in the xy-plane and `d` lifts off by one in z, so
/// the signed volume is the same 2⁻¹⁰⁴ value. Naive evaluation of the
/// cross-product dot cancels; the exact wrapper resolves Positive.
#[test]
fn naive_orient_3d_cancellation_reports_coplanar_but_exact_is_positive() {
    let t = 1.0 + f64::EPSILON;
    let u = 1.0 + 2.0 * f64::EPSILON;
    let a = [0.0, 0.0, 0.0];
    let b = [t, 1.0, 0.0];
    let c = [u, t, 0.0];
    let d = [0.0, 0.0, 1.0];

    let naive = (b[0] * c[1] - b[1] * c[0]) * d[2];
    assert_eq!(
        naive, 0.0,
        "construction invariant: naive f64 cancels to zero"
    );
    assert_eq!(orient_3d(a, b, c, d), Orientation::Positive);
}

/// The cancellation survives scaling across binades: multiplying the
/// constructed pair by 2²⁰⁰ and 2⁻²⁰⁰ keeps every coordinate f64-exact
/// and the exact determinant nonzero (2²⁹⁶ and 2⁻⁵⁰⁴ respectively), while
/// naive evaluation cancels at both scales.
#[test]
fn cancellation_persists_across_binades() {
    for scale in [2.0_f64.powi(200), 2.0_f64.powi(-200)] {
        let t = (1.0 + f64::EPSILON) * scale;
        let u = (1.0 + 2.0 * f64::EPSILON) * scale;
        let (bx, by) = (t, scale);
        let (cx, cy) = (u, t);
        assert_eq!(naive_orient_2d(0.0, 0.0, bx, by, cx, cy), 0.0);
        assert_eq!(
            orient_2d(
                &Point2::new(0.0, 0.0),
                &Point2::new(bx, by),
                &Point2::new(cx, cy)
            ),
            Orientation::Positive,
            "scale {scale:e}"
        );
    }
}

/// Naive in-circle evaluation (cofactor form) for a CCW triangle — the
/// non-robust oracle the exact wrapper must beat.
fn naive_incircle(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, dx: f64, dy: f64) -> f64 {
    let (az, bz, cz, dz) = (
        ax * ax + ay * ay,
        bx * bx + by * by,
        cx * cx + cy * cy,
        dx * dx + dy * dy,
    );
    (az - dz) * (bx - dx) * (cy - dy)
        + (bz - dz) * (cx - dx) * (ay - dy)
        + (cz - dz) * (ax - dx) * (by - dy)
        - (cz - dz) * (bx - dx) * (ay - dy)
        - (bz - dz) * (ax - dx) * (cy - dy)
        - (az - dz) * (cx - dx) * (by - dy)
}

/// Four exactly cocircular integer points — (0,0), (6,0), (0,8), and
/// (6,8) all lie on the circle of center (3,4) and radius 5 — with the
/// fourth displaced tangentially by `(−4δ, +3δ)` for `δ = 2⁻³⁰`. The
/// displacement is tangent to the circle at (6,8), so the linear term
/// cancels exactly and the squared-distance gain is `25δ² > 0`: the
/// point is strictly outside by an exactly derived margin, while naive
/// f64 evaluation cannot separate the configuration from cocircular
/// (its returned determinant sits at its own rounding-noise scale).
/// The disagreement proves the exact wrapper is live (GAIA-004).
#[test]
fn exact_incircle_resolves_a_tangential_margin_naive_cannot_see() {
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(6.0, 0.0);
    let c = Point2::new(0.0, 8.0);
    // abc is CCW: 6·8 > 0.
    assert_eq!(orient_2d(&a, &b, &c), Orientation::Positive);

    // Construction invariant: the unperturbed fourth corner is exactly
    // cocircular — both oracles agree on Degenerate.
    assert_eq!(
        incircle(&a, &b, &c, &Point2::new(6.0, 8.0)),
        Orientation::Degenerate
    );

    let delta = 2.0_f64.powi(-30);
    let d = Point2::new(6.0 - 4.0 * delta, 8.0 + 3.0 * delta);
    // |d − (3,4)|² = 25 + 25δ² > 25: strictly outside, so the CCW
    // convention maps it to Negative.
    assert_eq!(incircle(&a, &b, &c, &d), Orientation::Negative);

    let naive = naive_incircle(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
    // The true determinant is ≈ κ·25δ² ~ 1e-17; the naive cofactor terms
    // are O(5000), so its rounding noise floor is ~2⁻⁴⁰ per term. A
    // returned magnitude at that scale carries no sign information —
    // naive cannot resolve what the exact wrapper just did.
    assert!(
        naive.abs() <= 1e-11,
        "naive f64 sits at its own noise scale (got {naive:e}); the true margin is ~1e-17"
    );
}

/// The unit tetrahedron's circumsphere — center (0.5, 0.5, 0.5),
/// radius² 0.75, both dyadic-exact in f64 — with the cospherical corner
/// (1,1,1) displaced tangentially by `(δ, −δ, 0)` for `δ = 2⁻³⁰`. The
/// displacement is tangent to the sphere at (1,1,1), so the squared-
/// distance gain is exactly `2δ² > 0`: strictly outside by a derived
/// margin, invisible to naive f64 distance evaluation. The unit tet is
/// Shewchuk-negative oriented, so outside maps to Positive through the
/// wrapper, and the center point pins the flipped inside sign as
/// Negative.
#[test]
fn exact_insphere_resolves_a_tangential_margin_naive_cannot_see() {
    let a = [0.0, 0.0, 0.0];
    let b = [1.0, 0.0, 0.0];
    let c = [0.0, 1.0, 0.0];
    let d = [0.0, 0.0, 1.0];
    let center = [0.5, 0.5, 0.5];
    let r_sq = 0.75;

    // Construction invariant: (1,1,1) is exactly cospherical.
    assert_eq!(
        insphere(a, b, c, d, [1.0, 1.0, 1.0]),
        Orientation::Degenerate
    );
    // The center is strictly inside; the tet's negative orientation
    // flips the wrapper's inside sign to Negative.
    assert_eq!(insphere(a, b, c, d, center), Orientation::Negative);

    let delta = 2.0_f64.powi(-30);
    let e = [1.0 + delta, 1.0 - delta, 1.0];
    // |e − center|² = 0.75 + 2δ² > 0.75: strictly outside → Positive.
    assert_eq!(insphere(a, b, c, d, e), Orientation::Positive);

    // Naive circumcenter comparison: (0.5 ± δ)² rounds away the δ²
    // terms, so the distance difference cancels to exactly zero.
    let naive = (e[0] - center[0]) * (e[0] - center[0])
        + (e[1] - center[1]) * (e[1] - center[1])
        + (e[2] - center[2]) * (e[2] - center[2])
        - r_sq;
    assert_eq!(
        naive, 0.0,
        "construction invariant: naive f64 rounds the 2δ² margin away"
    );
}

// ── GAIA-004: permutation antisymmetry ─────────────────────────────────

/// The orientation determinant is alternating: for every permutation `π`
/// of `(a, b, c, d)`, `orient_3d(π) = sgn(π) · orient_3d(a, b, c, d)`,
/// and a degenerate configuration stays Degenerate under every
/// permutation. Verified exhaustively over all 24 permutations for a
/// generic and a coplanar tetrahedron.
#[test]
fn orient_3d_is_alternating_under_all_permutations() {
    // (permutation of (a,b,c,d) positions, parity)
    const PERMS: [([usize; 4], bool); 24] = [
        ([0, 1, 2, 3], true),
        ([0, 1, 3, 2], false),
        ([0, 2, 1, 3], false),
        ([0, 2, 3, 1], true),
        ([0, 3, 1, 2], true),
        ([0, 3, 2, 1], false),
        ([1, 0, 2, 3], false),
        ([1, 0, 3, 2], true),
        ([1, 2, 0, 3], true),
        ([1, 2, 3, 0], false),
        ([1, 3, 0, 2], false),
        ([1, 3, 2, 0], true),
        ([2, 0, 1, 3], true),
        ([2, 0, 3, 1], false),
        ([2, 1, 0, 3], false),
        ([2, 1, 3, 0], true),
        ([2, 3, 0, 1], true),
        ([2, 3, 1, 0], false),
        ([3, 0, 1, 2], false),
        ([3, 0, 2, 1], true),
        ([3, 1, 0, 2], true),
        ([3, 1, 2, 0], false),
        ([3, 2, 0, 1], false),
        ([3, 2, 1, 0], true),
    ];
    let generic = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.25, 0.25, 1.0],
    ];
    let coplanar = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ];
    for points in [generic, coplanar] {
        let base = orient_3d(points[0], points[1], points[2], points[3]);
        for (order, even) in PERMS {
            let permuted = [
                points[order[0]],
                points[order[1]],
                points[order[2]],
                points[order[3]],
            ];
            let got = orient_3d(permuted[0], permuted[1], permuted[2], permuted[3]);
            let flip = |o: Orientation| match o {
                Orientation::Positive => Orientation::Negative,
                Orientation::Negative => Orientation::Positive,
                Orientation::Degenerate => Orientation::Degenerate,
            };
            let expected = if even { base } else { flip(base) };
            assert_eq!(got, expected, "permutation {order:?} broke antisymmetry");
        }
    }
}

// ── GAIA-004: from_det mapping ───────────────────────────────────────────

#[test]
fn from_det_maps_boundary_values() {
    assert_eq!(Orientation::from_det(0.0), Orientation::Degenerate);
    assert_eq!(Orientation::from_det(-0.0), Orientation::Degenerate);
    assert_eq!(
        Orientation::from_det(f64::MIN_POSITIVE),
        Orientation::Positive
    );
    assert_eq!(
        Orientation::from_det(-f64::MIN_POSITIVE),
        Orientation::Negative
    );
}
