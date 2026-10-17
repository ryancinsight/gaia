//! Exact finite projection-range contracts on the unchanged renderer.

use gaia::application::render::transform::{perspective, to_homogeneous, to_ndc, transform, Vec4};
use gaia::domain::core::scalar::Point3r;

fn verify_depth_endpoints(near: f64) {
    let far = near + near;
    let matrix = perspective(core::f64::consts::FRAC_PI_2, 1.0, near, far)
        .expect("finite, positive, ordered frustum");
    for (depth, expected_z) in [(near, -1.0), (far, 1.0)] {
        let clip = transform(&matrix, &to_homogeneous(&Point3r::new(0.0, 0.0, -depth)));
        assert_eq!(
            clip[3].to_bits(),
            depth.to_bits(),
            "documented eye-space depth"
        );
        let ndc = to_ndc(&clip).expect("finite nonzero depth");
        assert_eq!((ndc.x, ndc.y, ndc.z), (0.0, 0.0, expected_z));
    }
}

#[test]
fn ordinary_dyadic_frustum_maps_both_depth_endpoints() {
    verify_depth_endpoints(1.0);
}

#[test]
fn finite_large_dyadic_frustum_maps_both_depth_endpoints() {
    // Canonical depth coefficients are exactly -3 and -2^514, both finite.
    // Only the unscaled intermediate 2*far*near exceeds the scalar range.
    verify_depth_endpoints(2.0_f64.powi(512));
}

#[test]
fn ordinary_perspective_divide_retains_values_and_zero_rejection() {
    let v = to_ndc(&Vec4::new([2.0, 4.0, 0.0, 2.0])).expect("nonzero w");
    assert_eq!((v.x, v.y, v.z), (1.0, 2.0, 0.0));
    assert!(to_ndc(&Vec4::new([1.0, 2.0, 3.0, 0.0])).is_none());
}

#[test]
fn nonzero_subnormal_perspective_divide_retains_representable_values() {
    let w = f64::from_bits(1);
    let v = to_ndc(&Vec4::new([w, w + w, 0.0, w])).expect("finite nonzero w");
    assert_eq!((v.x, v.y, v.z), (1.0, 2.0, 0.0));
}

#[test]
fn dyadic_frusta_retain_representable_depth_coefficients_across_exponents() {
    // For far = 2*near, the exact coefficients are -3 and -4*near.
    // Check coefficients separately: matrix-vector products can themselves
    // overflow at depths near the largest finite values.
    let mut near = f64::from_bits(1);
    while near <= 2.0_f64.powi(1020) {
        let m = perspective(core::f64::consts::FRAC_PI_2, 1.0, near, near + near)
            .expect("finite ordered dyadic frustum");
        assert_eq!(m[(2, 2)].to_bits(), (-3.0_f64).to_bits(), "near={near:e}");
        assert_eq!(
            m[(2, 3)].to_bits(),
            (-4.0 * near).to_bits(),
            "near={near:e}"
        );
        verify_depth_endpoints(near);
        near += near;
    }
}

#[test]
fn small_w_division_preserves_exact_values_and_signed_zero() {
    for bits in [1, 2, 3, 31, 1 << 20, 1 << 48] {
        for sign in [1.0, -1.0] {
            let w = f64::from_bits(bits) * sign;
            let clip = Vec4::new([w, w + w, -0.0, w]);
            let ndc = to_ndc(&clip).expect("nonzero finite denominator");
            assert_eq!(ndc.x.to_bits(), (1.0_f64).to_bits());
            assert_eq!(ndc.y.to_bits(), (2.0_f64).to_bits());
            assert_eq!(ndc.z.to_bits(), (-0.0 / w).to_bits());
        }
    }
}

#[test]
fn ordinary_projection_coefficients_keep_original_bits() {
    for near in [0.001, 0.1, 1.0, 3.0, 128.0] {
        for ratio in [1.5, 2.0, 13.0, 1000.0] {
            let far = near * ratio;
            let range = 1.0 / (near - far);
            let m = perspective(1.25, 1.5, near, far).expect("ordinary frustum");
            assert_eq!(m[(2, 2)].to_bits(), ((far + near) * range).to_bits());
            assert_eq!(m[(2, 3)].to_bits(), (2.0 * far * near * range).to_bits());
        }
    }
}

#[test]
fn ordinary_ndc_and_nonfinite_components_keep_original_results() {
    for w in [
        0.25,
        3.0,
        -7.0,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE * 0.5,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        let clip = Vec4::new([1.5, -0.0, -9.0, w]);
        let ndc = to_ndc(&clip).expect("only zero w is rejected");
        for (actual, input) in [(ndc.x, clip[0]), (ndc.y, clip[1]), (ndc.z, clip[2])] {
            let expected = input * (1.0 / w);
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
    for invalid in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let w = f64::from_bits(1);
        let ndc = to_ndc(&Vec4::new([invalid, 0.0, w, w])).expect("nonzero w");
        assert!(
            ndc.y.is_nan(),
            "preserve the original nonfinite-vector result"
        );
        assert_eq!(ndc.z.to_bits(), f64::INFINITY.to_bits());
    }
    for zero in [0.0, -0.0] {
        assert!(to_ndc(&Vec4::new([1.0, 2.0, 3.0, zero])).is_none());
    }
}

#[test]
fn perspective_preserves_existing_invalid_input_policy() {
    let fov = core::f64::consts::FRAC_PI_2;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(perspective(invalid, 1.0, 1.0, 2.0).is_none());
        assert!(perspective(fov, 1.0, invalid, 2.0).is_none());
        assert!(perspective(fov, 1.0, 1.0, invalid).is_none());
    }
    for aspect in [f64::NAN, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(perspective(fov, aspect, 1.0, 2.0).is_none());
    }
    let m = perspective(fov, f64::INFINITY, 1.0, 2.0)
        .expect("infinite aspect remains accepted by the existing policy");
    assert_eq!(m[(0, 0)].to_bits(), (0.0_f64).to_bits());
    assert_eq!(m[(2, 2)].to_bits(), (-3.0_f64).to_bits());
    assert_eq!(m[(2, 3)].to_bits(), (-4.0_f64).to_bits());
}
