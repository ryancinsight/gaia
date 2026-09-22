//! Clip-space transforms for the software renderer.
//!
//! Matrices are [`leto::FixedMatrix`] values and vectors are
//! [`leto::FixedVector`] values: this module adds no linear-algebra type of its
//! own, only the two constructions a renderer needs (`perspective`, `look_at`)
//! and the conventions that go with them.
//!
//! ## Conventions
//!
//! - Matrices are **row-major** and act on **column** vectors, so a point is
//!   transformed as `m * v`.
//! - The camera looks down its own `-z` axis in eye space, which is what makes
//!   `w` in clip space equal to the eye-space distance in front of the camera.
//!   The renderer relies on that: it uses `1/w` as the depth key, so a larger
//!   `1/w` is nearer. See [`crate::application::render::raster`].
//! - Clip space is the OpenGL convention: `x`, `y` and `z` in `[-w, w]`, and a
//!   point is in front of the camera exactly when `w > 0`.

use leto::{FixedMatrix, FixedVector};

use crate::domain::core::scalar::{Point3r, Real, Vector3r};

/// A 4x4 homogeneous transform.
pub type Mat4 = FixedMatrix<Real, 4, 4>;

/// A 4-component homogeneous vector.
pub type Vec4 = FixedVector<Real, 4>;

/// Lift a position into homogeneous coordinates with `w = 1`.
#[must_use]
pub fn to_homogeneous(p: &Point3r) -> Vec4 {
    Vec4::new([p.x, p.y, p.z, 1.0])
}

/// Apply `m` to a homogeneous vector.
#[must_use]
pub fn transform(m: &Mat4, v: &Vec4) -> Vec4 {
    *m * *v
}

/// Apply the rotation/translation part of `m` to a direction (that is, with
/// `w = 0`, so the translation column does not apply).
#[must_use]
pub fn transform_direction(m: &Mat4, d: &Vector3r) -> Vector3r {
    let v = *m * Vec4::new([d.x, d.y, d.z, 0.0]);
    Vector3r::new(v[0], v[1], v[2])
}

/// Build a perspective projection matrix.
///
/// `fov_y` is the full vertical field of view in radians, `aspect` is
/// `width / height`, and `near`/`far` are positive eye-space distances. Points
/// at eye-space depth `d` land at `w = d`, so a nearer point has a larger `1/w`.
///
/// Returns `None` when the frustum is degenerate — a non-positive or
/// non-finite `near`/`far`, `far <= near`, an empty field of view, or a
/// non-positive `aspect`. A viewer resizes its surface on user input, so these
/// are reachable values, not programmer errors, and a caller that receives
/// `None` should keep the previous camera rather than render garbage.
#[must_use]
pub fn perspective(fov_y: Real, aspect: Real, near: Real, far: Real) -> Option<Mat4> {
    // Each guard names its rejection positively and tests `is_nan` explicitly.
    // A negated comparison would read shorter but is *also* false for `NaN`,
    // which would silently admit a `NaN` field of view.
    if fov_y.is_nan() || fov_y <= 0.0 || fov_y >= core::f64::consts::PI {
        return None;
    }
    if aspect.is_nan() || aspect <= 0.0 {
        return None;
    }
    if near.is_nan() || near <= 0.0 || far.is_nan() || far <= near || !far.is_finite() {
        return None;
    }
    let f = 1.0 / (fov_y * 0.5).tan();
    if !f.is_finite() {
        return None;
    }
    let range = 1.0 / (near - far);
    Some(Mat4::from_rows([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * range, 2.0 * far * near * range],
        [0.0, 0.0, -1.0, 0.0],
    ]))
}

/// Build a right-handed view matrix looking from `eye` at `target`.
///
/// Returns `None` when the view direction is degenerate (eye and target
/// coincide) or parallel to `up`, because the basis is then undefined.
#[must_use]
pub fn look_at(eye: &Point3r, target: &Point3r, up: &Vector3r) -> Option<Mat4> {
    let forward = (target - eye).normalize();
    if !forward.x.is_finite() || !forward.y.is_finite() || !forward.z.is_finite() {
        return None;
    }
    let right = forward.cross(*up).normalize();
    if !right.x.is_finite() || !right.y.is_finite() || !right.z.is_finite() {
        return None;
    }
    let true_up = right.cross(forward);
    let eye_coords = eye.coords;
    Some(Mat4::from_rows([
        [right.x, right.y, right.z, -right.dot(eye_coords)],
        [true_up.x, true_up.y, true_up.z, -true_up.dot(eye_coords)],
        [-forward.x, -forward.y, -forward.z, forward.dot(eye_coords)],
        [0.0, 0.0, 0.0, 1.0],
    ]))
}

/// Perspective divide: clip space to normalised device coordinates.
///
/// Returns `None` for a point with `w == 0`, which has no projection.
#[must_use]
pub fn to_ndc(v: &Vec4) -> Option<Vector3r> {
    let w = v[3];
    if w == 0.0 {
        return None;
    }
    let inv = 1.0 / w;
    Some(Vector3r::new(v[0] * inv, v[1] * inv, v[2] * inv))
}

/// Linear blend of two homogeneous vectors.
#[must_use]
pub fn lerp4(a: &Vec4, b: &Vec4, t: Real) -> Vec4 {
    Vec4::new([
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye() -> Point3r {
        Point3r::new(0.0, 0.0, 0.0)
    }

    fn z_up() -> Vector3r {
        Vector3r::new(0.0, 0.0, 1.0)
    }

    fn y_up() -> Vector3r {
        Vector3r::new(0.0, 1.0, 0.0)
    }

    #[test]
    fn to_homogeneous_lifts_with_w_one() {
        let v = to_homogeneous(&Point3r::new(1.0, 2.0, 3.0));
        assert_eq!(v.as_array(), &[1.0, 2.0, 3.0, 1.0]);
    }

    /// The renderer's depth key is `1/w`, so `w` must equal the eye-space
    /// distance in front of the camera. If this ever changes, depth ordering
    /// silently inverts and nothing else in the test suite would notice.
    #[test]
    fn perspective_w_is_eye_space_depth() {
        let m = perspective(core::f64::consts::FRAC_PI_2, 1.0, 0.1, 100.0).expect("valid frustum");
        for depth in [0.5_f64, 1.0, 7.5, 99.0] {
            let clip = transform(&m, &to_homogeneous(&Point3r::new(0.0, 0.0, -depth)));
            assert!(
                (clip[3] - depth).abs() < 1e-12,
                "w should be the eye-space depth {depth}, got {}",
                clip[3]
            );
        }
    }

    #[test]
    fn perspective_rejects_degenerate_frusta() {
        let ok = core::f64::consts::FRAC_PI_2;
        assert!(perspective(ok, 1.0, 0.1, 100.0).is_some());
        assert!(perspective(0.0, 1.0, 0.1, 100.0).is_none(), "empty fov");
        assert!(perspective(ok, 0.0, 0.1, 100.0).is_none(), "zero aspect");
        assert!(perspective(ok, 1.0, 0.0, 100.0).is_none(), "zero near");
        assert!(perspective(ok, 1.0, 0.1, 0.1).is_none(), "far == near");
        assert!(perspective(ok, 1.0, 0.1, 0.05).is_none(), "far < near");
        assert!(
            perspective(ok, 1.0, 0.1, f64::INFINITY).is_none(),
            "infinite far"
        );
    }

    /// A `look_at` basis is orthonormal and puts the target on the view axis.
    #[test]
    fn look_at_basis_is_orthonormal_and_centres_the_target() {
        let view = look_at(&Point3r::new(3.0, -4.0, 5.0), &eye(), &z_up()).expect("valid view");
        let target = transform(&view, &to_homogeneous(&eye()));
        assert!(target[0].abs() < 1e-12, "target x should be 0");
        assert!(target[1].abs() < 1e-12, "target y should be 0");
        // Eye space looks down -z, so the target sits at negative z.
        assert!(target[2] < 0.0, "target should be in front of the camera");
        let d = (3.0_f64 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0).sqrt();
        assert!(
            (target[2] + d).abs() < 1e-12,
            "target depth should be -|eye|"
        );
    }

    #[test]
    fn look_at_rejects_a_degenerate_basis() {
        assert!(
            look_at(&eye(), &eye(), &z_up()).is_none(),
            "eye == target has no view direction"
        );
        assert!(
            look_at(&Point3r::new(0.0, 0.0, 5.0), &eye(), &z_up()).is_none(),
            "view direction parallel to up has no right vector"
        );
    }

    #[test]
    fn to_ndc_divides_by_w_and_rejects_zero() {
        let ndc = to_ndc(&Vec4::new([2.0, -4.0, 1.0, 2.0])).expect("w != 0");
        assert_eq!((ndc.x, ndc.y, ndc.z), (1.0, -2.0, 0.5));
        assert!(to_ndc(&Vec4::new([1.0, 1.0, 1.0, 0.0])).is_none());
    }

    #[test]
    fn lerp4_hits_both_endpoints() {
        let a = Vec4::new([0.0, 0.0, 0.0, 1.0]);
        let b = Vec4::new([2.0, 4.0, 6.0, 3.0]);
        assert_eq!(lerp4(&a, &b, 0.0).as_array(), a.as_array());
        assert_eq!(lerp4(&a, &b, 1.0).as_array(), b.as_array());
        assert_eq!(lerp4(&a, &b, 0.5).as_array(), &[1.0, 2.0, 3.0, 2.0]);
    }

    /// A direction is not translated, only rotated.
    #[test]
    fn transform_direction_ignores_translation() {
        // `up` must not be parallel to the view direction, so this looks down
        // `-z` with `+y` up rather than `+z` up, which `look_at` rejects as
        // degenerate.
        let view = look_at(&Point3r::new(0.0, 0.0, 10.0), &eye(), &y_up()).expect("valid view");
        let dir = transform_direction(&view, &Vector3r::new(0.0, 0.0, -1.0));
        // Looking down -z from +z, the world's -z is the camera's forward.
        assert!(dir.z < 0.0);
        let len = dir.norm();
        assert!((len - 1.0).abs() < 1e-12, "a rotation preserves length");
    }
}
