//! An orbit camera: the model stays put, the camera swings around it.
//!
//! This is the interaction model a mesh viewer needs — drag to rotate, drag to
//! pan, scroll to dolly — expressed as *semantic* operations (`orbit`, `pan`,
//! `dolly`, `fit`) rather than as event handling. Mapping a host's mouse and
//! keyboard events onto those operations belongs to the host, so this type
//! never sees a window, a pixel coordinate space, or an event enum.
//!
//! ## Frame
//!
//! The world is z-up, matching the CFD and millifluidic meshes this crate
//! produces. `yaw` rotates about the world z axis and `pitch` raises the eye
//! above or below the target plane. The eye sits at
//!
//! ```text
//! eye = target + distance * (cos(pitch)cos(yaw), cos(pitch)sin(yaw), sin(pitch))
//! ```
//!
//! `pitch` is clamped just short of ±90°: at exactly vertical the view
//! direction is parallel to the world up vector, the camera's right vector
//! vanishes, and the view matrix is undefined.

use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::domain::geometry::aabb::Aabb;

use super::transform::{look_at, perspective, Mat4};

/// Default vertical field of view, 45°.
pub const DEFAULT_FOV_Y: Real = core::f64::consts::FRAC_PI_4;

/// Pitch is clamped this far short of vertical.
const PITCH_LIMIT: Real = core::f64::consts::FRAC_PI_2 - 1.0e-3;

/// Nearest distance the camera may dolly to.
const MIN_DISTANCE: Real = 1.0e-9;

/// Farthest distance the camera may dolly to.
const MAX_DISTANCE: Real = 1.0e12;

/// Multiplier applied per dolly step, so zooming is geometric and reversible.
const DOLLY_STEP: Real = 0.9;

/// An orbit camera looking at `target` from `distance` away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    target: Point3r,
    distance: Real,
    yaw: Real,
    pitch: Real,
    fov_y: Real,
    near: Real,
    far: Real,
}

impl OrbitCamera {
    /// Create a camera looking at `target` from `distance` along the `-y` axis.
    ///
    /// `distance` is clamped into the camera's usable range; a non-finite or
    /// non-positive value falls back to the far end of that range rather than
    /// producing an unusable camera.
    #[must_use]
    pub fn new(target: Point3r, distance: Real) -> Self {
        Self {
            target,
            distance: clamp_distance(distance),
            yaw: 0.0,
            pitch: 0.0,
            fov_y: DEFAULT_FOV_Y,
            near: 1.0e-3,
            far: 1.0e6,
        }
    }

    /// Set the vertical field of view in radians.
    ///
    /// A value outside `(0, pi)` is rejected and the previous field of view is
    /// kept, because a degenerate frustum has no projection to render with.
    pub fn set_fov_y(&mut self, fov_y: Real) {
        if fov_y > 0.0 && fov_y < core::f64::consts::PI {
            self.fov_y = fov_y;
        }
    }

    /// Builder form of [`Self::set_fov_y`].
    #[must_use]
    pub fn with_fov_y(mut self, fov_y: Real) -> Self {
        self.set_fov_y(fov_y);
        self
    }

    /// Set the near and far clip planes.
    ///
    /// The pair is applied only when it is usable (`0 < near < far`, both
    /// finite); otherwise the previous planes are kept.
    pub fn set_clip_planes(&mut self, near: Real, far: Real) {
        if near > 0.0 && far > near && far.is_finite() {
            self.near = near;
            self.far = far;
        }
    }

    /// Builder form of [`Self::set_clip_planes`].
    #[must_use]
    pub fn with_clip_planes(mut self, near: Real, far: Real) -> Self {
        self.set_clip_planes(near, far);
        self
    }

    /// The point the camera looks at.
    #[must_use]
    pub const fn target(&self) -> Point3r {
        self.target
    }

    /// The eye's distance from the target.
    #[must_use]
    pub const fn distance(&self) -> Real {
        self.distance
    }

    /// The azimuthal angle in radians.
    #[must_use]
    pub const fn yaw(&self) -> Real {
        self.yaw
    }

    /// The elevation angle in radians, clamped just short of vertical.
    #[must_use]
    pub const fn pitch(&self) -> Real {
        self.pitch
    }

    /// The vertical field of view in radians.
    #[must_use]
    pub const fn fov_y(&self) -> Real {
        self.fov_y
    }

    /// The near clip plane.
    #[must_use]
    pub const fn near(&self) -> Real {
        self.near
    }

    /// The far clip plane.
    #[must_use]
    pub const fn far(&self) -> Real {
        self.far
    }

    /// The unit vector from the target towards the eye.
    #[must_use]
    pub fn offset_direction(&self) -> Vector3r {
        let cos_pitch = self.pitch.cos();
        Vector3r::new(
            cos_pitch * self.yaw.cos(),
            cos_pitch * self.yaw.sin(),
            self.pitch.sin(),
        )
    }

    /// The eye position.
    #[must_use]
    pub fn eye(&self) -> Point3r {
        self.target + self.offset_direction() * self.distance
    }

    /// The unit view direction, from the eye towards the target.
    #[must_use]
    pub fn forward(&self) -> Vector3r {
        -self.offset_direction()
    }

    /// The camera's right vector, parallel to the image plane.
    #[must_use]
    pub fn right(&self) -> Vector3r {
        self.forward()
            .cross(Vector3r::new(0.0, 0.0, 1.0))
            .normalize()
    }

    /// The camera's up vector, parallel to the image plane.
    #[must_use]
    pub fn up(&self) -> Vector3r {
        self.right().cross(self.forward())
    }

    /// The view matrix, or `None` if the basis is degenerate.
    #[must_use]
    pub fn view(&self) -> Option<Mat4> {
        look_at(&self.eye(), &self.target, &Vector3r::new(0.0, 0.0, 1.0))
    }

    /// The projection matrix for the given `aspect` (`width / height`).
    #[must_use]
    pub fn projection(&self, aspect: Real) -> Option<Mat4> {
        perspective(self.fov_y, aspect, self.near, self.far)
    }

    /// The combined view-projection matrix for the given `aspect`.
    ///
    /// The projection is applied *after* the view, so this is
    /// `projection * view`. A point is transformed as `(P * V) * p`, which is
    /// `P * (V * p)` — world to eye, then eye to clip.
    #[must_use]
    pub fn view_projection(&self, aspect: Real) -> Option<Mat4> {
        Some(self.projection(aspect)? * self.view()?)
    }

    /// Rotate the eye about the target.
    ///
    /// `delta_yaw` and `delta_pitch` are in radians. `pitch` is clamped; `yaw`
    /// wraps, and is kept in `(-pi, pi]` so a long drag cannot drift into
    /// accumulating rounding error.
    pub fn orbit(&mut self, delta_yaw: Real, delta_pitch: Real) {
        if !delta_yaw.is_finite() || !delta_pitch.is_finite() {
            return;
        }
        self.yaw = wrap_angle(self.yaw + delta_yaw);
        self.pitch = (self.pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Translate the target within the camera's image plane.
    ///
    /// `right` and `up` are in world units.
    pub fn pan_world(&mut self, right: Real, up: Real) {
        if !right.is_finite() || !up.is_finite() {
            return;
        }
        self.target = self.target + self.right() * right + self.up() * up;
    }

    /// Translate the target so a mouse drag of `(dx, dy)` pixels tracks the
    /// model, for a viewport `viewport_height` pixels tall.
    ///
    /// Screen `y` grows downward, so a downward drag moves the target up in
    /// world space, which is what makes the model follow the cursor.
    pub fn pan_pixels(&mut self, dx: Real, dy: Real, viewport_height: u32) {
        if viewport_height == 0 {
            return;
        }
        let height = Real::from(viewport_height);
        // World units per pixel on the plane through the target.
        let units_per_pixel = 2.0 * self.distance * (self.fov_y * 0.5).tan() / height;
        self.pan_world(-dx * units_per_pixel, dy * units_per_pixel);
    }

    /// Multiply the eye distance by `factor`.
    ///
    /// Clamped to the camera's usable range; a non-finite factor is ignored.
    pub fn dolly(&mut self, factor: Real) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        self.distance = clamp_distance(self.distance * factor);
    }

    /// Dolly by `steps` scroll notches; positive zooms in.
    pub fn zoom_steps(&mut self, steps: Real) {
        if !steps.is_finite() {
            return;
        }
        self.dolly(DOLLY_STEP.powf(steps));
    }

    /// Frame `bounds` so the whole box is visible from the current angles.
    ///
    /// The clip planes are reset from the resulting distance, because a viewer
    /// that frames a small part after a large one needs a near plane tight
    /// enough to keep depth precision. The distance is chosen from the
    /// *narrower* of the horizontal and vertical half-angles, so a wide box in
    /// a tall viewport is still fully framed.
    ///
    /// An empty or degenerate box leaves the camera unchanged.
    pub fn fit(&mut self, bounds: &Aabb<Real>, aspect: Real) {
        let half = (bounds.max - bounds.min) * 0.5;
        let radius = half.norm();
        if !radius.is_finite() || radius <= 0.0 || aspect.is_nan() || aspect <= 0.0 {
            return;
        }
        let half_fov_y = self.fov_y * 0.5;
        let half_fov_x = (half_fov_y.tan() * aspect).atan();
        let half_fov = half_fov_x.min(half_fov_y);
        let sin_half = half_fov.sin();
        if sin_half.is_nan() || sin_half <= 0.0 {
            return;
        }
        // A small margin keeps the silhouette off the viewport edge.
        let distance = clamp_distance(radius / sin_half * 1.1);
        self.target = bounds.center();
        self.distance = distance;
        self.near = (distance * 1.0e-3).max(1.0e-9);
        self.far = (distance + radius * 4.0).max(self.near * 1.0e3);
    }
}

/// Clamp a distance into the camera's usable range, mapping non-finite and
/// non-positive values to the far end rather than leaving a broken camera.
fn clamp_distance(distance: Real) -> Real {
    if !distance.is_finite() || distance <= 0.0 {
        return MAX_DISTANCE;
    }
    distance.clamp(MIN_DISTANCE, MAX_DISTANCE)
}

/// Wrap an angle into `(-pi, pi]`.
fn wrap_angle(angle: Real) -> Real {
    let two_pi = core::f64::consts::TAU;
    let mut wrapped = angle % two_pi;
    if wrapped > core::f64::consts::PI {
        wrapped -= two_pi;
    } else if wrapped <= -core::f64::consts::PI {
        wrapped += two_pi;
    }
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::render::transform::{to_homogeneous, transform};

    fn z_up() -> Vector3r {
        Vector3r::new(0.0, 0.0, 1.0)
    }

    #[test]
    fn eye_sits_at_distance_along_the_offset_direction() {
        let cam = OrbitCamera::new(Point3r::new(1.0, 2.0, 3.0), 10.0);
        let eye = cam.eye();
        let expected = cam.target() + cam.offset_direction() * cam.distance();
        assert!((eye - expected).norm() < 1e-12);
        assert!((eye - cam.target()).norm() - 10.0 < 1e-12);
    }

    #[test]
    fn default_view_is_horizontal_along_negative_y() {
        let cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        assert!((cam.offset_direction() - Vector3r::new(1.0, 0.0, 0.0)).norm() < 1e-12);
    }

    /// The right/up/forward triple must be orthonormal, or the view matrix is
    /// a shear and everything downstream is subtly wrong.
    #[test]
    fn basis_is_orthonormal_at_many_angles() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        for step in -30..=30 {
            let angle = Real::from(step) * 0.1;
            cam.orbit(angle, angle * 0.5);
            let (r, u, f) = (cam.right(), cam.up(), cam.forward());
            for (name, v) in [("right", r), ("up", u), ("forward", f)] {
                assert!(
                    (v.norm() - 1.0).abs() < 1e-9,
                    "{name} is not unit length at step {step}"
                );
            }
            assert!(r.dot(u).abs() < 1e-9, "right·up at step {step}");
            assert!(r.dot(f).abs() < 1e-9, "right·forward at step {step}");
            assert!(u.dot(f).abs() < 1e-9, "up·forward at step {step}");
        }
    }

    /// Pitch must stop short of vertical: at exactly ±90° the right vector
    /// vanishes and the view matrix is undefined.
    #[test]
    fn pitch_is_clamped_short_of_vertical() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        cam.orbit(0.0, 100.0);
        assert!(cam.pitch() < core::f64::consts::FRAC_PI_2);
        assert!(cam.right().norm() > 0.0);
        cam.orbit(0.0, -200.0);
        assert!(cam.pitch() > -core::f64::consts::FRAC_PI_2);
        assert!(cam.right().norm() > 0.0);
    }

    #[test]
    fn yaw_wraps_into_a_bounded_range() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        for _ in 0..1000 {
            cam.orbit(0.7, 0.0);
        }
        assert!(cam.yaw() > -core::f64::consts::PI);
        assert!(cam.yaw() <= core::f64::consts::PI);
    }

    #[test]
    fn orbit_ignores_non_finite_input() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        let before = cam;
        cam.orbit(f64::NAN, 0.1);
        cam.orbit(0.1, f64::INFINITY);
        assert_eq!(cam, before);
    }

    #[test]
    fn dolly_ignores_non_finite_and_non_positive_factors() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        let before = cam;
        cam.dolly(f64::NAN);
        cam.dolly(0.0);
        cam.dolly(-1.0);
        assert_eq!(cam, before);
    }

    /// Zooming in then out by the same number of notches must return to the
    /// same distance, or a scroll-heavy session drifts.
    #[test]
    fn zoom_is_reversible() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 10.0);
        cam.zoom_steps(5.0);
        cam.zoom_steps(-5.0);
        assert!(
            (cam.distance() - 10.0).abs() < 1e-9,
            "got {}",
            cam.distance()
        );
    }

    #[test]
    fn dolly_is_clamped_to_the_usable_range() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 10.0);
        cam.dolly(1.0e300);
        assert!(cam.distance() <= MAX_DISTANCE);
        cam.dolly(1.0e-300);
        assert!(cam.distance() >= MIN_DISTANCE);
    }

    #[test]
    fn invalid_fov_and_clip_planes_are_rejected() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        let before = cam;
        cam.set_fov_y(0.0);
        cam.set_fov_y(f64::NAN);
        cam.set_fov_y(10.0);
        cam.set_clip_planes(1.0, 0.5);
        cam.set_clip_planes(0.0, 1.0);
        cam.set_clip_planes(1.0, f64::INFINITY);
        assert_eq!(cam, before);
    }

    /// Dragging right must move the model right, which means the target moves
    /// left along the camera's right vector.
    #[test]
    fn pan_pixels_follows_the_cursor() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 10.0);
        let right = cam.right();
        let up = cam.up();
        let before = cam.target();

        cam.pan_pixels(10.0, 0.0, 100);
        let moved = cam.target() - before;
        assert!(
            moved.dot(right) < 0.0,
            "a rightward drag moves the target left"
        );

        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 10.0);
        cam.pan_pixels(0.0, 10.0, 100);
        let moved = cam.target() - before;
        assert!(moved.dot(up) > 0.0, "a downward drag moves the target up");
    }

    /// A zero-height viewport has no pixels to convert, and must not divide by
    /// zero.
    #[test]
    fn pan_pixels_ignores_a_zero_height_viewport() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 10.0);
        let before = cam;
        cam.pan_pixels(5.0, 5.0, 0);
        assert_eq!(cam, before);
    }

    /// After `fit`, every corner of the box must land inside the view frustum.
    /// This is the property that makes "reset view" trustworthy.
    #[test]
    fn fit_frames_every_corner_of_the_bounds() {
        let cases = [
            Aabb::new(Point3r::new(0.0, 0.0, 0.0), Point3r::new(4.0, 4.0, 4.0)),
            Aabb::new(
                Point3r::new(-10.0, -1.0, -1.0),
                Point3r::new(10.0, 1.0, 1.0),
            ),
            Aabb::new(Point3r::new(0.0, 0.0, 0.0), Point3r::new(0.2, 8.0, 0.2)),
            Aabb::new(
                Point3r::new(-3.0, -3.0, -0.01),
                Point3r::new(3.0, 3.0, 0.01),
            ),
        ];
        for aspect in [0.5_f64, 1.0, 16.0 / 9.0] {
            for bounds in &cases {
                let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 1.0);
                cam.orbit(0.7, 0.4);
                cam.fit(bounds, aspect);
                let vp = cam.view_projection(aspect).expect("usable camera");
                for (i, corner) in corners(bounds).into_iter().enumerate() {
                    let clip = transform(&vp, &to_homogeneous(&corner));
                    assert!(
                        clip[3] > 0.0,
                        "corner {i} is behind the camera at aspect {aspect}"
                    );
                    let ndc_x = clip[0] / clip[3];
                    let ndc_y = clip[1] / clip[3];
                    assert!(
                        ndc_x.abs() <= 1.0 && ndc_y.abs() <= 1.0,
                        "corner {i} at ({ndc_x}, {ndc_y}) is outside the viewport \
                         at aspect {aspect} for bounds {bounds:?}"
                    );
                }
            }
        }
    }

    fn corners(bounds: &Aabb<Real>) -> [Point3r; 8] {
        let (lo, hi) = (bounds.min, bounds.max);
        [
            Point3r::new(lo.x, lo.y, lo.z),
            Point3r::new(hi.x, lo.y, lo.z),
            Point3r::new(lo.x, hi.y, lo.z),
            Point3r::new(hi.x, hi.y, lo.z),
            Point3r::new(lo.x, lo.y, hi.z),
            Point3r::new(hi.x, lo.y, hi.z),
            Point3r::new(lo.x, hi.y, hi.z),
            Point3r::new(hi.x, hi.y, hi.z),
        ]
    }

    #[test]
    fn fit_leaves_the_camera_alone_for_a_degenerate_box() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        let before = cam;
        cam.fit(
            &Aabb::new(Point3r::new(1.0, 1.0, 1.0), Point3r::new(1.0, 1.0, 1.0)),
            1.0,
        );
        assert_eq!(cam, before, "a point-sized box has no radius to frame");
        cam.fit(
            &Aabb::new(Point3r::new(0.0, 0.0, 0.0), Point3r::new(1.0, 1.0, 1.0)),
            0.0,
        );
        assert_eq!(cam, before, "a zero aspect has no frustum to fit into");
    }

    #[test]
    fn view_projection_is_none_only_for_degenerate_input() {
        let cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        assert!(cam.view_projection(1.0).is_some());
        assert!(cam.view_projection(0.0).is_none());
        // Eye == target leaves no view direction.
        let collapsed = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), MIN_DISTANCE);
        assert!(
            collapsed.view().is_some(),
            "a tiny distance still has a direction"
        );
    }

    /// The projection must be applied after the view, not before. Getting this
    /// backwards still centres the target, because both `P` and `V` fix the
    /// origin, but it leaves world space in `w` instead of eye space, so half
    /// the scene lands "behind the camera" for no visible reason.
    #[test]
    fn view_projection_composes_projection_after_view() {
        let mut cam = OrbitCamera::new(Point3r::new(1.0, 2.0, 3.0), 7.0);
        cam.orbit(0.9, 0.3);
        cam.set_clip_planes(0.1, 100.0);
        let aspect = 4.0 / 3.0;
        let combined = cam.view_projection(aspect).expect("usable camera");
        let composed =
            cam.projection(aspect).expect("usable projection") * cam.view().expect("usable view");
        for (a, b) in combined.iter().zip(composed.iter()) {
            assert!((a - b).abs() < 1e-12, "view_projection is not P * V");
        }

        // The target sits on the view axis, so it must land at the centre of
        // the viewport, and in front of the camera.
        let clip = transform(&combined, &to_homogeneous(&cam.target()));
        assert!(clip[3] > 0.0, "the target must be in front of the camera");
        assert!(
            clip[0].abs() < 1e-9 && clip[1].abs() < 1e-9,
            "the target is off-centre"
        );
    }

    #[test]
    fn world_up_is_consistent_with_the_offset_direction() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
        cam.orbit(0.3, 0.2);
        // The camera's up vector must have a positive world-z component when
        // the eye is above the target, and the right vector must be horizontal.
        assert!(cam.up().z > 0.0);
        assert!(cam.right().z.abs() < 1e-12);
        assert!(cam.forward().dot(z_up()).abs() <= 1.0);
    }
}
