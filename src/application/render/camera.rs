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

    /// Frame a sphere of `radius` about `center` from the current angles.
    ///
    /// This is the framing primitive. A box is framed through its circumscribed
    /// sphere (see [`Self::fit`]), and a caller holding a bounding sphere — a
    /// mesh's, or one fitted to a point set — can frame it without inventing a
    /// box around it.
    ///
    /// The clip planes are reset from the resulting distance, because a viewer
    /// that frames a small part after a large one needs a near plane tight
    /// enough to keep depth precision. The distance is chosen from the
    /// *narrower* of the horizontal and vertical half-angles, so a wide sphere
    /// in a tall viewport is still fully framed.
    ///
    /// A non-positive or non-finite radius, or an unusable `aspect`
    /// (non-finite or `<= 0`), leaves the camera unchanged.
    pub fn fit_sphere(&mut self, center: Point3r, radius: Real, aspect: Real) {
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
        self.target = center;
        self.distance = distance;
        self.near = (distance * 1.0e-3).max(1.0e-9);
        self.far = (distance + radius * 4.0).max(self.near * 1.0e3);
    }

    /// Frame `bounds` so the whole box is visible from the current angles.
    ///
    /// The box is framed through its circumscribed sphere, so this is
    /// [`Self::fit_sphere`] with the half-diagonal as the radius.
    ///
    /// An empty or degenerate box leaves the camera unchanged.
    pub fn fit(&mut self, bounds: &Aabb<Real>, aspect: Real) {
        let half = (bounds.max - bounds.min) * 0.5;
        self.fit_sphere(bounds.center(), half.norm(), aspect);
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
mod tests;
