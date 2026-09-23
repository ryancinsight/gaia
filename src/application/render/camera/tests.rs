//! Tests for the orbit camera's geometry, clamping, and framing.

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

/// The sphere is the framing primitive: the silhouette is the worst case, so a
/// probe set on the surface must land inside the viewport, and the distance must
/// clear the sphere by the margin from the narrower half-angle.
#[test]
fn fit_sphere_frames_the_sphere() {
    let cases = [
        (Point3r::new(0.0, 0.0, 0.0), 2.0),
        (Point3r::new(-5.0, 3.0, 1.0), 0.25),
        (Point3r::new(1.0e3, -2.0e3, 3.0), 10.0),
    ];
    for aspect in [0.5_f64, 1.0, 16.0 / 9.0] {
        for (center, radius) in cases {
            let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 1.0);
            cam.orbit(0.7, 0.4);
            cam.fit_sphere(center, radius, aspect);
            assert_eq!(cam.target(), center, "the sphere centre becomes the target");

            let half_fov_y = cam.fov_y() * 0.5;
            let half_fov_x = (half_fov_y.tan() * aspect).atan();
            let half_fov = half_fov_x.min(half_fov_y);
            assert!(
                cam.distance() * half_fov.sin() >= radius,
                "distance {} does not clear radius {radius}",
                cam.distance()
            );

            let vp = cam.view_projection(aspect).expect("usable camera");
            for (i, probe) in surface_probes(center, radius).into_iter().enumerate() {
                let clip = transform(&vp, &to_homogeneous(&probe));
                assert!(clip[3] > 0.0, "probe {i} is behind the camera");
                let (ndc_x, ndc_y) = (clip[0] / clip[3], clip[1] / clip[3]);
                assert!(
                    ndc_x.abs() <= 1.0 && ndc_y.abs() <= 1.0,
                    "probe {i} at ({ndc_x}, {ndc_y}) is outside the viewport"
                );
            }
        }
    }
}

/// Six surface points: the sphere's extent along each world axis. The silhouette
/// is the true worst case, so these are inside it.
fn surface_probes(center: Point3r, radius: Real) -> [Point3r; 6] {
    [
        center + Vector3r::new(radius, 0.0, 0.0),
        center + Vector3r::new(-radius, 0.0, 0.0),
        center + Vector3r::new(0.0, radius, 0.0),
        center + Vector3r::new(0.0, -radius, 0.0),
        center + Vector3r::new(0.0, 0.0, radius),
        center + Vector3r::new(0.0, 0.0, -radius),
    ]
}

#[test]
fn fit_sphere_leaves_the_camera_alone_for_a_degenerate_sphere() {
    let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 5.0);
    let before = cam;
    let c = Point3r::new(1.0, 2.0, 3.0);
    cam.fit_sphere(c, 0.0, 1.0);
    cam.fit_sphere(c, -1.0, 1.0);
    cam.fit_sphere(c, f64::NAN, 1.0);
    cam.fit_sphere(c, f64::INFINITY, 1.0);
    cam.fit_sphere(c, 1.0, 0.0);
    cam.fit_sphere(c, 1.0, -2.0);
    cam.fit_sphere(c, 1.0, f64::NAN);
    assert_eq!(cam, before);
}

/// `fit` is the box case of the sphere primitive: it frames the box's
/// circumscribed sphere, so the two must agree for every box.
#[test]
fn fit_is_the_circumscribed_sphere_of_the_box() {
    let cases = [
        Aabb::new(Point3r::new(0.0, 0.0, 0.0), Point3r::new(4.0, 4.0, 4.0)),
        Aabb::new(
            Point3r::new(-10.0, -1.0, -1.0),
            Point3r::new(10.0, 1.0, 1.0),
        ),
        Aabb::new(Point3r::new(0.5, -0.25, 2.0), Point3r::new(0.75, 0.25, 2.5)),
    ];
    for aspect in [0.5_f64, 1.0, 16.0 / 9.0] {
        for bounds in &cases {
            let mut by_box = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 1.0);
            by_box.orbit(0.7, 0.4);
            by_box.fit(bounds, aspect);

            let half = (bounds.max - bounds.min) * 0.5;
            let mut by_sphere = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 1.0);
            by_sphere.orbit(0.7, 0.4);
            by_sphere.fit_sphere(bounds.center(), half.norm(), aspect);

            assert_eq!(by_box, by_sphere, "fit must be the sphere case of the box");
        }
    }
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
