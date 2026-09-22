//! Frame-level tests for the renderer: coverage, culling, depth, and lifecycle.

use crate::domain::core::scalar::Point3r;
use crate::domain::mesh::indexed::IndexedMesh;

use super::super::camera::OrbitCamera;
use super::error::RenderError;
use super::renderer::Renderer;
use super::settings::{CullMode, RenderSettings};
use super::stats::RenderStats;
use super::tests_support::{assert_coverage_agrees, camera_facing_cube, cube, quad_facing_camera};
use super::MAX_PIXELS;

#[test]
fn dimensions_over_the_pixel_limit_are_refused() {
    assert!(matches!(
        Renderer::new(MAX_PIXELS as u32, 2),
        Err(RenderError::DimensionsTooLarge { .. })
    ));
    assert!(Renderer::new(64, 64).is_ok());
    // A collapsed window is legal and simply draws nothing.
    assert!(Renderer::new(0, 0).is_ok());
}

#[test]
fn render_rejects_a_short_colour_buffer() {
    let mut renderer = Renderer::new(16, 16).expect("small viewport");
    let mut color = vec![0_u32; 16 * 16 - 1];
    let mesh = cube();
    let cam = camera_facing_cube();
    let settings = RenderSettings::default();
    assert!(matches!(
        renderer.render(&mesh, &cam, &mut color, &settings),
        Err(RenderError::ColorBufferTooSmall { .. })
    ));
}

#[test]
fn a_zero_sized_viewport_renders_nothing_without_error() {
    let mut renderer = Renderer::new(0, 0).expect("collapsed viewport");
    let mut color: Vec<u32> = Vec::new();
    let stats = renderer
        .render(
            &cube(),
            &camera_facing_cube(),
            &mut color,
            &RenderSettings::default(),
        )
        .expect("a collapsed viewport is not an error");
    assert_eq!(stats, RenderStats::default());
}

/// The cube must actually be drawn: pixels differ from the background and
/// the depth buffer is populated exactly where they do.
#[test]
fn a_framed_cube_covers_pixels_and_fills_depth() {
    let mut renderer = Renderer::new(120, 90).expect("viewport");
    let mut color = vec![0_u32; 120 * 90];
    let settings = RenderSettings::default();
    let stats = renderer
        .render(&cube(), &camera_facing_cube(), &mut color, &settings)
        .expect("render");

    assert_eq!(stats.faces_considered, 12);
    assert_eq!(stats.degenerate_faces, 0);
    assert!(
        stats.fragments_passed > 1000,
        "the cube should cover a large area, got {}",
        stats.fragments_passed
    );
    assert!(stats.triangles_rasterized >= 3);
    assert_coverage_agrees(&renderer, &color, settings.background, stats);
}

/// Culling removes triangles, not pixels: on a closed convex surface every
/// back-facing triangle is behind a front-facing one, so the image must be
/// identical. Only the *write count* may differ, because an unculled
/// back-facing fragment is written and then overwritten by the nearer
/// front-facing one at the same pixel.
#[test]
fn culling_backfaces_removes_triangles_but_not_pixels() {
    let mut renderer = Renderer::new(120, 90).expect("viewport");
    let cam = camera_facing_cube();
    let mut culled_color = vec![0_u32; 120 * 90];
    let mut unculled_color = vec![0_u32; 120 * 90];

    let culled = renderer
        .render(&cube(), &cam, &mut culled_color, &RenderSettings::default())
        .expect("render");
    let unculled = renderer
        .render(
            &cube(),
            &cam,
            &mut unculled_color,
            &RenderSettings {
                cull: CullMode::None,
                ..RenderSettings::default()
            },
        )
        .expect("render");

    assert_eq!(culled.backface_culled + culled.triangles_rasterized, 12);
    assert_eq!(unculled.backface_culled, 0);
    assert_eq!(unculled.triangles_rasterized, 12);
    assert!(
        unculled.triangles_rasterized > culled.triangles_rasterized,
        "culling should submit fewer triangles"
    );
    assert_eq!(
        culled_color, unculled_color,
        "culling must not change a single pixel"
    );
    // Every pixel the culled pass fills is also filled by the unculled pass,
    // so dropping back faces can never lower the write count. It is not an
    // equality: a back-facing fragment is only *written* when it is drawn
    // before the nearer front-facing one, so the excess depends on face
    // order.
    assert!(
        unculled.fragments_passed >= culled.fragments_passed,
        "unculled writes {} should be at least the culled {}",
        unculled.fragments_passed,
        culled.fragments_passed
    );
}

/// A face with three identical positions has no normal and must be skipped
/// rather than producing a NaN transform.
#[test]
fn a_degenerate_face_is_counted_and_skipped() {
    let mut mesh = IndexedMesh::new();
    let v = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
    mesh.add_face(v, v, v);
    let mut renderer = Renderer::new(32, 32).expect("viewport");
    let mut color = vec![0_u32; 32 * 32];
    let stats = renderer
        .render(
            &mesh,
            &camera_facing_cube(),
            &mut color,
            &RenderSettings::default(),
        )
        .expect("render");
    assert_eq!(stats.degenerate_faces, 1);
    assert_eq!(stats.triangles_rasterized, 0);
    assert_eq!(stats.fragments_passed, 0);
}

/// A camera inside a closed mesh draws nothing while back-face culling is
/// on: every front-facing face is behind the eye, and every face in front of
/// the eye is back-facing. Pinned because it is the blank frame a host has
/// to be able to explain rather than report as a renderer failure, and
/// because the same scene drawn without culling is not blank.
#[test]
fn a_camera_inside_a_closed_mesh_draws_nothing_while_culling() {
    let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 0.2);
    cam.orbit(0.6, 0.4);
    cam.set_clip_planes(0.1, 100.0);
    let mut renderer = Renderer::new(120, 90).expect("viewport");
    let mut color = vec![0_u32; 120 * 90];

    let culled = renderer
        .render(&cube(), &cam, &mut color, &RenderSettings::default())
        .expect("render");
    assert_eq!(
        culled.fragments_passed, 0,
        "inside the cube every face in front of the eye is back-facing"
    );

    // Without culling the far interior walls are visible. The walls between
    // the eye and the target are behind the eye, so this is also a
    // near-plane clipping case.
    let unculled = renderer
        .render(
            &cube(),
            &cam,
            &mut color,
            &RenderSettings {
                cull: CullMode::None,
                ..RenderSettings::default()
            },
        )
        .expect("render");
    assert!(
        unculled.near_plane_clipped > 0,
        "the walls behind the eye must be clipped"
    );
    assert!(
        unculled.fragments_passed > 0,
        "the far interior walls are in front of the eye"
    );
    assert_coverage_agrees(
        &renderer,
        &color,
        RenderSettings::default().background,
        unculled,
    );
}

/// Geometry entirely behind the eye has `w < near` at every vertex, so it
/// must be clipped away rather than projected through the origin.
#[test]
fn geometry_behind_the_camera_is_clipped_away() {
    let mut mesh = IndexedMesh::new();
    // The default camera sits at x = +6 looking towards -x, so a quad at
    // x = +10 is behind the eye.
    quad_facing_camera(&mut mesh, 10.0);
    let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 6.0);
    cam.set_clip_planes(0.1, 100.0);

    let mut renderer = Renderer::new(64, 64).expect("viewport");
    let mut color = vec![0_u32; 64 * 64];
    let stats = renderer
        .render(
            &mesh,
            &cam,
            &mut color,
            &RenderSettings {
                cull: CullMode::None,
                ..RenderSettings::default()
            },
        )
        .expect("render");

    assert_eq!(stats.near_plane_clipped, 2, "both faces are behind the eye");
    assert_eq!(stats.triangles_rasterized, 0);
    assert_eq!(stats.fragments_passed, 0);
    assert!(color
        .iter()
        .all(|&p| p == RenderSettings::default().background.packed()));
}

/// Geometry outside the frustum but in front of the camera must also draw
/// nothing, without being mistaken for a clipping failure.
#[test]
fn geometry_outside_the_frustum_draws_nothing() {
    let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 8.0);
    cam.set_clip_planes(0.1, 100.0);
    // Pan the target far along the camera's up vector so the cube leaves
    // the viewport entirely.
    cam.pan_world(0.0, 10_000.0);
    let mut renderer = Renderer::new(60, 60).expect("viewport");
    let mut color = vec![0_u32; 60 * 60];
    let stats = renderer
        .render(&cube(), &cam, &mut color, &RenderSettings::default())
        .expect("render");
    assert_eq!(stats.fragments_passed, 0);
    assert_eq!(
        stats.near_plane_clipped, 0,
        "the cube is in front of the camera, just off screen"
    );
}

/// A camera whose eye rounds to its target has no view direction, and that
/// must be reported rather than rendered as garbage.
#[test]
fn a_camera_with_no_view_direction_is_reported() {
    // At this magnitude a distance of 1e-9 is below the representable
    // offset from the target, so `eye` equals `target` exactly.
    let cam = OrbitCamera::new(Point3r::new(1.0e300, 0.0, 0.0), 1.0e-9);
    assert!(cam.view().is_none(), "the eye rounds onto the target");

    let mut renderer = Renderer::new(8, 8).expect("viewport");
    let mut color = vec![0_u32; 64];
    assert!(matches!(
        renderer.render(&cube(), &cam, &mut color, &RenderSettings::default()),
        Err(RenderError::DegenerateCamera)
    ));
}

/// Two coplanar quads at different depths must resolve by depth, not by
/// submission order: the nearer one wins.
#[test]
fn depth_resolves_occlusion_independently_of_submission_order() {
    let mut mesh = IndexedMesh::new();
    quad_facing_camera(&mut mesh, 0.0); // 6 units from the eye
    quad_facing_camera(&mut mesh, -1.0); // 7 units from the eye

    let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 6.0);
    cam.set_clip_planes(0.1, 100.0);
    let mut renderer = Renderer::new(40, 40).expect("viewport");
    let mut color = vec![0_u32; 40 * 40];
    renderer
        .render(
            &mesh,
            &cam,
            &mut color,
            &RenderSettings {
                cull: CullMode::None,
                ..RenderSettings::default()
            },
        )
        .expect("render");

    // The centre pixel is covered by both quads; the nearer one must win,
    // so its depth key is the larger of 1/6 and 1/7.
    let centre = 20 * 40 + 20;
    let depth = renderer.depth()[centre];
    assert!(
        depth > 1.0 / 6.5 && depth < 1.0 / 5.5,
        "the centre should hold the near quad at 1/6, got {depth}"
    );
}

/// A mesh drawn twice must produce identical output: nothing in the
/// rasteriser may depend on previous frame state beyond the depth buffer,
/// which is cleared.
#[test]
fn rendering_is_deterministic_across_frames() {
    let mut renderer = Renderer::new(64, 48).expect("viewport");
    let cam = camera_facing_cube();
    let settings = RenderSettings::default();
    let mut first = vec![0_u32; 64 * 48];
    let mut second = vec![0_u32; 64 * 48];
    let a = renderer
        .render(&cube(), &cam, &mut first, &settings)
        .expect("render");
    let b = renderer
        .render(&cube(), &cam, &mut second, &settings)
        .expect("render");
    assert_eq!(a, b);
    assert_eq!(first, second);
}

/// Resizing must keep the frame consistent: the depth buffer is resized
/// with the colour buffer, so a stale pixel cannot survive a resize.
#[test]
fn resizing_reallocates_depth_and_clears_it() {
    let mut renderer = Renderer::new(32, 32).expect("viewport");
    let mut color = vec![0_u32; 32 * 32];
    renderer
        .render(
            &cube(),
            &camera_facing_cube(),
            &mut color,
            &RenderSettings::default(),
        )
        .expect("render");
    assert!(renderer.depth().iter().any(|&d| d > 0.0));

    renderer.resize(64, 48).expect("resize");
    assert_eq!(renderer.depth().len(), 64 * 48);
    assert!(
        renderer.depth().iter().all(|&d| d == 0.0),
        "a resize must not leave depth from the previous size"
    );

    let mut color = vec![0_u32; 64 * 48];
    let stats = renderer
        .render(
            &cube(),
            &camera_facing_cube(),
            &mut color,
            &RenderSettings::default(),
        )
        .expect("render");
    assert!(stats.fragments_passed > 0);
}
