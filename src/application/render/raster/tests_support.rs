//! Fixtures shared by the rasteriser's frame-level tests.
//!
//! The scene builders and the coverage invariant are used by both the frame
//! tests and the near-plane clipping test, so they live here rather than being
//! repeated in each.

use crate::domain::core::scalar::{Point3r, Real};
use crate::domain::mesh::indexed::IndexedMesh;

use super::super::camera::OrbitCamera;
use super::color::Rgba8;
use super::renderer::Renderer;
use super::stats::RenderStats;

/// An axis-aligned cube of half-extent 1, wound outward.
pub(super) fn cube() -> IndexedMesh<Real> {
    let mut mesh = IndexedMesh::new();
    let ids: Vec<_> = [
        Point3r::new(-1.0, -1.0, -1.0),
        Point3r::new(1.0, -1.0, -1.0),
        Point3r::new(1.0, 1.0, -1.0),
        Point3r::new(-1.0, 1.0, -1.0),
        Point3r::new(-1.0, -1.0, 1.0),
        Point3r::new(1.0, -1.0, 1.0),
        Point3r::new(1.0, 1.0, 1.0),
        Point3r::new(-1.0, 1.0, 1.0),
    ]
    .into_iter()
    .map(|p| mesh.add_vertex_pos(p))
    .collect();
    for q in [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 4, 7, 3],
    ] {
        mesh.add_face(ids[q[0]], ids[q[1]], ids[q[2]]);
        mesh.add_face(ids[q[0]], ids[q[2]], ids[q[3]]);
    }
    mesh
}

/// A quad spanning y and z in `[-1, 1]` at a fixed `x`, so it faces the
/// camera's default `+x` eye.
pub(super) fn quad_facing_camera(mesh: &mut IndexedMesh<Real>, x: Real) {
    let a = mesh.add_vertex_pos(Point3r::new(x, -1.0, -1.0));
    let b = mesh.add_vertex_pos(Point3r::new(x, 1.0, -1.0));
    let c = mesh.add_vertex_pos(Point3r::new(x, 1.0, 1.0));
    let d = mesh.add_vertex_pos(Point3r::new(x, -1.0, 1.0));
    mesh.add_face(a, b, c);
    mesh.add_face(a, c, d);
}

/// The default camera looks along `-x` from `+x`, so this views the cube
/// from a corner.
pub(super) fn camera_facing_cube() -> OrbitCamera {
    let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 8.0);
    cam.orbit(0.6, 0.4);
    cam.set_clip_planes(0.1, 100.0);
    cam
}

/// Colour and depth coverage must agree, and the reported fragment count
/// must match the distinct pixels covered up to shared-edge double counts.
pub(super) fn assert_coverage_agrees(
    renderer: &Renderer,
    color: &[u32],
    background: Rgba8,
    stats: RenderStats,
) {
    let background = background.packed();
    let covered = color.iter().filter(|&&p| p != background).count();
    let with_depth = renderer.depth().iter().filter(|&&d| d > 0.0).count();
    assert_eq!(
        covered, with_depth,
        "colour coverage ({covered}) and depth coverage ({with_depth}) disagree"
    );
    assert!(
        stats.fragments_passed >= covered,
        "a covered pixel must have been written: {} writes for {covered} pixels",
        stats.fragments_passed
    );
    assert!(
        stats.fragments_passed <= covered + covered / 50 + 1,
        "only shared-edge pixels may be counted twice: {} writes for {covered} pixels",
        stats.fragments_passed
    );
}
