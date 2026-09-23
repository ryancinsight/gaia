//! Tests for the exact containment predicate and the midpoint fallback.

use super::geom::{dominant_normal_axes, inside_triangle};
use super::*;
use crate::domain::core::constants::MAX_STEINER_PER_FACE;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

fn p(x: Real, y: Real, z: Real) -> Point3r {
    Point3r::new(x, y, z)
}

#[test]
fn inside_triangle_exact_accepts_edge_point() {
    let a = p(0.0, 0.0, 0.0);
    let b = p(1.0, 0.0, 0.0);
    let c = p(0.0, 1.0, 0.0);
    let n = (b - a).cross(c - a);
    let (axis_u, axis_v) = dominant_normal_axes(n / n.norm());

    let edge_point = p(0.5, 0.5, 0.0); // on edge b-c
    assert!(inside_triangle(edge_point, a, b, c, n, axis_u, axis_v));
}

#[test]
fn inside_triangle_exact_rejects_outside_point() {
    let a = p(0.0, 0.0, 0.0);
    let b = p(1.0, 0.0, 0.0);
    let c = p(0.0, 1.0, 0.0);
    let n = (b - a).cross(c - a);
    let (axis_u, axis_v) = dominant_normal_axes(n / n.norm());

    let outside = p(1.1, 0.2, 0.0);
    assert!(!inside_triangle(outside, a, b, c, n, axis_u, axis_v));
}

#[test]
fn inside_triangle_exact_handles_reversed_winding() {
    let a = p(0.0, 0.0, 0.0);
    let b = p(0.0, 1.0, 0.0);
    let c = p(1.0, 0.0, 0.0); // reversed orientation vs CCW XY
    let n = (b - a).cross(c - a);
    let (axis_u, axis_v) = dominant_normal_axes(n / n.norm());

    let inside = p(0.2, 0.2, 0.0);
    assert!(inside_triangle(inside, a, b, c, n, axis_u, axis_v));
}

/// Regression: when total Steiner count exceeds MAX_STEINER_PER_FACE (256),
/// corefine_face must fall back to midpoint_subdivide and return non-empty.
///
/// This guards against O(s²) CDT blowup from complex multi-branch junctions.
#[test]
fn corefine_face_steiner_guard_triggers_midpoint_fallback() {
    use crate::application::csg::intersect::SnapSegment;

    let mut pool = VertexPool::new(1e-6_f64); // 1µm weld cell → all interior pts unique
    let n = Vector3r::new(0.0, 0.0, 1.0);

    // 10mm × 10mm right-triangle face in the XY plane.
    let v0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let v1 = pool.insert_or_weld(Point3r::new(0.01, 0.0, 0.0), n);
    let v2 = pool.insert_or_weld(Point3r::new(0.0, 0.01, 0.0), n);
    let face = FaceData::untagged(v0, v1, v2);

    // Generate enough unique horizontal segments to exceed MAX_STEINER_PER_FACE.
    // We need > 32768 segments. We use a 200 x 200 grid.
    let mut segments: Vec<SnapSegment> = Vec::new();
    let target_len = MAX_STEINER_PER_FACE;
    'outer: for i in 1..300_usize {
        for j in 1..300_usize {
            let x = i as Real * 1.5e-5;
            let y = j as Real * 1.5e-5;
            let x2 = x + 0.5e-5;
            if x + y < 0.009 && x2 + y < 0.009 {
                segments.push(SnapSegment {
                    start: Point3r::new(x, y, 0.0),
                    end: Point3r::new(x2, y, 0.0),
                });
                if segments.len() >= target_len {
                    break 'outer;
                }
            }
        }
    }

    assert!(
        segments.len() >= MAX_STEINER_PER_FACE / 2,
        "test requires at least {} segments; generated {}",
        MAX_STEINER_PER_FACE / 2,
        segments.len()
    );

    // Must not panic; Steiner guard triggers midpoint fallback before CDT receives
    // a pathological O(s²) input.
    let mut scratch = CorefinerScratch::new();
    let result = corefine_face(
        &face,
        &segments,
        &mut pool,
        &SeamVertexMap::new(),
        &mut scratch,
    );
    assert!(
        !result.is_empty(),
        "midpoint fallback must produce at least one triangle"
    );
}
