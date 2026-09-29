//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::geometry::tpms::Gyroid;

#[test]
fn box_clip_produces_nonempty_mesh() {
    let params = TpmsBoxParams {
        bounds: [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
        period: 2.5,
        resolution: 16,
        iso_value: 0.0,
    };
    let mesh = build_tpms_box(&Gyroid, &params).expect("should succeed");
    assert!(
        mesh.face_count() > 0,
        "gyroid-in-box must produce at least one face"
    );
    assert!(
        mesh.vertex_count() > 0,
        "gyroid-in-box must produce at least one vertex"
    );
}

#[test]
fn box_clip_validates_degenerate_bounds() {
    let params = TpmsBoxParams {
        bounds: [5.0, 0.0, 0.0, 5.0, 10.0, 10.0], // x_max == x_min
        period: 2.5,
        resolution: 16,
        iso_value: 0.0,
    };
    assert!(build_tpms_box(&Gyroid, &params).is_err());
}

#[test]
fn box_clip_validates_low_resolution() {
    let params = TpmsBoxParams {
        bounds: [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        period: 2.5,
        resolution: 2,
        iso_value: 0.0,
    };
    assert!(build_tpms_box(&Gyroid, &params).is_err());
}

#[test]
fn box_clip_all_vertices_within_bounds() {
    let params = TpmsBoxParams {
        bounds: [-3.0, -2.0, -1.0, 4.0, 5.0, 6.0],
        period: 3.0,
        resolution: 16,
        iso_value: 0.0,
    };
    let mesh = build_tpms_box(&Gyroid, &params).expect("should succeed");
    let eps = params.period / params.resolution as f64; // one voxel tolerance
    for vid in 0..mesh.vertex_count() {
        let p = mesh.vertices.position(VertexId(vid as u32));
        assert!(
            p.x >= -3.0 - eps && p.x <= 4.0 + eps,
            "vertex x={} out of bounds",
            p.x,
        );
        assert!(
            p.y >= -2.0 - eps && p.y <= 5.0 + eps,
            "vertex y={} out of bounds",
            p.y,
        );
        assert!(
            p.z >= -1.0 - eps && p.z <= 6.0 + eps,
            "vertex z={} out of bounds",
            p.z,
        );
    }
}

// ── Graded builder tests ──────────────────────────────────────────────

#[test]
fn graded_uniform_matches_box_clip() {
    // A graded builder with constant period should produce the same mesh
    // topology as build_tpms_box (same face count ± small tolerance from
    // floating point differences).
    let bounds = [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0];
    let period = 2.5;
    let params = TpmsBoxParams {
        bounds,
        period,
        resolution: 16,
        iso_value: 0.0,
    };
    let uniform = build_tpms_box(&Gyroid, &params).unwrap();
    let graded = build_tpms_box_graded(&Gyroid, bounds, 16, 0.0, |_x, _y, _z| period).unwrap();
    assert_eq!(
        uniform.face_count(),
        graded.face_count(),
        "constant-period graded must produce same face count as uniform"
    );
}

#[test]
fn graded_mesh_nonempty() {
    // A graded mesh with period varying from 1.5 (walls) to 5.0 (center)
    // should produce a non-empty mesh.
    let bounds = [0.0, 0.0, 0.0, 10.0, 10.0, 5.0];
    let mesh = build_tpms_box_graded(&Gyroid, bounds, 20, 0.0, |_x, y, _z| {
        // Y ranges [0, 10]: center at 5.0
        let y_frac = y / 10.0;
        let wall_dist = (2.0 * (y_frac - 0.5)).abs();
        5.0 * (1.0 - wall_dist) + 1.5 * wall_dist
    })
    .unwrap();
    assert!(mesh.face_count() > 0, "graded gyroid must produce faces");
}

#[test]
fn graded_rejects_degenerate_bounds() {
    assert!(build_tpms_box_graded(
        &Gyroid,
        [0.0, 0.0, 0.0, 0.0, 10.0, 10.0],
        16,
        0.0,
        |_, _, _| 3.0,
    )
    .is_err());
}
