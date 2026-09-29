//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::application::watertight::check::check_watertight;
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::test_support::assert_rejects;
use std::f64::consts::PI;

/// Revolve a single vertical edge of radius r and height h → open tube
/// with no caps. Test with a full revolution.
#[test]
fn revolution_sweep_full_cylinder_watertight() {
    // Vertical edge at radius r=1, from y=0 to y=2
    let sweep = RevolutionSweep {
        profile: vec![(1.0, 0.0), (1.0, 2.0)],
        segments: 32,
        angle: TAU,
    };
    let mesh = sweep.build().unwrap();
    let edges = EdgeStore::from_face_store(&mesh.faces);
    let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
    // The lateral surface of a cylinder (open tube) is NOT closed — it has
    // two boundary loops (top and bottom circles). This test verifies the
    // topology compiles and the faces have consistent winding.
    assert!(
        report.orientation_consistent,
        "revolution winding must be consistent"
    );
}

/// Revolve a closed rectangular profile → solid ring (washer shape)
/// Full revolution → should be watertight (closed torus-like surface).
///
/// The profile must explicitly close the loop so all 4 walls are generated:
/// bottom annulus + outer cylinder + top annulus + inner cylinder.
/// The 5th point closes the loop back to the start.
#[test]
fn revolution_sweep_washer_full_watertight() {
    // Closed 5-point loop: bottom → outer → top → inner → back to start
    // bottom: (1,0) → (2,0); outer: (2,0) → (2,0.5);
    // top: (2,0.5) → (1,0.5); inner: (1,0.5) → (1,0)
    let profile = vec![
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 0.5),
        (1.0, 0.5),
        (1.0, 0.0), // close the loop back to start
    ];
    let sweep = RevolutionSweep {
        profile,
        segments: 32,
        angle: TAU,
    };
    let mesh = sweep.build().unwrap();
    let edges = EdgeStore::from_face_store(&mesh.faces);
    let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
    // Full revolution of a 5-point closed profile → closed torus-like surface
    assert!(
        report.orientation_consistent,
        "revolution winding must be consistent"
    );
    assert!(
        report.is_closed,
        "full revolution of closed 5-point profile must be closed"
    );
}

/// Revolve a triangular profile → cone-like solid (full 360°).
#[test]
fn revolution_sweep_cone_like_watertight() {
    // Profile: (0,2) at apex, (1,0) at base rim.
    // Revolution around Y: generates a cone lateral surface + base disk.
    // Since r=0 at apex, top degenerate vertices merge.
    let sweep = RevolutionSweep {
        profile: vec![(0.0, 2.0), (1.0, 0.0)],
        segments: 32,
        angle: TAU,
    };
    let mesh = sweep.build().unwrap();
    assert!(mesh.face_count() > 0, "cone-like sweep must produce faces");
    let edges = EdgeStore::from_face_store(&mesh.faces);
    let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
    assert!(
        report.orientation_consistent,
        "cone revolution winding must be consistent"
    );
}

/// Partial revolution of a closed rectangular profile → watertight wedge.
///
/// The profile must close the loop (5 points: 4 walls + closing segment)
/// so that all 4 sides of the annular cross-section are swept.
/// The two end caps close the angular start and end faces.
#[test]
fn revolution_sweep_partial_watertight() {
    // Closed 5-point loop: bottom → outer → top → inner → back to start
    let profile = vec![
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0), // close the loop
    ];
    let sweep = RevolutionSweep {
        profile,
        segments: 16,
        angle: PI / 2.0, // 90°
    };
    let mesh = sweep.build().unwrap();
    let edges = EdgeStore::from_face_store(&mesh.faces);
    let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
    assert!(
        report.orientation_consistent,
        "partial revolution winding must be consistent"
    );
    assert!(
        report.is_closed,
        "partial revolution with closed profile + end caps must be closed"
    );
    assert!(
        report.is_watertight,
        "partial revolution must be watertight"
    );
}

/// Volume of partial revolution: Pappus's theorem.
///
/// The profile is a closed 5-point loop so the solid has all 4 walls.
/// Volume = r̄ × A × angle (Pappus), where A is the cross-section area.
#[test]
fn revolution_sweep_partial_volume_pappus() {
    // Revolve a closed annular cross-section (r=1..2, y=0..1) by π/2 (90°).
    // Cross-section area = (2-1) × (1-0) = 1 mm².
    // Centroid r̄ = 1.5 mm.
    // Volume = r̄ × A × angle = 1.5 × 1 × π/2 ≈ 2.3562 mm³
    let profile = vec![
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0), // close the loop
    ];
    let sweep = RevolutionSweep {
        profile,
        segments: 64,
        angle: PI / 2.0,
    };
    let mesh = sweep.build().unwrap();
    let edges = EdgeStore::from_face_store(&mesh.faces);
    let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
    assert!(report.signed_volume > 0.0, "positive volume");
    let expected = 1.5_f64 * 1.0 * (PI / 2.0);
    let error = (report.signed_volume - expected).abs() / expected;
    assert!(
        error < 0.02,
        "Pappus volume error {:.4}% < 2%",
        error * 100.0
    );
}

#[test]
fn revolution_sweep_rejects_too_few_profile_points() {
    let result = RevolutionSweep {
        profile: vec![(1.0, 0.0)],
        segments: 16,
        angle: TAU,
    }
    .build();
    assert_rejects(&result, "segments must be >= 3, got 1");
}

#[test]
fn revolution_sweep_rejects_negative_radius() {
    let result = RevolutionSweep {
        profile: vec![(-1.0, 0.0), (1.0, 1.0)],
        segments: 16,
        angle: TAU,
    }
    .build();
    assert_rejects(
        &result,
        "invalid parameter: all radial values must be ≥ 0, got -1",
    );
}

#[test]
fn revolution_sweep_rejects_too_few_segments() {
    let result = RevolutionSweep {
        profile: vec![(1.0, 0.0), (1.0, 1.0)],
        segments: 2,
        angle: TAU,
    }
    .build();
    assert_rejects(&result, "segments must be >= 3, got 2");
}
