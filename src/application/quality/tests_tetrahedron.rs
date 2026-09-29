//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::mesh::TetrahedralMeshBuilder;

fn analytical_quality<T: Scalar>() -> TetrahedronQuality<T> {
    let scalar = |value| <T as Scalar>::from_f64(value);
    tetrahedron_quality([
        Point3::new(scalar(0.0), scalar(0.0), scalar(0.0)),
        Point3::new(scalar(1.0), scalar(0.0), scalar(0.0)),
        Point3::new(scalar(0.0), scalar(1.0), scalar(0.0)),
        Point3::new(scalar(0.0), scalar(0.0), scalar(1.0)),
    ])
    .expect("analytical tetrahedron is non-degenerate")
}

fn assert_analytical_quality<T: Scalar>() {
    let quality = analytical_quality::<T>();
    assert!((quality.volume.to_f64() - 1.0 / 6.0).abs() < 1e-6);
    assert!((quality.radius_edge_ratio.to_f64() - 3.0_f64.sqrt() / 2.0).abs() < 1e-6);
    assert!((quality.normalized_volume.to_f64() - 0.769800358919501).abs() < 1e-6);
    assert!(quality.min_dihedral_angle.to_f64().to_degrees() > 54.0);
}

#[test]
fn analytical_tetrahedron_has_expected_native_metrics() {
    assert_analytical_quality::<f32>();
    assert_analytical_quality::<f64>();
}

#[test]
fn sliver_has_worse_dihedral_and_normalized_volume() {
    let regular = analytical_quality::<f64>();
    let sliver = tetrahedron_quality([
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.45, 0.45, 1e-4),
    ])
    .expect("sliver remains non-degenerate");
    assert!(sliver.min_dihedral_angle < regular.min_dihedral_angle);
    assert!(sliver.normalized_volume < regular.normalized_volume);
    assert!(sliver.radius_edge_ratio > regular.radius_edge_ratio);
}

#[test]
fn shape_metrics_are_translation_invariant_and_volume_scales_cubically() {
    let points: [Point3<f64>; 4] = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ];
    let translated_and_scaled = points.map(|point| {
        Point3::new(
            11.0 + 7.0 * point.x,
            -3.0 + 7.0 * point.y,
            5.0 + 7.0 * point.z,
        )
    });
    let baseline = tetrahedron_quality(points).expect("baseline is valid");
    let transformed =
        tetrahedron_quality(translated_and_scaled).expect("transformed cell is valid");

    assert!((transformed.volume - baseline.volume * 343.0).abs() < 1e-12);
    assert!((transformed.radius_edge_ratio - baseline.radius_edge_ratio).abs() < 1e-12);
    assert!((transformed.min_dihedral_angle - baseline.min_dihedral_angle).abs() < 1e-12);
    assert!((transformed.normalized_volume - baseline.normalized_volume).abs() < 1e-12);
}

#[test]
fn criteria_distinguish_slivers_shape_failures_and_oversized_cells() {
    let criteria = TetrahedralQualityCriteria::<f64>::try_new(2.0, 0.5, 0.5, Some(1.0))
        .expect("criteria are valid");
    let sliver = TetrahedronQuality {
        volume: 0.1,
        radius_edge_ratio: 1.5,
        min_dihedral_angle: 0.25,
        normalized_volume: 0.25,
    };
    let poor_shape = TetrahedronQuality {
        volume: 0.1,
        radius_edge_ratio: 2.5,
        min_dihedral_angle: 0.75,
        normalized_volume: 0.75,
    };
    let oversized = TetrahedronQuality {
        volume: 2.0,
        radius_edge_ratio: 1.5,
        min_dihedral_angle: 0.75,
        normalized_volume: 0.75,
    };
    let invalid = TetrahedronQuality {
        volume: 0.1,
        radius_edge_ratio: 1.5,
        min_dihedral_angle: 0.75,
        normalized_volume: 1.1,
    };

    assert_eq!(criteria.classify(sliver), TetrahedronQualityClass::Sliver);
    assert_eq!(
        criteria.classify(poor_shape),
        TetrahedronQualityClass::PoorShape
    );
    assert_eq!(
        criteria.classify(oversized),
        TetrahedronQualityClass::Oversized
    );
    assert_eq!(criteria.classify(invalid), TetrahedronQualityClass::Invalid);
}

#[test]
fn criteria_assessment_is_native_and_counts_invalid_cells() {
    fn exercise<T: Scalar>() {
        let scalar = |value| <T as Scalar>::from_f64(value);
        let mut builder = TetrahedralMeshBuilder::<T>::new();
        let valid = [
            builder.vertex_array([scalar(0.0), scalar(0.0), scalar(0.0)]),
            builder.vertex_array([scalar(1.0), scalar(0.0), scalar(0.0)]),
            builder.vertex_array([scalar(0.0), scalar(1.0), scalar(0.0)]),
            builder.vertex_array([scalar(0.0), scalar(0.0), scalar(1.0)]),
        ];
        builder
            .tetrahedron(valid)
            .expect("valid tetrahedron is inserted");
        let mut mesh = builder.build();
        mesh.cells
            .push(crate::domain::topology::Cell::tetrahedron(0, 0, 0, 0));

        let criteria = TetrahedralQualityCriteria::try_new(
            scalar(2.0),
            scalar(0.5),
            scalar(0.5),
            Some(scalar(1.0)),
        )
        .expect("criteria are valid");
        let acceptance = criteria.assess(&mesh).expect("tetrahedral cells exist");
        assert_eq!(acceptance.accepted_cell_count, 1);
        assert_eq!(acceptance.invalid_cell_count, 1);
        assert_eq!(acceptance.rejected_cell_count(), 1);
        assert!(!acceptance.passed());
    }

    exercise::<f32>();
    exercise::<f64>();
}

#[test]
fn criteria_reject_non_finite_and_out_of_domain_bounds() {
    assert_eq!(
        TetrahedralQualityCriteria::<f64>::try_new(0.0, 0.5, 0.5, None),
        Err(TetrahedralQualityCriteriaError::InvalidMaxRadiusEdgeRatio)
    );
    assert_eq!(
        TetrahedralQualityCriteria::<f64>::try_new(2.0, -0.1, 0.5, None),
        Err(TetrahedralQualityCriteriaError::InvalidMinDihedralAngle)
    );
    assert_eq!(
        TetrahedralQualityCriteria::<f64>::try_new(2.0, 0.5, 1.1, None),
        Err(TetrahedralQualityCriteriaError::InvalidMinNormalizedVolume)
    );
    assert_eq!(
        TetrahedralQualityCriteria::<f64>::try_new(2.0, 0.5, 0.5, Some(0.0)),
        Err(TetrahedralQualityCriteriaError::InvalidMaxVolume)
    );
}

#[test]
fn report_counts_invalid_tetrahedra_without_defaulting_metrics() {
    let mut builder = TetrahedralMeshBuilder::<f64>::new();
    let a = builder.vertex_xyz(0.0, 0.0, 0.0);
    let b = builder.vertex_xyz(1.0, 0.0, 0.0);
    let c = builder.vertex_xyz(0.0, 1.0, 0.0);
    let d = builder.vertex_xyz(0.0, 0.0, 1.0);
    builder
        .tetrahedron([a, b, c, d])
        .expect("valid tetrahedron");
    let mut mesh = builder.build();
    mesh.cells
        .push(crate::domain::topology::Cell::tetrahedron(0, 0, 0, 0));
    let report = tetrahedral_quality_report(&mesh).expect("tetrahedral cells exist");
    assert_eq!(report.valid_cell_count, 1);
    assert_eq!(report.invalid_cell_count, 1);
    assert_eq!(report.volume.expect("valid metric").count, 1);
}
