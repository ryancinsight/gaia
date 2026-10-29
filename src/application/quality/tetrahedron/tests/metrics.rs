//! Native tetrahedral shape metrics.

#[cfg(test)]
mod tests {
    use crate::application::quality::tetrahedron::{tetrahedron_quality, TetrahedronQuality};
    use crate::domain::core::scalar::Scalar;
    use leto::geometry::Point3;

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
        assert!((quality.normalized_volume.to_f64() - 0.769_800_358_919_501).abs() < 1e-6);
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
}
