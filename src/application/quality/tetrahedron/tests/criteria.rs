//! Criteria classification, assessment, validation, and typed units.

#[cfg(test)]
mod tests {
    use crate::application::quality::tetrahedron::{
        TetrahedralQualityCriteria, TetrahedralQualityCriteriaError, TetrahedronQuality,
        TetrahedronQualityClass,
    };
    use crate::domain::core::scalar::Scalar;
    use crate::domain::mesh::TetrahedralMeshBuilder;
    use aequitas::systems::si::quantities::{Angle, Dimensionless};

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
    fn typed_criteria_constructor_matches_scalar_constructor() {
        let typed = TetrahedralQualityCriteria::<f64>::try_new_typed(
            Dimensionless::from_base(2.0),
            Angle::from_base(0.5),
            Dimensionless::from_base(0.5),
            Some(1.0),
        )
        .expect("typed criteria are valid");
        let scalar = TetrahedralQualityCriteria::<f64>::try_new(2.0, 0.5, 0.5, Some(1.0))
            .expect("scalar criteria are valid");
        assert_eq!(typed, scalar);
        assert_eq!(typed.max_radius_edge_ratio_quantity().into_base(), 2.0);
        assert_eq!(typed.min_dihedral_angle_quantity().into_base(), 0.5);
        assert_eq!(typed.min_normalized_volume_quantity().into_base(), 0.5);
    }
}
