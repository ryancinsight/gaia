//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::index::VertexId;
use crate::domain::mesh::TetrahedralMeshBuilder;

fn unit_tetrahedron<T: Scalar>() -> IndexedMesh<T> {
    let scalar = |value| <T as Scalar>::from_f64(value);
    let mut builder = TetrahedralMeshBuilder::<T>::new();
    let vertices = [
        builder.vertex_array([scalar(0.0), scalar(0.0), scalar(0.0)]),
        builder.vertex_array([scalar(1.0), scalar(0.0), scalar(0.0)]),
        builder.vertex_array([scalar(0.0), scalar(1.0), scalar(0.0)]),
        builder.vertex_array([scalar(0.0), scalar(0.0), scalar(1.0)]),
    ];
    builder
        .tetrahedron(vertices)
        .expect("unit tetrahedron is valid");
    builder.build()
}

fn cell_criteria<T: Scalar>() -> TetrahedralQualityCriteria<T> {
    let scalar = |value| <T as Scalar>::from_f64(value);
    TetrahedralQualityCriteria::try_new(scalar(2.0), scalar(0.7), scalar(0.7), Some(scalar(1.0)))
        .expect("cell criteria are valid")
}

fn facet_criteria<T: Scalar>(max_edge_length: Option<f64>) -> BoundaryFacetQualityCriteria<T> {
    let scalar = |value| <T as Scalar>::from_f64(value);
    BoundaryFacetQualityCriteria::try_new(
        aequitas::systems::si::quantities::Angle::from_base(scalar(0.7)),
        aequitas::systems::si::quantities::Dimensionless::from_base(scalar(0.6)),
        max_edge_length
            .map(|value| aequitas::systems::si::quantities::Length::from_base(scalar(value))),
    )
    .expect("facet criteria are valid")
}

#[test]
fn boundary_acceptance_is_native_for_f32_and_f64() {
    fn exercise<T: Scalar>() {
        let mesh = unit_tetrahedron::<T>();
        let acceptance = cell_criteria::<T>()
            .assess_boundary(&mesh, &facet_criteria::<T>(Some(1.5)))
            .expect("tetrahedral cells exist");
        assert_eq!(acceptance.boundary_cell_count, 1);
        assert_eq!(acceptance.accepted_boundary_cell_count, 1);
        assert_eq!(acceptance.rejected_boundary_cell_count, 0);
        assert_eq!(acceptance.boundary_facet_acceptance.accepted_facet_count, 4);
        assert!(acceptance.passed());
    }

    exercise::<f32>();
    exercise::<f64>();
}

#[test]
fn aequitas_length_conversion_is_native_and_canonical() {
    fn exercise<T: Scalar + eunomia::UnitScalar>() {
        let length = aequitas::systems::si::quantities::Length::<T>::from_unit::<
            aequitas::systems::si::units::Millimeter,
        >(<T as Scalar>::from_f64(2.0));
        assert!((length.into_base().to_f64() - 0.002).abs() < 1e-8);
    }

    exercise::<f32>();
    exercise::<f64>();
}

#[test]
fn oversized_boundary_facets_reject_their_boundary_cell() {
    let mesh = unit_tetrahedron::<f64>();
    let acceptance = cell_criteria::<f64>()
        .assess_boundary(&mesh, &facet_criteria::<f64>(Some(1.2)))
        .expect("tetrahedral cells exist");
    assert_eq!(acceptance.boundary_cell_count, 1);
    assert_eq!(acceptance.accepted_boundary_cell_count, 0);
    assert_eq!(acceptance.rejected_boundary_cell_count, 1);
    assert_eq!(
        acceptance.boundary_facet_acceptance.oversized_facet_count,
        4
    );
    assert!(!acceptance.passed());
}

#[test]
fn invalid_boundary_face_is_rejected_without_panicking() {
    let mut mesh = unit_tetrahedron::<f64>();
    mesh.faces.get_mut(FaceId::from_usize(0)).vertices[0] =
        VertexId::from_usize(mesh.vertex_count() + 1);
    let acceptance = cell_criteria::<f64>()
        .assess_boundary(&mesh, &facet_criteria::<f64>(Some(1.5)))
        .expect("tetrahedral cells exist");
    assert_eq!(acceptance.invalid_boundary_cell_count, 1);
    assert_eq!(acceptance.boundary_facet_acceptance.invalid_facet_count, 1);
    assert!(!acceptance.passed());
}

#[test]
fn malformed_tetrahedral_topology_is_not_treated_as_interior() {
    let mut mesh = unit_tetrahedron::<f64>();
    let mut malformed = crate::domain::topology::Cell::tetrahedron(0, 0, 0, 0);
    malformed.vertex_ids = vec![0, 1, 2, 3];
    mesh.cells.push(malformed);
    let acceptance = cell_criteria::<f64>()
        .assess_boundary(&mesh, &facet_criteria::<f64>(Some(1.5)))
        .expect("tetrahedral cells exist");
    assert_eq!(acceptance.boundary_cell_count, 2);
    assert_eq!(acceptance.invalid_boundary_cell_count, 1);
    assert_eq!(acceptance.rejected_boundary_cell_count, 1);
    assert!(!acceptance.passed());
}

#[test]
fn non_manifold_face_is_not_treated_as_interior() {
    let mut mesh = unit_tetrahedron::<f64>();
    for _ in 0..2 {
        let mut duplicate = crate::domain::topology::Cell::tetrahedron(0, 1, 2, 3);
        duplicate.vertex_ids = vec![0, 1, 2, 3];
        mesh.cells.push(duplicate);
    }

    let acceptance = cell_criteria::<f64>()
        .assess_boundary(&mesh, &facet_criteria::<f64>(Some(1.5)))
        .expect("tetrahedral cells exist");
    assert_eq!(acceptance.boundary_cell_count, 3);
    assert_eq!(acceptance.accepted_boundary_cell_count, 0);
    assert_eq!(acceptance.rejected_boundary_cell_count, 3);
    assert_eq!(acceptance.invalid_boundary_cell_count, 3);
    assert!(!acceptance.passed());
}

#[test]
fn boundary_facet_criteria_reject_invalid_bounds() {
    assert_eq!(
        BoundaryFacetQualityCriteria::<f64>::try_new(
            aequitas::systems::si::quantities::Angle::from_base(std::f64::consts::PI / 2.0,),
            aequitas::systems::si::quantities::Dimensionless::from_base(0.5),
            Some(aequitas::systems::si::quantities::Length::from_base(1.0)),
        ),
        Err(BoundaryFacetQualityCriteriaError::InvalidMinAngle)
    );
    assert_eq!(
        BoundaryFacetQualityCriteria::<f64>::try_new(
            aequitas::systems::si::quantities::Angle::from_base(0.5),
            aequitas::systems::si::quantities::Dimensionless::from_base(1.1),
            Some(aequitas::systems::si::quantities::Length::from_base(1.0)),
        ),
        Err(BoundaryFacetQualityCriteriaError::InvalidMinEdgeLengthRatio)
    );
    assert_eq!(
        BoundaryFacetQualityCriteria::<f64>::try_new(
            aequitas::systems::si::quantities::Angle::from_base(0.5),
            aequitas::systems::si::quantities::Dimensionless::from_base(0.5),
            Some(aequitas::systems::si::quantities::Length::from_base(0.0)),
        ),
        Err(BoundaryFacetQualityCriteriaError::InvalidMaxEdgeLength)
    );
}
