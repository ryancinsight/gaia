//! Boundary-facet and boundary-cell quality acceptance.
//!
//! Three-dimensional mesh generators need separate surface-facet and volume-
//! cell criteria. A volume cell can satisfy its radius-edge and dihedral
//! bounds while an exposed facet is oversized or poorly shaped. This module
//! keeps that distinction explicit and composes the facet policy with Gaia's
//! existing [`super::tetrahedron::TetrahedralQualityCriteria`].

use aequitas::systems::si::quantities::{Angle, Dimensionless, Length};
use core::mem::{align_of, size_of};
use eunomia::{NumericElement, RealField};
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;
use leto::geometry::Point3;

use crate::application::quality::tetrahedron::{
    cell_tetrahedron_quality, TetrahedralQualityCriteria, TetrahedronQualityClass,
};
use crate::application::quality::triangle::{edge_length_ratio_native, min_angle_native};
use crate::domain::core::index::FaceId;
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::ElementType;

const _: () = {
    assert!(size_of::<Angle<f32>>() == size_of::<f32>());
    assert!(size_of::<Dimensionless<f32>>() == size_of::<f32>());
    assert!(size_of::<Length<f32>>() == size_of::<f32>());
    assert!(align_of::<Angle<f32>>() == align_of::<f32>());
    assert!(align_of::<Dimensionless<f32>>() == align_of::<f32>());
    assert!(align_of::<Length<f32>>() == align_of::<f32>());
    assert!(size_of::<Angle<f64>>() == size_of::<f64>());
    assert!(size_of::<Dimensionless<f64>>() == size_of::<f64>());
    assert!(size_of::<Length<f64>>() == size_of::<f64>());
    assert!(align_of::<Angle<f64>>() == align_of::<f64>());
    assert!(align_of::<Dimensionless<f64>>() == align_of::<f64>());
    assert!(align_of::<Length<f64>>() == align_of::<f64>());
};

/// Native-precision quality values for one exposed triangular facet.
///
/// Aequitas carries the SI dimension of each metric at compile time. The
/// quantity wrappers are transparent over `T`, so this boundary adds no
/// storage or dynamic-dispatch cost to the quality result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryFacetQuality<T> {
    /// Smallest interior angle in radians.
    pub min_angle: Angle<T>,
    /// Shortest edge divided by the longest edge.
    pub edge_length_ratio: Dimensionless<T>,
    /// Length of the longest facet edge in canonical SI units.
    pub max_edge_length: Length<T>,
}

/// Error returned when a boundary-facet criterion is outside its domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryFacetQualityCriteriaError {
    /// The minimum angle is not finite or is outside `[0, π/3]`.
    InvalidMinAngle,
    /// The minimum edge-length ratio is not finite or is outside `[0, 1]`.
    InvalidMinEdgeLengthRatio,
    /// The optional maximum edge length is not finite and strictly positive.
    InvalidMaxEdgeLength,
}

/// Explicit acceptance criteria for exposed triangular facets.
///
/// The minimum angle and edge-length ratio are scale invariant. The optional
/// maximum edge length is the facet sizing boundary in canonical SI units.
/// There is no `Default` implementation because surface quality and sizing are
/// consumer policy rather than universal mesh invariants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryFacetQualityCriteria<T> {
    min_angle: Angle<T>,
    min_edge_length_ratio: Dimensionless<T>,
    max_edge_length: Option<Length<T>>,
}

impl<T: Scalar> BoundaryFacetQualityCriteria<T> {
    /// Create validated boundary-facet shape and sizing criteria.
    ///
    /// `min_angle` is an SI angle in radians, `min_edge_length_ratio` is a
    /// dimensionless value in `[0, 1]`, and `max_edge_length` is an SI length
    /// in metres, matching the mesh coordinate contract.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a bound is non-finite or outside its
    /// mathematical domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use aequitas::systems::si::quantities::{Angle, Dimensionless, Length};
    /// use gaia::application::quality::BoundaryFacetQualityCriteria;
    ///
    /// let criteria = BoundaryFacetQualityCriteria::<f64>::try_new(
    ///     Angle::from_base(0.5),
    ///     Dimensionless::from_base(0.5),
    ///     Some(Length::from_base(2.0)),
    /// )
    /// .expect("criteria are finite and valid");
    /// assert_eq!(criteria.max_edge_length(), Some(Length::from_base(2.0)));
    /// ```
    #[must_use = "handle invalid boundary-facet criteria"]
    pub fn try_new(
        min_angle: Angle<T>,
        min_edge_length_ratio: Dimensionless<T>,
        max_edge_length: Option<Length<T>>,
    ) -> Result<Self, BoundaryFacetQualityCriteriaError> {
        let min_angle_value = min_angle.into_base();
        let min_edge_length_ratio_value = min_edge_length_ratio.into_base();
        let zero = <T as NumericElement>::ZERO;
        let one = <T as NumericElement>::ONE;
        let sixty_degrees = <T as RealField>::PI / <T as Scalar>::from_f64(3.0);

        if !<T as NumericElement>::is_finite(min_angle_value)
            || min_angle_value < zero
            || min_angle_value > sixty_degrees
        {
            return Err(BoundaryFacetQualityCriteriaError::InvalidMinAngle);
        }
        if !<T as NumericElement>::is_finite(min_edge_length_ratio_value)
            || min_edge_length_ratio_value < zero
            || min_edge_length_ratio_value > one
        {
            return Err(BoundaryFacetQualityCriteriaError::InvalidMinEdgeLengthRatio);
        }
        if max_edge_length.is_some_and(|length| {
            let value = length.into_base();
            !<T as NumericElement>::is_finite(value) || value <= zero
        }) {
            return Err(BoundaryFacetQualityCriteriaError::InvalidMaxEdgeLength);
        }

        Ok(Self {
            min_angle,
            min_edge_length_ratio,
            max_edge_length,
        })
    }

    /// Return the minimum accepted facet angle in radians.
    #[must_use]
    pub fn min_angle(&self) -> Angle<T> {
        self.min_angle
    }

    /// Return the minimum accepted shortest-to-longest edge ratio.
    #[must_use]
    pub fn min_edge_length_ratio(&self) -> Dimensionless<T> {
        self.min_edge_length_ratio
    }

    /// Return the optional maximum accepted facet edge length.
    #[must_use]
    pub fn max_edge_length(&self) -> Option<Length<T>> {
        self.max_edge_length
    }

    /// Classify one measured boundary facet against these criteria.
    #[must_use]
    pub fn classify(&self, quality: BoundaryFacetQuality<T>) -> BoundaryFacetQualityClass {
        let min_angle = quality.min_angle.into_base();
        let edge_length_ratio = quality.edge_length_ratio.into_base();
        let max_edge_length = quality.max_edge_length.into_base();
        let finite = <T as NumericElement>::is_finite(min_angle)
            && <T as NumericElement>::is_finite(edge_length_ratio)
            && <T as NumericElement>::is_finite(max_edge_length);
        let zero = <T as NumericElement>::ZERO;
        let one = <T as NumericElement>::ONE;
        let pi = <T as RealField>::PI;
        if !finite
            || min_angle < zero
            || min_angle > pi
            || edge_length_ratio <= zero
            || edge_length_ratio > one
            || max_edge_length <= zero
        {
            return BoundaryFacetQualityClass::Invalid;
        }
        if min_angle < self.min_angle.into_base()
            || edge_length_ratio < self.min_edge_length_ratio.into_base()
        {
            return BoundaryFacetQualityClass::PoorShape;
        }
        if self
            .max_edge_length
            .is_some_and(|max_length| max_edge_length > max_length.into_base())
        {
            return BoundaryFacetQualityClass::Oversized;
        }
        BoundaryFacetQualityClass::Accepted
    }
}

/// Quality category assigned to one exposed triangular facet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryFacetQualityClass {
    /// The facet satisfies every configured shape and size bound.
    Accepted,
    /// The facet violates an angle or edge-ratio bound.
    PoorShape,
    /// The facet exceeds the configured maximum edge length.
    Oversized,
    /// The facet is degenerate, non-finite, or outside the metric domain.
    Invalid,
}

/// Counts produced by boundary-facet acceptance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundaryFacetQualityAcceptance {
    /// Number of facets satisfying all criteria.
    pub accepted_facet_count: usize,
    /// Number of facets failing a shape criterion.
    pub poor_shape_facet_count: usize,
    /// Number of facets exceeding the maximum edge length.
    pub oversized_facet_count: usize,
    /// Number of malformed or degenerate facets.
    pub invalid_facet_count: usize,
}

impl BoundaryFacetQualityAcceptance {
    /// Return the number of facets rejected by the criteria.
    #[must_use]
    pub fn rejected_facet_count(self) -> usize {
        self.poor_shape_facet_count + self.oversized_facet_count + self.invalid_facet_count
    }

    /// Return whether every measured boundary facet was accepted.
    #[must_use]
    pub fn passed(self) -> bool {
        self.rejected_facet_count() == 0
    }
}

/// Combined acceptance result for boundary cells and their exposed facets.
///
/// A boundary cell passes only when its tetrahedral cell criteria pass and all
/// of its exposed facets pass the supplied boundary-facet criteria. Facet
/// counts are independent of cell counts: one rejected facet can reject one or
/// more incident boundary cells, but it is counted once in the facet report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryTetrahedralQualityAcceptance {
    /// Number of tetrahedral cells with at least one boundary facet.
    pub boundary_cell_count: usize,
    /// Number of boundary cells satisfying both cell and facet criteria.
    pub accepted_boundary_cell_count: usize,
    /// Number of boundary cells rejected by cell or facet criteria.
    pub rejected_boundary_cell_count: usize,
    /// Number of rejected boundary cells with malformed geometry or topology.
    pub invalid_boundary_cell_count: usize,
    /// Acceptance counts for each geometric boundary facet.
    pub boundary_facet_acceptance: BoundaryFacetQualityAcceptance,
}

impl BoundaryTetrahedralQualityAcceptance {
    /// Return whether every boundary cell and exposed facet was accepted.
    #[must_use]
    pub fn passed(self) -> bool {
        self.rejected_boundary_cell_count == 0 && self.boundary_facet_acceptance.passed()
    }
}

/// Measure one triangular facet in native scalar precision.
#[must_use]
pub fn boundary_facet_quality<T: Scalar>(
    points: [Point3<T>; 3],
) -> Option<BoundaryFacetQuality<T>> {
    if points.iter().any(|point| {
        !<T as NumericElement>::is_finite(point.x)
            || !<T as NumericElement>::is_finite(point.y)
            || !<T as NumericElement>::is_finite(point.z)
    }) {
        return None;
    }

    let [a, b, c] = points;
    let edge_lengths = [(b - a).norm(), (c - b).norm(), (a - c).norm()];
    let zero = <T as NumericElement>::ZERO;
    if edge_lengths
        .iter()
        .any(|&length| !<T as NumericElement>::is_finite(length) || length <= zero)
    {
        return None;
    }
    let max_edge_length = edge_lengths
        .iter()
        .copied()
        .fold(zero, |current, length| current.max_scalar(length));
    let min_angle = min_angle_native(&a, &b, &c);
    let edge_length_ratio = edge_length_ratio_native(&a, &b, &c);
    if !<T as NumericElement>::is_finite(min_angle)
        || !<T as NumericElement>::is_finite(edge_length_ratio)
        || !<T as NumericElement>::is_finite(max_edge_length)
    {
        return None;
    }

    Some(BoundaryFacetQuality {
        min_angle: Angle::from_base(min_angle),
        edge_length_ratio: Dimensionless::from_base(edge_length_ratio),
        max_edge_length: Length::from_base(max_edge_length),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct BoundaryCellState {
    rejected_facet: bool,
    invalid: bool,
}

#[derive(Clone, Copy, Debug)]
struct FaceIncidence {
    owner: usize,
    count: usize,
}

pub(crate) fn assess_boundary_cells<T: Scalar>(
    cell_criteria: &TetrahedralQualityCriteria<T>,
    facet_criteria: &BoundaryFacetQualityCriteria<T>,
    mesh: &IndexedMesh<T>,
) -> Option<BoundaryTetrahedralQualityAcceptance> {
    let tetrahedral_cell_count = mesh
        .cells
        .iter()
        .filter(|cell| cell.element_type == ElementType::Tetrahedron)
        .count();
    if tetrahedral_cell_count == 0 {
        return None;
    }

    let mut face_incidence: HashMap<FaceId, FaceIncidence> =
        HashMap::with_capacity(mesh.faces.len());
    let mut boundary_cells: HashMap<usize, BoundaryCellState> =
        HashMap::with_capacity(tetrahedral_cell_count);

    for (cell_id, cell) in mesh.cells.iter().enumerate() {
        if cell.element_type != ElementType::Tetrahedron {
            continue;
        }
        let invalid_topology = cell.faces.len() != 4
            || cell.faces.iter().enumerate().any(|(index, &face)| {
                face >= mesh.faces.len() || cell.faces[..index].contains(&face)
            });
        if invalid_topology {
            boundary_cells.insert(
                cell_id,
                BoundaryCellState {
                    rejected_facet: false,
                    invalid: true,
                },
            );
        }
        for &face_index in &cell.faces {
            if face_index >= mesh.faces.len() {
                continue;
            }
            let face_id = FaceId::from_usize(face_index);
            match face_incidence.entry(face_id) {
                Entry::Vacant(entry) => {
                    entry.insert(FaceIncidence {
                        owner: cell_id,
                        count: 1,
                    });
                }
                Entry::Occupied(mut entry) => {
                    entry.get_mut().count = entry.get().count.saturating_add(1);
                }
            }
        }
    }

    let mut boundary_facet_acceptance = BoundaryFacetQualityAcceptance::default();
    for (face_id, incidence) in face_incidence {
        if incidence.count != 1 {
            continue;
        }
        let classification = face_quality(mesh, face_id)
            .map_or(BoundaryFacetQualityClass::Invalid, |quality| {
                facet_criteria.classify(quality)
            });
        match classification {
            BoundaryFacetQualityClass::Accepted => {
                boundary_facet_acceptance.accepted_facet_count += 1;
            }
            BoundaryFacetQualityClass::PoorShape => {
                boundary_facet_acceptance.poor_shape_facet_count += 1;
            }
            BoundaryFacetQualityClass::Oversized => {
                boundary_facet_acceptance.oversized_facet_count += 1;
            }
            BoundaryFacetQualityClass::Invalid => {
                boundary_facet_acceptance.invalid_facet_count += 1;
            }
        }

        let state = boundary_cells.entry(incidence.owner).or_default();
        state.rejected_facet |= classification != BoundaryFacetQualityClass::Accepted;
        state.invalid |= classification == BoundaryFacetQualityClass::Invalid;
    }

    let boundary_cell_count = boundary_cells.len();
    let mut accepted_boundary_cell_count = 0;
    let mut rejected_boundary_cell_count = 0;
    let mut invalid_boundary_cell_count = 0;
    for (cell_id, state) in boundary_cells {
        let quality = mesh
            .cells
            .get(cell_id)
            .and_then(|cell| cell_tetrahedron_quality(mesh, cell));
        if state.invalid {
            invalid_boundary_cell_count += 1;
            rejected_boundary_cell_count += 1;
            continue;
        }
        let Some(quality) = quality else {
            invalid_boundary_cell_count += 1;
            rejected_boundary_cell_count += 1;
            continue;
        };
        let cell_class = cell_criteria.classify(quality);
        if cell_class == TetrahedronQualityClass::Accepted && !state.rejected_facet {
            accepted_boundary_cell_count += 1;
        } else {
            rejected_boundary_cell_count += 1;
        }
    }

    Some(BoundaryTetrahedralQualityAcceptance {
        boundary_cell_count,
        accepted_boundary_cell_count,
        rejected_boundary_cell_count,
        invalid_boundary_cell_count,
        boundary_facet_acceptance,
    })
}

fn face_quality<T: Scalar>(
    mesh: &IndexedMesh<T>,
    face_id: FaceId,
) -> Option<BoundaryFacetQuality<T>> {
    let face = mesh.faces.get(face_id);
    if face
        .vertices
        .iter()
        .any(|&vertex| vertex.as_usize() >= mesh.vertices.len())
    {
        return None;
    }
    let points = face.vertices.map(|vertex| *mesh.vertices.position(vertex));
    boundary_facet_quality(points)
}

#[cfg(test)]
mod tests {
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
        TetrahedralQualityCriteria::try_new(
            scalar(2.0),
            scalar(0.7),
            scalar(0.7),
            Some(scalar(1.0)),
        )
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
}
