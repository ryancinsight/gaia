//! Validated direction sets sampled from the unit sphere.

use leto::geometry::UnitVector3;

use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::geometry::primitives::{GeodesicSphere, PrimitiveError, PrimitiveMesh};

/// A deterministic set of unit directions covering a geodesic sphere.
///
/// The directions are the welded vertices of the canonical [`GeodesicSphere`]
/// primitive with unit radius and origin centre.  The primitive's topology
/// therefore determines the cardinality: `10 * frequency² + 2`.  Direction
/// order is stable for a fixed frequency and is part of this type's value
/// semantics, which permits reproducible downstream sampling and serialization.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitSphereDirectionSet {
    directions: Box<[UnitVector3<f64>]>,
}

impl UnitSphereDirectionSet {
    /// Builds a unit-direction set from geodesic subdivision frequency.
    ///
    /// `frequency = 1` produces the twelve icosahedron vertices.  Increasing
    /// the frequency adds directions according to `10 * frequency² + 2`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitiveError::InvalidParam`] when `frequency` is zero.
    pub fn geodesic(frequency: usize) -> Result<Self, PrimitiveError> {
        let mesh = GeodesicSphere {
            radius: 1.0,
            center: Point3r::origin(),
            frequency,
        }
        .build()?;

        let directions = mesh
            .vertices
            .positions()
            .map(unit_direction)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self { directions })
    }

    /// Returns the number of sampled directions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.directions.len()
    }

    /// Reports whether the set contains no directions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }

    /// Borrows the sampled directions in deterministic mesh order.
    #[must_use]
    pub fn as_slice(&self) -> &[UnitVector3<f64>] {
        &self.directions
    }

    /// Iterates over the sampled directions in deterministic mesh order.
    pub fn iter(&self) -> impl Iterator<Item = &UnitVector3<f64>> {
        self.directions.iter()
    }
}

fn unit_direction(point: &Point3r) -> UnitVector3<f64> {
    // GeodesicSphere was built with radius 1 and origin centre, so every
    // welded position is already a unit vector.  Avoiding a second
    // normalization preserves the primitive's exact deterministic values.
    UnitVector3::new_unchecked(Vector3r::new(point.x, point.y, point.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_frequency_with_the_primitive_error() {
        let error = UnitSphereDirectionSet::geodesic(0).expect_err("zero frequency must fail");
        assert_eq!(
            error.to_string(),
            "invalid parameter: frequency must be ≥ 1"
        );
    }

    #[test]
    fn cardinality_matches_geodesic_vertex_formula() {
        for frequency in [1, 2] {
            let directions = UnitSphereDirectionSet::geodesic(frequency).unwrap();
            assert_eq!(directions.len(), 10 * frequency * frequency + 2);
            assert!(!directions.is_empty());
        }
    }

    #[test]
    fn directions_are_unit_and_order_is_reproducible() {
        let first = UnitSphereDirectionSet::geodesic(2).unwrap();
        let second = UnitSphereDirectionSet::geodesic(2).unwrap();
        assert_eq!(first, second);

        // The sphere construction performs a bounded sequence of elementary
        // operations and one square root per coordinate triplet.  32ε is a
        // conservative bound for the resulting norm round-off.
        let norm_bound = f64::EPSILON * 32.0;
        for direction in first.iter() {
            let norm = direction.as_vector().norm();
            assert!((norm - 1.0).abs() <= norm_bound);
        }
    }
}
