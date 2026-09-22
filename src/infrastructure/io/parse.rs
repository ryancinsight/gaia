//! Shared primitives for the mesh importers.
//!
//! OBJ and PLY are both line-oriented ASCII formats whose fields are
//! file-supplied numbers and indices. Both importers have to make the same
//! three decisions — how to read a field as a number, how to read three fields
//! as a point, and what to do when a file names an index that is not there —
//! so those decisions live here once rather than once per format.
//!
//! The finiteness gate is shared with STL as well, including its binary reader.
//! That reader has no field to parse and therefore nothing to hand to
//! [`parse_point`], but it does have a point to check, so [`finite_point`] is
//! exposed separately rather than folded into the parser.

use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};

/// Parse one field as a [`Real`].
///
/// # Errors
/// Returns [`MeshError::Other`] if the field is not a valid number.
pub(crate) fn parse_real(field: &str) -> MeshResult<Real> {
    field
        .parse::<Real>()
        .map_err(|_| MeshError::Other(format!("invalid number: {field}")))
}

/// The diagnostic for a 3-component value that is not entirely finite.
///
/// `ordinal` is a *file* position, not a mesh id, so it is saturated into the
/// reported [`VertexId`] rather than converted: [`VertexId::from_usize`] panics
/// above `u32::MAX`, and a PLY ordinal descends from a file-controlled element
/// count.
fn invalid_coordinate(ordinal: usize, point: Point3r) -> MeshError {
    MeshError::InvalidCoordinate {
        vertex: VertexId::new(u32::try_from(ordinal).unwrap_or(u32::MAX)),
        point,
    }
}

/// Reject a point if any component is not finite.
///
/// `ordinal` is the position of the vertex in the *file*: `0` for the first `v`
/// record, or the first vertex of a PLY vertex element. It is carried into the
/// error so the diagnostic names a location in the source file rather than a
/// post-weld mesh id, which welding may have merged with a different vertex.
///
/// This is the whole of the finiteness rule; the text importers reach it
/// through [`parse_point`], and the binary STL reader — which reads a record
/// rather than a field — calls it directly.
///
/// Finiteness is checked separately from parsing because `Real`'s parser
/// *accepts* `"nan"`, `"inf"` and `"-inf"`. A parser that only reports syntax
/// errors therefore lets a non-finite coordinate through, and a non-finite
/// position is not merely inaccurate: it propagates into every predicate,
/// normal and volume that touches the vertex.
///
/// # Errors
/// Returns [`MeshError::InvalidCoordinate`] if any component is NaN or infinite.
pub(crate) fn finite_point(point: Point3r, ordinal: usize) -> MeshResult<Point3r> {
    if point.x.is_finite() && point.y.is_finite() && point.z.is_finite() {
        Ok(point)
    } else {
        Err(invalid_coordinate(ordinal, point))
    }
}

/// Parse three fields as a point whose components are all finite.
///
/// # Errors
/// Returns [`MeshError::Other`] if a field is not a number, and
/// [`MeshError::InvalidCoordinate`] if any component is NaN or infinite.
pub(crate) fn parse_point(fields: [&str; 3], ordinal: usize) -> MeshResult<Point3r> {
    let x = parse_real(fields[0])?;
    let y = parse_real(fields[1])?;
    let z = parse_real(fields[2])?;
    finite_point(Point3r::new(x, y, z), ordinal)
}

/// Parse three fields as a normal whose components are all finite.
///
/// A normal is not a position, but it is stored per vertex and used the same
/// way: it is interpolated, shaded with, and compared for degeneracy, and a
/// non-finite one stays in the vertex pool exactly as a non-finite position
/// would. The check is therefore the same check, and the diagnostic reuses
/// [`MeshError::InvalidCoordinate`] — a normal component is a coordinate in the
/// same sense.
///
/// # Errors
/// Returns [`MeshError::Other`] if a field is not a number, and
/// [`MeshError::InvalidCoordinate`] if any component is NaN or infinite.
pub(crate) fn parse_normal(fields: [&str; 3], ordinal: usize) -> MeshResult<Vector3r> {
    let point = parse_point(fields, ordinal)?;
    Ok(Vector3r::new(point.x, point.y, point.z))
}

/// Resolve a file-supplied index into a buffer.
///
/// A malformed file may name any index it likes, and indexing a slice directly
/// turns that into a panic — an input-dependent failure in library code. The
/// crate's lint floor denies that class in principle, but the lint that would
/// catch it (`clippy::indexing_slicing`) is not enabled, so the guard has to be
/// explicit. Keeping it here means OBJ and PLY cannot disagree about it.
///
/// `what` names the referring construct, e.g. `"OBJ face vertex position"`, so
/// the message identifies which index in the file was wrong.
///
/// # Errors
/// Returns [`MeshError::Other`] if `index` does not address `buf`.
pub(crate) fn resolve<T: Copy>(buf: &[T], index: usize, what: &str) -> MeshResult<T> {
    buf.get(index).copied().ok_or_else(|| {
        MeshError::Other(format!(
            "{what} index {index} is out of range ({} present)",
            buf.len()
        ))
    })
}

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_real_rejects_non_numbers() {
        assert!(parse_real("1.5").is_ok());
        assert!(parse_real("-2e3").is_ok());
        assert!(parse_real("").is_err());
        assert!(parse_real("1,5").is_err());
    }

    /// `f64`'s parser accepts these spellings, so a syntax check alone is not a
    /// finiteness check.
    #[test]
    fn parse_real_accepts_non_finite_spellings() {
        assert!(parse_real("nan").unwrap().is_nan());
        assert!(parse_real("inf").unwrap().is_infinite());
        assert!(parse_real("-inf").unwrap().is_infinite());
    }

    #[test]
    fn parse_point_reads_three_finite_fields() {
        let point = parse_point(["1.0", "-2.5", "3"], 0).unwrap();
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, -2.5);
        assert_eq!(point.z, 3.0);
    }

    #[test]
    fn parse_point_rejects_each_non_finite_component() {
        for fields in [["nan", "0", "0"], ["0", "inf", "0"], ["0", "0", "-inf"]] {
            let err = parse_point(fields, 7).unwrap_err();
            assert!(
                matches!(
                    err,
                    MeshError::InvalidCoordinate { vertex, .. } if vertex.as_usize() == 7
                ),
                "expected InvalidCoordinate naming file ordinal 7, got {err:?}"
            );
        }
    }

    /// `VertexId::from_usize` panics above `u32::MAX`, and a PLY ordinal
    /// descends from a file-controlled element count, so the ordinal must be
    /// saturated rather than converted.
    #[test]
    fn parse_point_saturates_an_ordinal_that_cannot_be_a_vertex_id() {
        let err = parse_point(["nan", "0", "0"], usize::MAX).unwrap_err();
        assert!(
            matches!(
                err,
                MeshError::InvalidCoordinate { vertex, .. } if vertex.raw() == u32::MAX
            ),
            "expected the ordinal to saturate, got {err:?}"
        );
    }

    #[test]
    fn finite_point_accepts_a_fully_finite_point() {
        let point = Point3r::new(1.0, -2.0, 3.0);
        let checked = finite_point(point, 0).unwrap();
        assert_eq!((checked.x, checked.y, checked.z), (1.0, -2.0, 3.0));
    }

    #[test]
    fn finite_point_rejects_a_bad_component_on_any_axis() {
        for bad in [
            Point3r::new(Real::NAN, 0.0, 0.0),
            Point3r::new(0.0, Real::NAN, 0.0),
            Point3r::new(0.0, 0.0, Real::NAN),
            Point3r::new(Real::INFINITY, 0.0, 0.0),
            Point3r::new(0.0, Real::NEG_INFINITY, 0.0),
        ] {
            let err = finite_point(bad, 3).unwrap_err();
            assert!(
                matches!(
                    err,
                    MeshError::InvalidCoordinate { vertex, .. } if vertex.as_usize() == 3
                ),
                "{bad:?} was accepted or misreported: {err:?}"
            );
        }
    }

    #[test]
    fn parse_normal_reads_three_finite_fields() {
        let normal = parse_normal(["0", "0", "1"], 0).unwrap();
        assert_eq!((normal.x, normal.y, normal.z), (0.0, 0.0, 1.0));
    }

    /// The gap this closes: `parse_real` accepts `"nan"` and `"inf"`, so a
    /// normal read with it alone reaches the vertex pool non-finite.
    #[test]
    fn parse_normal_rejects_a_non_finite_component() {
        for fields in [["nan", "0", "0"], ["0", "inf", "0"], ["0", "0", "-inf"]] {
            let err = parse_normal(fields, 5).unwrap_err();
            assert!(
                matches!(
                    err,
                    MeshError::InvalidCoordinate { vertex, .. } if vertex.as_usize() == 5
                ),
                "expected InvalidCoordinate naming ordinal 5, got {err:?}"
            );
        }
    }

    #[test]
    fn parse_normal_still_reports_a_genuinely_invalid_field() {
        let err = parse_normal(["1.2.3", "0", "0"], 0).unwrap_err();
        assert!(
            matches!(err, MeshError::Other(ref m) if m.contains("invalid number")),
            "expected the parse failure, got {err:?}"
        );
    }

    #[test]
    fn resolve_returns_the_addressed_element() {
        let buf = [10, 20, 30];
        assert_eq!(resolve(&buf, 0, "x").unwrap(), 10);
        assert_eq!(resolve(&buf, 2, "x").unwrap(), 30);
    }

    #[test]
    fn resolve_rejects_past_the_end_and_empty_buffers() {
        let buf = [10, 20, 30];
        let err = resolve(&buf, 3, "OBJ face vertex position").unwrap_err();
        assert!(
            err.to_string().contains("OBJ face vertex position index 3"),
            "message should name the construct and index: {err}"
        );

        let empty: [u8; 0] = [];
        assert!(resolve(&empty, 0, "x").is_err());
        // The classic wrap-around: `usize::MAX` must not be read as `-1`.
        assert!(resolve(&buf, usize::MAX, "x").is_err());
    }
}
