//! Shared primitives for the ASCII mesh importers.
//!
//! OBJ and PLY are both line-oriented ASCII formats whose fields are
//! file-supplied numbers and indices. Both importers have to make the same
//! three decisions — how to read a field as a number, how to read three fields
//! as a point, and what to do when a file names an index that is not there —
//! so those decisions live here once rather than once per format.

use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Point3r, Real};

/// Parse one field as a [`Real`].
///
/// # Errors
/// Returns [`MeshError::Other`] if the field is not a valid number.
pub(crate) fn parse_real(field: &str) -> MeshResult<Real> {
    field
        .parse::<Real>()
        .map_err(|_| MeshError::Other(format!("invalid number: {field}")))
}

/// Parse three fields as a point whose components are all finite.
///
/// `ordinal` is the position of the vertex in the *file*: `0` for the first `v`
/// record, or the first vertex of a PLY vertex element. It is carried into the
/// error so the diagnostic names a location in the source file rather than a
/// post-weld mesh id, which welding may have merged with a different vertex.
/// It is a *diagnostic* only — it is saturated into the reported [`VertexId`]
/// rather than converted, because [`VertexId::from_usize`] panics above
/// `u32::MAX` and a PLY ordinal descends from a file-controlled element count.
///
/// Finiteness is checked separately from parsing because `Real`'s parser
/// *accepts* `"nan"`, `"inf"` and `"-inf"`. A parser that only reports syntax
/// errors therefore lets a non-finite coordinate through, and a non-finite
/// position is not merely inaccurate: it propagates into every predicate,
/// normal and volume that touches the vertex.
///
/// # Errors
/// Returns [`MeshError::Other`] if a field is not a number, and
/// [`MeshError::InvalidCoordinate`] if any component is NaN or infinite.
pub(crate) fn parse_point(fields: [&str; 3], ordinal: usize) -> MeshResult<Point3r> {
    let x = parse_real(fields[0])?;
    let y = parse_real(fields[1])?;
    let z = parse_real(fields[2])?;
    let point = Point3r::new(x, y, z);
    if !(x.is_finite() && y.is_finite() && z.is_finite()) {
        return Err(MeshError::InvalidCoordinate {
            vertex: VertexId::new(u32::try_from(ordinal).unwrap_or(u32::MAX)),
            point,
        });
    }
    Ok(point)
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
