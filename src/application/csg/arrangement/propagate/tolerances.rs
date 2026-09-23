//! Named tolerances for seam propagation.
//!
//! The values live in [`crate::domain::core::constants`] — the SSOT module that
//! states each threshold's unit and the theorem behind it — and are re-exported
//! here under the names this module reads better with. Each entry below says
//! what question it answers, because three of them are deliberately different
//! values in the same parameter space: dedup asks "are these the same
//! crossing?", the margin asks "is this strictly interior?", the span asks "is
//! this interval anything at all?" — and two are squared world lengths at
//! different scales (a degenerate-direction floor and a coincidence noise
//! floor).

use crate::domain::core::constants::{
    POINT_ON_EDGE_SIN2_TOL, SEAM_COINCIDENT_LEN_SQ, SEAM_DEGENERATE_LEN_SQ, SEAM_MIN_HASH_CELL,
    SEAM_PARAM_DEDUP_TOL, SEAM_PARAM_MARGIN, SEAM_PARAM_MIN_SPAN,
};
use crate::domain::core::scalar::Real;

/// Collinearity tolerance for point-on-edge detection in seam propagation.
///
/// A point P is collinear with edge [Va, Vb] if:
///   `|cross(Vb-Va, P-Va)|² < COLLINEAR_TOL_SQ * |Vb-Va|² * |P-Va|²`
///
/// This is a true angular (dimensionless) check:
///   `sin²(angle) < COLLINEAR_TOL_SQ ≈ 1e-6` → `sin(angle) < 1e-3` (0.06°).
///
/// # Theorem — Scale-Invariant Collinearity
///
/// For any edge [Va, Vb] and point P, the cross product satisfies:
///   `|cross(Vb-Va, P-Va)| = |Vb-Va| · |P-Va| · sin(θ)`
/// where θ is the angle between `Vb-Va` and `P-Va`.  Therefore:
///   `|cross|² / (|edge|² · |sp|²) = sin²(θ)`
/// is dimensionless and scale-invariant.  ∎
///
/// The previous check `|cross|² ≤ C · |edge|²` was an absolute
/// perpendicular-distance check (d_perp² ≤ C) that caused false positives
/// at millimetre scale where d_perp < 1 mm for geometrically distant points.
///
/// Delegates to [`POINT_ON_EDGE_SIN2_TOL`] (SSOT).
pub(super) const COLLINEAR_TOL_SQ: Real = POINT_ON_EDGE_SIN2_TOL;

/// Squared length below which an edge, segment, or normal is treated as a
/// point.
///
/// Applied to `|v|²` of a difference of two points, so it is in *squared* world
/// units: `1e-20` is a length of `1e-10`. An edge this short has no direction
/// to project onto, so every downstream parameter would be a division by zero.
///
/// Delegates to [`SEAM_DEGENERATE_LEN_SQ`] (SSOT).
pub(super) const DEGENERATE_LEN_SQ: Real = SEAM_DEGENERATE_LEN_SQ;

/// Squared length below which a point is treated as coincident with the
/// reference point it was measured from.
///
/// Deliberately tighter than [`DEGENERATE_LEN_SQ`] — `1e-30` is a length of
/// `1e-15`, so this fires only at the noise floor of `f64` at unit scale. It is
/// a separate threshold because it answers a different question: the angular
/// collinearity test divides by `|sp|²`, so a point that *is* the edge start
/// would make that ratio `0/0` and be reported as collinear at every angle.
///
/// Delegates to [`SEAM_COINCIDENT_LEN_SQ`] (SSOT).
pub(super) const COINCIDENT_LEN_SQ: Real = SEAM_COINCIDENT_LEN_SQ;

/// Parameter margin for "strictly interior" on a normalised segment.
///
/// A parameter `t ∈ [0, 1]` counts as interior only within
/// `(PARAM_MARGIN, 1 − PARAM_MARGIN)`. An endpoint touch is not an interior
/// crossing — it is already a vertex of the neighbouring face — so accepting it
/// here would inject a zero-length snap segment and split the edge into a
/// duplicate Steiner vertex.
///
/// Named once because both passes need it under the same meaning; it was
/// previously `MARGIN` in one function and `SEG_MARGIN` in the other.
///
/// Delegates to [`SEAM_PARAM_MARGIN`] (SSOT).
pub(super) const PARAM_MARGIN: Real = SEAM_PARAM_MARGIN;

/// Tolerance for merging two intersection parameters that describe the same
/// crossing.
///
/// Both values are parameters on the *same* normalised segment, so this is
/// dimensionless and scale-invariant: `1e-9` is `1e-9` of the edge length in
/// world units. It is needed because one crossing solved from two different
/// axis pairs can land a few ULP apart, and emitting both would split the
/// neighbouring edge twice at points it cannot distinguish.
///
/// Delegates to [`SEAM_PARAM_DEDUP_TOL`] (SSOT).
pub(super) const PARAM_DEDUP_TOL: Real = SEAM_PARAM_DEDUP_TOL;

/// Shortest sub-interval, in parameter space, worth emitting as a snap segment.
///
/// Tighter than [`PARAM_DEDUP_TOL`] because it answers a different question:
/// dedup asks "are these the same crossing?", this asks "is this interval
/// anything at all?". A zero-width interval yields a zero-length segment.
///
/// Delegates to [`SEAM_PARAM_MIN_SPAN`] (SSOT).
pub(super) const PARAM_MIN_SPAN: Real = SEAM_PARAM_MIN_SPAN;

/// Floor on the seam-position spatial-hash cell size, in world units.
///
/// The cell is sized from the longest rim edge (`edge_len / 8`, so the 27-cell
/// neighbourhood stays complete for every rim face in the pass). A degenerate
/// or single-point rim would give a cell of zero and an infinite inverse, so
/// the cell is floored rather than left to the geometry.
///
/// Delegates to [`SEAM_MIN_HASH_CELL`] (SSOT).
pub(super) const MIN_HASH_CELL: Real = SEAM_MIN_HASH_CELL;
