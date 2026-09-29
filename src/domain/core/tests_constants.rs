//! Tests for the parent module, extracted from the module body.

use super::*;

/// Compared through a call so the check is not folded into an assertion on a
/// constant (which the workspace lint rejects) while still pinning the
/// documented ordering.
fn assert_ordering(lo: Real, hi: Real, why: &str) {
    assert!(lo < hi, "{why} (got {lo:e} < {hi:e})");
}

/// The docs above state these orderings; an edit that breaks one silently
/// changes which failure mode the pair catches, so they are pinned.
#[test]
fn documented_tolerance_orderings_hold() {
    assert_ordering(
        SEAM_COINCIDENT_LEN_SQ,
        SEAM_DEGENERATE_LEN_SQ,
        "a coincidence floor looser than the degenerate-direction floor would \
             let a zero-direction edge reach a division",
    );
    assert_ordering(
        SEAM_PARAM_MIN_SPAN,
        SEAM_PARAM_DEDUP_TOL,
        "span below dedup",
    );
    assert_ordering(
        SEAM_PARAM_DEDUP_TOL,
        SEAM_PARAM_MARGIN,
        "dedup below margin",
    );
    assert_ordering(
        SEAM_PARAM_MARGIN,
        SNAP_ROUND_EDGE_PARAM_MARGIN,
        "the snap-round screen is wider than the interior test it feeds",
    );
    assert_ordering(
        GWN_OUTSIDE_THRESHOLD,
        GWN_INSIDE_THRESHOLD,
        "the tiebreaker band [OUTSIDE, INSIDE] must be non-empty",
    );
    assert_ordering(
        SLIVER_AREA_RATIO_SQ,
        SLIVER_AREA2D_REL,
        "the classification sliver bound is tighter than the corefine one",
    );
    assert_ordering(
        DEGENERATE_SEGMENT_REL_SQ,
        DEGENERATE_NORMAL_REL_SQ,
        "a segment floor above the normal floor would collapse real segments",
    );
    assert_ordering(
        DEGENERATE_NORMAL_REL_SQ,
        BOOLEAN_DEGENERACY_SIN2_TOL,
        "the repair bound must be looser than the sliver bound, or a repair \
             pass would ignore faces classification already drops",
    );
}

/// `POINT_ON_EDGE_SIN2_TOL` bounds `sin²θ`, so it must lie strictly inside
/// `(0, 1)`: at `0` the test never fires, at `≥ 1` it always does.
#[test]
fn point_on_edge_tolerance_is_a_valid_sin_squared_bound() {
    let tol = POINT_ON_EDGE_SIN2_TOL;
    assert!(
        tol > 0.0 && tol < 1.0,
        "sin²θ bound must be in (0, 1), got {tol:e}"
    );
}

#[test]
fn gwn_thresholds_match_the_derived_symmetric_margin() {
    assert_eq!(GWN_INSIDE_THRESHOLD, 1.0 - GWN_OUTSIDE_THRESHOLD);
    assert_eq!(GWN_OUTSIDE_THRESHOLD.min(0.5 - GWN_OUTSIDE_THRESHOLD), 0.25);
}
