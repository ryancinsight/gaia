//! Physical and geometric constants for millifluidic design.

use crate::domain::core::scalar::Real;

/// π
pub const PI: Real = std::f64::consts::PI as Real;

/// 2π
pub const TAU: Real = std::f64::consts::TAU as Real;

/// π/2
pub const FRAC_PI_2: Real = std::f64::consts::FRAC_PI_2 as Real;

// ── Unit conversions (to meters) ──────────────────────────────

/// 1 mm in meters.
pub const MM: Real = 1e-3 as Real;

/// 1 μm in meters.
pub const UM: Real = 1e-6 as Real;

/// 1 cm in meters.
pub const CM: Real = 1e-2 as Real;

// ── Millifluidic defaults ─────────────────────────────────────

/// Default channel diameter for millifluidic devices (mm).
pub const DEFAULT_CHANNEL_DIAMETER_MM: Real = 1.0 as Real;

/// Default substrate height (mm).
pub const DEFAULT_SUBSTRATE_HEIGHT_MM: Real = 10.0 as Real;

/// Default wall thickness (mm).
pub const DEFAULT_WALL_THICKNESS_MM: Real = 2.0 as Real;

/// Minimum segment length before it is collapsed (mm).
pub const MIN_SEGMENT_LENGTH_MM: Real = 1e-3 as Real;

// ── Mesh quality defaults ─────────────────────────────────────

/// Minimum acceptable triangle quality score [0, 1].
pub const DEFAULT_MIN_QUALITY: Real = 0.3 as Real;

/// Maximum acceptable aspect ratio.
pub const DEFAULT_MAX_ASPECT_RATIO: Real = 10.0 as Real;

/// Minimum acceptable interior angle (degrees).
pub const DEFAULT_MIN_ANGLE_DEG: Real = 15.0 as Real;

/// Maximum acceptable interior angle (degrees).
pub const DEFAULT_MAX_ANGLE_DEG: Real = 150.0 as Real;

// ── Mesh quality constants needed by various modules ─────────

/// Minimum acceptable interior angle (radians) for quality checks.
pub const DEFAULT_MIN_ANGLE: Real = 15.0 * std::f64::consts::PI as Real / 180.0 as Real;

/// Maximum acceptable equiangle skewness [0, 1].
pub const DEFAULT_MAX_SKEWNESS: Real = 0.8 as Real;

/// Minimum acceptable edge-length ratio [0, 1].
pub const DEFAULT_MIN_EDGE_RATIO: Real = 0.1 as Real;

/// Default channel radius for millifluidic devices (m).
pub const DEFAULT_CHANNEL_RADIUS: Real = 0.5e-3 as Real;

// ── CSG / GWN numerical tolerances (SSOT) ────────────────────────────────────

/// GWN solid-angle denominator guard.
///
/// The van Oosterom–Strackee solid-angle formula uses `atan2(num, den)`.
/// When both `|num|` and `|den|` are below this threshold the face contributes
/// a near-zero solid angle and is skipped to avoid `atan2(0, 0) = NaN`.
///
/// For `f32` meshes use `GWN_DENOMINATOR_GUARD_F32` instead; this constant
/// is only safe for `f64` arithmetic.
pub const GWN_DENOMINATOR_GUARD: Real = 1e-30;

/// Solid-angle clip margin for bounded GWN evaluation.
///
/// Each triangle's solid angle is clamped to `|Ω| ≤ 2π − δ`, where
/// `δ = GWN_SOLID_ANGLE_CLIP`. For a finite triangle, `|Ω| ≤ 2π`, so clamping
/// changes its winding contribution `Ω/(4π)` by at most `δ/(4π)`. If `k`
/// triangles are clipped, the total change is at most `kδ/(4π)` by the
/// triangle inequality. This bound depends on the number of clipped faces;
/// it gives no query-distance threshold or classification guarantee.
pub const GWN_SOLID_ANGLE_CLIP: Real = 1e-6;

/// GWN threshold: `|wn| > GWN_INSIDE_THRESHOLD` → query is classified inside.
///
/// The symmetric thresholds are `t = GWN_OUTSIDE_THRESHOLD` and `1 − t`.
/// For a consistently oriented, watertight solid away from its boundary, the
/// ideal winding magnitudes are 0 outside and 1 inside. Across a planar face,
/// the one-sided limits differ by one, so their midpoint is 0.5. The minimum
/// distance from `{0, 0.5, 1}` to the two decision thresholds is
/// `min(t, 0.5 − t)`, maximized at `t = 0.25`. The resulting values 0.25 and
/// 0.75 give a deterministic maximin margin for those reference magnitudes.
///
/// This is not a finite-precision error bound or a misclassification
/// probability. The generalized winding number can take other values for open,
/// non-manifold, duplicated, or inconsistently oriented triangle soups. Its
/// scale invariance follows from its dimensionless solid-angle definition.
pub const GWN_INSIDE_THRESHOLD: Real = 0.75;

/// GWN threshold: `|wn| < GWN_OUTSIDE_THRESHOLD` → query is classified outside.
///
/// The band `[GWN_OUTSIDE_THRESHOLD, GWN_INSIDE_THRESHOLD]` triggers the
/// tiebreaker predicates in `classify_fragment`. Its value is the symmetric
/// maximin choice derived above.
pub const GWN_OUTSIDE_THRESHOLD: Real = 0.25;

/// Scale-relative tolerance for the nearest-face signed distance tiebreaker.
///
/// # Theorem — Scale Invariance
///
/// The signed distance `d = cp · n / ‖n‖` has unit (length).  The face
/// characteristic scale is `√(area) ≈ √(‖n‖/2)` where `n = ab × ac`.
/// A fragment is coplanar when `|d| < TIEBREAK_SIGN_REL_TOL × √(area)`.
///
/// For a 1 mm edge triangle (area ≈ 4.3 × 10⁻⁷ m², scale ≈ 6.6 × 10⁻⁴ m),
/// the threshold is ≈ 6.6 × 10⁻¹¹ m — well below any physical geometry.
/// For a 1 m edge triangle (scale ≈ 0.7 m), threshold ≈ 7 × 10⁻⁸ m.
///
/// **Previous absolute threshold 1e-9** breaks for meshes at scales ≫ 1 m
/// (threshold too tight) or ≪ 1 mm (threshold too loose). ∎
pub const TIEBREAK_SIGN_REL_TOL: Real = 1e-7;

/// Sliver face exclusion ratio for Phase 4 fragment classification.
///
/// A fragment is considered a numerically degenerate sliver and excluded when
/// `area_sq < SLIVER_AREA_RATIO_SQ * max_edge_sq`.
///
/// # Theorem — Scale-Correct Threshold
///
/// `sqrt(1e-14)` = 1e-7.  A fragment is skipped only when its altitude-to-
/// edge ratio is below 1e-7.  For millifluidic meshes the minimum physically
/// meaningful ratio is ≈ 5e-4 (50 µm altitude on a 4 mm edge), safely above
/// the threshold.  Numerically degenerate slivers produced by near-parallel
/// face intersections have ratios of ~10⁻¹⁰ – 10⁻¹⁵, correctly below. ∎
///
/// **Previous value `1e-10`** (altitude ratio ~3 × 10⁻⁵) incorrectly skipped
/// valid 80:1 aspect-ratio millifluidic faces (50 µm / 4 mm edge).
pub const SLIVER_AREA_RATIO_SQ: Real = 1e-14;

/// CDT co-refinement weld tolerance squared (metres²).
///
/// A snap endpoint is classified as lying on an edge when its 3-D distance
/// to the edge's projection point is less than `2 * sqrt(COREFINE_WELD_TOL_SQ)`.
///
/// Set to `1e-12` to provide a 1 µm weld distance. (Previously `1e-6` resulted
/// in a 1 mm weld distance, causing entire millifluidic channels to web/weld together).
pub const COREFINE_WELD_TOL_SQ: Real = 1e-12;

/// CDT co-refinement edge-endpoint exclusion margin.
///
/// Snap endpoints within this normalised parameter distance of an edge corner
/// are treated as corner snaps, not interior edge Steiner insertions.
pub const COREFINE_EDGE_EPS: Real = 1e-6;

/// Angular tolerance for "this point lies on that edge", as `sin²θ`.
///
/// A point P lies on edge `[Va, Vb]` when
///
/// ```text
/// |cross(Vb − Va, P − Va)|² < POINT_ON_EDGE_SIN2_TOL × |Vb − Va|² × |P − Va|²
/// ```
///
/// # Dimensionless
///
/// `|cross(u, v)| = |u||v|sinθ`, so dividing through by `|u|²|v|²` leaves
/// `sin²θ < POINT_ON_EDGE_SIN2_TOL` — no world units, and therefore the same
/// decision at 10 µm and at 1 km. `1e-6` accepts `sinθ < 1e-3`, i.e. θ below
/// 0.057°.
///
/// The perpendicular-distance form of the same test (`d_perp < tol × |edge|`)
/// is equivalent, which is why both seam passes and `snap_round` share this
/// constant: they differ in what they do with the answer, not in what "on the
/// edge" means.
pub const POINT_ON_EDGE_SIN2_TOL: Real = 1e-6;

/// The historical name for [`POINT_ON_EDGE_SIN2_TOL`], kept because it is public.
///
/// Three names used to spell this one threshold — this one had no callers, and
/// its doc stated `|cross|² < SEAM_COLLINEAR_TOL_SQ × |Vb − Va|²`, which is not
/// the test the call sites perform (they divide by `|Vb − Va|² · |P − Va|²` and
/// so compare against `sin²θ`). The value now has a single definition above;
/// this alias exists so that retiring the name does not break a published API.
///
/// Prefer [`POINT_ON_EDGE_SIN2_TOL`]: it says what the threshold measures.
pub const SEAM_COLLINEAR_TOL_SQ: Real = POINT_ON_EDGE_SIN2_TOL;

/// Maximum Steiner vertices per face during CDT co-refinement.
///
/// When the total count (edge Steiners + interior Steiners) exceeds this
/// bound, `corefine_face` falls back to `midpoint_subdivide` to prevent
/// O(s²) CDT blowup from complex multi-branch junction geometries.
pub const MAX_STEINER_PER_FACE: usize = 32768;

/// Relative AABB expansion factor for broad-phase Boolean operations.
///
/// The mesh-level AABB is expanded by `AABB_RELATIVE_EXPANSION * diagonal`
/// to guard against floating-point precision misses on snapped vertices.
///
/// # Theorem — Scale Correctness
///
/// For a mesh with AABB diagonal `d`, the expansion is `1e-6 · d`.  This
/// ensures the relative guard is constant across scales: a 10 µm mesh
/// expands by ≈ 10 fm (sub-atomic, harmless), while a 1 m mesh expands by
/// ≈ 1 µm.  The previous absolute 1e-6 m expansion was 10 % of a 10 µm
/// mesh's diagonal — large enough to merge disjoint features. ∎
pub const AABB_RELATIVE_EXPANSION: Real = 1e-6;

/// Relative degenerate-normal threshold factor for CDT corefine and
/// fragment classification.
///
/// A cross-product normal `n = (B−A) × (C−A)` is degenerate when
/// `‖n‖² < DEGENERATE_NORMAL_REL_SQ · ‖B−A‖² · ‖C−A‖²`.
///
/// # Theorem — Dimensionless Bound
///
/// `‖n‖² = ‖B−A‖² ‖C−A‖² sin²θ` where θ is the included angle.  The
/// threshold `sin²θ < 1e-20` triggers only for θ < 1e-10 rad (≈ 6e-9°),
/// regardless of scale.  Previously the absolute check `‖n‖² < 1e-20`
/// triggered for any triangle with edge lengths below ≈ 3e-10, which
/// misclassified micro-scale millifluidic geometry. ∎
pub const DEGENERATE_NORMAL_REL_SQ: Real = 1e-20;

/// Relative degenerate-segment threshold factor.
///
/// A snap segment with `‖end − start‖² < DEGENERATE_SEGMENT_REL_SQ · diag²`
/// is collapsed, where `diag` is the face's maximum edge length.
///
/// # Theorem — Scale Correctness
///
/// At any scale, a segment shorter than `1e-12 · max_edge` is below
/// double-precision resolution for that geometry.  The previous absolute
/// threshold `1e-24` is `(1e-12)²` which is correct for unit-scale meshes
/// but too strict at macro scale (1 m mesh: segments up to 1e-12 m would
/// be kept, wasting CDT effort on sub-picometer features). ∎
pub const DEGENERATE_SEGMENT_REL_SQ: Real = 1e-24;

/// Relative 2-D projected-area threshold for sliver detection in corefine.
///
/// A boundary polygon is degenerate when `|area2D| < SLIVER_AREA2D_REL · ∑edge²`.
///
/// # Theorem — Scale Independence
///
/// The 2-D shoelace area scales as length², so comparing against ∑edge²
/// yields a dimensionless ratio that is independent of mesh scale. ∎
pub const SLIVER_AREA2D_REL: Real = 1e-10;

/// Relative interval-overlap tolerance for T-T intersection segment
/// computation.
///
/// Two 1-D intervals `[t₁_min, t₁_max]` and `[t₂_min, t₂_max]` are
/// considered non-overlapping when `t_enter > t_leave + EPS · span` where
/// `span = max(|t₁_max − t₁_min|, |t₂_max − t₂_min|, diag)`.
///
/// # Theorem — Scale Independence
///
/// The 1-D projections onto the intersection line scale linearly with
/// mesh dimensions.  Using a relative tolerance ensures the gap test
/// is uniform across scales. ∎
pub const INTERVAL_OVERLAP_REL: Real = 1e-12;

// ── Seam propagation and snap-round thresholds (SSOT) ───────────────────────
//
// The seam passes (general propagation, barrel cap-seam injection) ask three
// different questions about the same parameter space, and need two
// squared-length floors at different scales. Every value below is defined here
// and re-exported under its local name by the module that uses it, so a
// threshold has one definition and one stated unit — the arrangement passes
// previously kept private copies, one of which had drifted from this module's
// doc for the same threshold.

/// Squared world length below which a direction (edge, segment, normal) is
/// treated as a point. `(squared world length)`: `1e-20` is a length of `1e-10`.
///
/// An edge this short has no direction to project onto, so every downstream
/// parameter would be a division by zero.
pub const SEAM_DEGENERATE_LEN_SQ: Real = 1e-20;

/// Squared world length below which two points count as coincident.
/// `(squared world length)`: `1e-30` is `1e-15`, the `f64` noise floor at unit
/// scale.
///
/// Deliberately tighter than [`SEAM_DEGENERATE_LEN_SQ`]: the angular
/// collinearity test divides by `|P − Va|²`, so a point that *is* the edge start
/// would make that ratio `0/0` and read as collinear at every angle.
pub const SEAM_COINCIDENT_LEN_SQ: Real = 1e-30;

/// Parameter margin for "strictly interior" on a normalised segment.
/// `(dimensionless parameter)`: a `t ∈ (SEAM_PARAM_MARGIN, 1 − SEAM_PARAM_MARGIN)`
/// counts as interior.
///
/// An endpoint touch is not an interior crossing — it is already a vertex of the
/// neighbouring face — so accepting it here would inject a zero-length snap
/// segment and split the edge at a duplicate Steiner vertex.
pub const SEAM_PARAM_MARGIN: Real = 1e-7;

/// Tolerance for merging two intersection parameters that describe the same
/// crossing. `(dimensionless parameter)` on the same normalised segment.
///
/// One crossing solved from two different axis pairs can land a few ULP apart;
/// emitting both would split the neighbouring edge twice at points it cannot
/// distinguish.
pub const SEAM_PARAM_DEDUP_TOL: Real = 1e-9;

/// Shortest sub-interval, in parameter space, worth emitting as a snap segment.
/// `(dimensionless parameter)`.
///
/// Tighter than [`SEAM_PARAM_DEDUP_TOL`] because it answers a different
/// question: dedup asks "are these the same crossing?", this asks "is this
/// interval anything at all?" — a zero-width interval yields a zero-length
/// segment.
pub const SEAM_PARAM_MIN_SPAN: Real = 1e-12;

/// Floor on the seam-position spatial-hash cell size. `(world length)`.
///
/// The cell is sized from the longest rim edge (`edge_len / 8`, so the 27-cell
/// neighbourhood stays complete for every rim face in the pass); a degenerate or
/// single-point rim would give a cell of zero and an infinite inverse, so the
/// cell is floored rather than left to the geometry.
pub const SEAM_MIN_HASH_CELL: Real = 1e-6;

/// Endpoint margin for the snap-round edge-parameter test `t`.
/// `(dimensionless parameter)`: candidate split vertices must be strictly
/// interior to the edge and at least this fraction away from either endpoint.
///
/// Wider than [`SEAM_PARAM_MARGIN`] because it screens which vertices are worth
/// splitting at, rather than deciding whether a found crossing is interior.
pub const SNAP_ROUND_EDGE_PARAM_MARGIN: Real = 5e-3;

// ── Boolean repair, 2-D clip and self-intersection thresholds (SSOT) ────────
//
// Folded in from three private clusters so that a threshold is defined once and
// its *dimension* is stated where it lives. Three of these are not
// scale-invariant, which the dimension notes record rather than hide: they are
// the same defect class the seam pass's near-parallel threshold was measured and
// fixed for, and each needs its own measurement before it moves.

/// Squared world length below which two vertices of a Boolean result are merged
/// as coincident. `(squared world length)`: `1e-18` is a length of `1e-9`.
///
/// The degenerate-face collapse pass treats a degenerate face whose shortest
/// edge is below this as a coincident-vertex pair and unions the two vertices.
pub const BOOLEAN_COINCIDENT_LEN_SQ: Real = 1e-18;

/// `sin²θ` below which the collapse pass repairs a face as degenerate.
/// `(dimensionless)`: the pass tests `cross² < BOOLEAN_DEGENERACY_SIN2_TOL ×
/// |ab|² × |ac|²`, the same shape as [`DEGENERATE_NORMAL_REL_SQ`], so a face's
/// classification does not depend on the mesh's scale.
///
/// Distinct from [`DEGENERATE_NORMAL_REL_SQ`] (`1e-20`) because the two answer
/// different questions: that one decides whether a *fragment* is a numerical
/// sliver worth excluding from classification, this one whether a *face* is worth
/// repairing — and a repair pass may reasonably act on geometry that
/// classification should not silently drop. The value is the one the pass already
/// had: at unit edge length the previous `cross² / max_edge² < 1e-12` was the
/// same test, so unit-scale behaviour is unchanged and only the scale dependence
/// is gone.
pub const BOOLEAN_DEGENERACY_SIN2_TOL: Real = 1e-12;

/// The historical name for [`BOOLEAN_DEGENERACY_SIN2_TOL`], kept because it is
/// public.
///
/// The name asserts a squared length, and that claim *was* the defect: the pass
/// compared a `cross² / max_edge²` ratio — which carries a `length²` — against
/// this value, so the same face at a given `sin²θ` was classified differently at
/// different mesh scales. The value is dimensionless now; prefer
/// [`BOOLEAN_DEGENERACY_SIN2_TOL`], which says what the threshold measures.
pub const BOOLEAN_DEGENERACY_LEN_SQ: Real = BOOLEAN_DEGENERACY_SIN2_TOL;

/// Weld distance for the 2-D clip CDT's point grid and intersection welds.
/// `(world length)` in the clip plane, which spans the same units as the mesh
/// it clips.
pub const CLIP2D_WELD_LEN: Real = 1e-8;

/// Parameter margin for "strictly interior" on a 2-D clip segment.
/// `(dimensionless parameter)`: an edge crossing counts only within
/// `(CLIP2D_INTERSECT_PARAM_MARGIN, 1 − CLIP2D_INTERSECT_PARAM_MARGIN)`.
///
/// Tighter than [`SEAM_PARAM_MARGIN`] on purpose: here the endpoints are polygon
/// corners whose own edges already carry the crossing, so accepting a
/// corner-touching crossing would shatter an edge at a vertex that exists.
pub const CLIP2D_INTERSECT_PARAM_MARGIN: Real = 1e-10;

/// Parameter margin for the 2-D clip CDT's shattered-edge collector.
/// `(dimensionless parameter)`.
///
/// A different question from [`CLIP2D_INTERSECT_PARAM_MARGIN`] — it screens
/// which points along an edge are interior enough to become Steiner vertices,
/// rather than whether a found crossing is interior — so it keeps its own value
/// even though the two are within two orders of magnitude.
pub const CLIP2D_SHATTER_PARAM_MARGIN: Real = 1e-8;

/// Squared distance tolerance for the 2-D clip CDT's shattered-edge collector.
/// `(squared world length)`: `1e-12` is a length of `1e-6`.
pub const CLIP2D_SHATTER_DIST_SQ: Real = 1e-12;

/// `sin²θ` between two face normals below which the self-intersection narrow
/// phase treats their planes as parallel, and conservatively reports no
/// intersection. `(dimensionless)`.
///
/// The test is `|n₁ × n₂|² < TOL × |n₁|²|n₂|²`, which divides out to
/// `sin²θ(n₁, n₂)`. The previous form compared `|n₁ × n₂|²` directly — a
/// quantity that scales as `L⁸` for unnormalised normals — so the same pair of
/// planes was judged parallel at one mesh scale and not at another.
pub const SELF_INTERSECT_NORMAL_SIN2_TOL: Real = 1e-20;

/// Plane-band width for the self-intersection narrow phase, relative to the
/// longest edge of the triangles involved. `(dimensionless)`.
///
/// A vertex counts as lying *on* the opposing plane when its true distance to
/// that plane is within `SELF_INTERSECT_PLANE_REL × edge_scale`. The previous
/// form compared the plane equation `n · p + d` directly, which scales as `L³`
/// for an unnormalised normal, so the band's real width drifted with mesh scale.
pub const SELF_INTERSECT_PLANE_REL: Real = 1e-10;

/// The historical name for [`SELF_INTERSECT_NORMAL_SIN2_TOL`], kept because it
/// is public.
///
/// Its name asserts a squared direction and its doc recorded the `L⁸` scaling as
/// an open item; the quantity is dimensionless once the normals are normalised,
/// which is what made the old test scale-dependent. Prefer the constant above.
pub const SELF_INTERSECT_LINE_DIR_SQ_EPS: Real = SELF_INTERSECT_NORMAL_SIN2_TOL;

/// The historical name for [`SELF_INTERSECT_PLANE_REL`], kept because it is
/// public.
///
/// It was an absolute plane-equation value (`L³` for an unnormalised normal);
/// the band is now relative to the triangle scale. Prefer the constant above.
pub const SELF_INTERSECT_PLANE_EPS: Real = SELF_INTERSECT_PLANE_REL;

/// Vertex-consolidation distance for merging cross-mesh duplicates during
/// multi-resolution arrangement. `(world length)`.
///
/// Twice the weld of the pass that produces the near-duplicates, which is what
/// makes it correct. The passes in this family inherit from different upstream
/// tolerances rather than being stale copies: this one is `2e-4` (from a `1e-4`
/// weld) and `patch_small_boundary_holes` merges at `2e-3`. Unifying the numbers
/// would break whichever pass moved.
pub const MULTI_MESH_CONSOLIDATE_LEN: Real = 2e-4;

#[cfg(test)]
mod tests {
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
}
