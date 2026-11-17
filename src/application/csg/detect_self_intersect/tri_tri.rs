//! Narrow phase: Möller (1997) triangle-triangle test.
//!
//! Split out of `detect_self_intersect.rs` so the broad phase - BVH candidate
//! generation plus the AABB pass - keeps one file and the exact pair test keeps
//! another. `tri_tri_intersects` is the only entry the parent calls; the
//! interval and projection helpers below exist for it alone.

use crate::domain::core::constants::{SELF_INTERSECT_NORMAL_SIN2_TOL, SELF_INTERSECT_PLANE_REL};
use crate::domain::core::scalar::{Point3r, Real};

/// Return `true` iff triangles `ta` and `tb` properly intersect.
///
/// Implements the interval-overlap method from Möller (1997):
///
/// 1. Compute plane P1 of `ta` (n1, d1).  Test signed distances of `tb` verts.
///    If all same sign → `tb` lies on one side → no intersection.
/// 2. Compute plane P2 of `tb` (n2, d2).  Test signed distances of `ta` verts.
///    If all same sign → `ta` lies on one side → no intersection.
/// 3. Intersection line `L = n1 × n2`.  If |L|² < ε² → coplanar → `false`.
/// 4. Project all 6 vertices onto the dominant axis of `L`.  Compute overlap
///    intervals for `ta` and `tb`; return whether they overlap.
#[must_use]
#[expect(
    clippy::similar_names,
    reason = "paired triangle interval notation is standard in triangle-triangle overlap tests"
)]
pub(super) fn tri_tri_intersects(ta: &[Point3r; 3], tb: &[Point3r; 3]) -> bool {
    // Both tolerances below are relative to the geometry's own scale, so the same
    // configuration is decided the same way at 1 µm and at 1 km.
    let edge_scale = longest_edge(ta).max(longest_edge(tb));
    let plane_band = SELF_INTERSECT_PLANE_REL * edge_scale;

    // ── Plane of ta ──────────────────────────────────────────────────────────
    let n1 = (ta[1] - ta[0]).cross(ta[2] - ta[0]);
    let n1_norm = n1.norm();
    if n1_norm <= 0.0 {
        // Degenerate `ta` has no plane, so nothing can straddle it.
        return false;
    }
    let d1 = -n1.dot(ta[0].coords);

    // True signed distances of tb vertices to plane of ta (lengths).
    let db = [
        (n1.dot(tb[0].coords) + d1) / n1_norm,
        (n1.dot(tb[1].coords) + d1) / n1_norm,
        (n1.dot(tb[2].coords) + d1) / n1_norm,
    ];
    // All same strict sign → tb on one side → no intersection.
    if (db[0] > 0.0 && db[1] > 0.0 && db[2] > 0.0) || (db[0] < 0.0 && db[1] < 0.0 && db[2] < 0.0) {
        return false;
    }

    // ── Plane of tb ──────────────────────────────────────────────────────────
    let n2 = (tb[1] - tb[0]).cross(tb[2] - tb[0]);
    let n2_norm = n2.norm();
    if n2_norm <= 0.0 {
        return false;
    }
    let d2 = -n2.dot(tb[0].coords);

    // True signed distances of ta vertices to plane of tb (lengths).
    let da = [
        (n2.dot(ta[0].coords) + d2) / n2_norm,
        (n2.dot(ta[1].coords) + d2) / n2_norm,
        (n2.dot(ta[2].coords) + d2) / n2_norm,
    ];
    if (da[0] > 0.0 && da[1] > 0.0 && da[2] > 0.0) || (da[0] < 0.0 && da[1] < 0.0 && da[2] < 0.0) {
        return false;
    }

    // ── Intersection line ─────────────────────────────────────────────────────
    // `|n₁ × n₂| / (|n₁||n₂|) = sinθ` between the planes' normals, so the
    // parallel test is `sin²θ < TOL` and carries no world units. Comparing
    // `|n₁ × n₂|²` directly would scale as `length⁸`.
    let l_dir = n1.cross(n2);
    let sin2 = l_dir.norm_squared() / (n1_norm * n1_norm * n2_norm * n2_norm);
    if sin2 < SELF_INTERSECT_NORMAL_SIN2_TOL {
        // Near-parallel planes: the intersection line is ill-conditioned, so
        // conservatively report no intersection.
        return false;
    }

    // Project onto the dominant axis of L for numerical stability.
    let axis = dominant_axis(&l_dir);
    let p_ta = [ta[0][axis], ta[1][axis], ta[2][axis]];
    let p_tb = [tb[0][axis], tb[1][axis], tb[2][axis]];

    // ── Interval computation & overlap test ───────────────────────────────────
    let (ta_min, ta_max) = tri_interval(&p_ta, &da, plane_band);
    let (tb_min, tb_max) = tri_interval(&p_tb, &db, plane_band);

    ta_min <= tb_max && tb_min <= ta_max
}

/// Longest edge of a triangle — the scale its tolerances are relative to.
#[inline]
fn longest_edge(t: &[Point3r; 3]) -> Real {
    (t[1] - t[0])
        .norm()
        .max((t[2] - t[1]).norm())
        .max((t[0] - t[2]).norm())
}

/// Axis (0=X, 1=Y, 2=Z) along which `v` has its maximum absolute component.
#[inline]
fn dominant_axis(v: &leto::geometry::Vector3<Real>) -> usize {
    let ax = v[0].abs();
    let ay = v[1].abs();
    let az = v[2].abs();
    if ax >= ay && ax >= az {
        0
    } else if ay >= az {
        1
    } else {
        2
    }
}

/// Compute the projection interval `[min_t, max_t]` on the intersection line
/// for a triangle with vertex projections `pv` and signed distances `dv` to the
/// opposing plane.
///
/// ## Edge crossing detection (all-edge approach)
///
/// A vertex at `dv[i] ≈ 0` contributes its projection `pv[i]` directly.
/// An edge `[i,j]` with `sign(dv[i]) ≠ sign(dv[j])` (strictly, both
/// `|dv| ≥ plane_band`) contributes the interpolated crossing parameter.
///
/// `plane_band` is a length, already scaled to the triangles involved, so the
/// same geometry takes the same branch at any mesh scale.
///
/// This all-edge approach correctly handles degenerate cases where one vertex
/// lies exactly on the opposing plane (the standard "isolated vertex" method
/// fails for `d[isolated] = 0` with mixed-sign other vertices).
///
/// ## Theorem (Interval Correctness)
///
/// For any triangle strictly straddling a plane (early exits have passed),
/// the triangle intersects the plane in a segment.  The projection of that
/// segment onto the dominant axis is exactly `[min_t, max_t]`.  The all-edge
/// scan finds all crossing points (edges + on-plane vertices) and the
/// min/max of their projections equals the true interval endpoints.  QED.
fn tri_interval(pv: &[Real; 3], dv: &[Real; 3], plane_band: Real) -> (Real, Real) {
    let mut t = [0.0_f64; 3];
    let mut n = 0usize;

    // On-plane vertices.
    for i in 0..3 {
        if dv[i].abs() < plane_band && n < 3 {
            t[n] = pv[i];
            n += 1;
        }
    }

    // Proper sign-change edge crossings (both endpoints clear of coplanar band).
    for &(i, j) in &[(0usize, 1usize), (1, 2), (2, 0)] {
        let di = dv[i];
        let dj = dv[j];
        if di.abs() >= plane_band && dj.abs() >= plane_band && (di > 0.0) != (dj > 0.0) && n < 3 {
            t[n] = lerp_crossing(pv[i], pv[j], di, dj, plane_band);
            n += 1;
        }
    }

    if n == 0 {
        // No crossings — degenerate; return an empty interval.
        return (Real::INFINITY, Real::NEG_INFINITY);
    }

    let mut tmin = t[0];
    let mut tmax = t[0];
    for &ti in &t[1..n] {
        if ti < tmin {
            tmin = ti;
        }
        if ti > tmax {
            tmax = ti;
        }
    }
    (tmin, tmax)
}

/// Linear interpolation to find where the signed distance crosses zero on edge
/// `[pa, pb]` with distances `da` and `db`.
///
/// Returns `pa + (pb - pa) * da / (da - db)` — the projection parameter at
/// which the plane-crossing occurs. `plane_band` (a length scaled to the
/// geometry) guards the division: below it the two distances are
/// indistinguishable and the midpoint is returned.
#[inline]
fn lerp_crossing(pa: Real, pb: Real, da: Real, db: Real, plane_band: Real) -> Real {
    let denom = da - db;
    if denom.abs() < plane_band {
        return (pa + pb) * 0.5;
    }
    pa + (pb - pa) * da / denom
}
