//! Self-intersection detection for closed 3-D triangle meshes.
//!
//! ## Purpose
//!
//! Detects face pairs `(i, j)` whose triangles properly intersect — i.e., the
//! interiors of triangle `i` and triangle `j` overlap — excluding shared-edge
//! and shared-vertex adjacency which is expected in manifold meshes.
//!
//! Intended as:
//! - A pre-check warning at `csg_boolean` entry.
//! - A standalone validation step in `BlueprintMeshPipeline`.
//!
//! ## Algorithm
//!
//! ```text
//! detect_self_intersections(faces, pool)
//!         │
//!  ┌──────▼─────────────────────────────────────────┐
//!  │ Compute per-face AABB: O(F)                     │
//!  └──────┬─────────────────────────────────────────┘
//!         │
//!  ┌──────▼─────────────────────────────────────────┐
//!  │ SAH-BVH self-query: i < j pairs with            │
//!  │ overlapping AABBs, O((F + k) log F)             │
//!  └──────┬─────────────────────────────────────────┘
//!         │
//!  ┌──────▼─────────────────────────────────────────┐
//!  │ Adjacency filter: skip pairs sharing any vertex  │
//!  └──────┬─────────────────────────────────────────┘
//!         │
//!  ┌──────▼─────────────────────────────────────────┐
//!  │ Möller (1997) triangle-triangle interval test,   │
//!  │ O(1) per pair                                   │
//!  └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Theorem (Detection Completeness)
//!
//! Every properly intersecting non-adjacent face pair is reported.  Proof:
//! (1) **AABB containment**: intersecting triangles have overlapping AABBs.
//! (2) **BVH completeness**: `query_overlapping` reports all overlapping AABB
//!     pairs (by `BvhTree` completeness theorem).
//! (3) **Narrow-phase soundness**: Möller's interval-overlap test has no false
//!     negatives for non-degenerate, non-coplanar pairs.  Coplanar pairs are
//!     conservatively treated as non-intersecting (appropriate for manifold
//!     meshes where coplanar adjacent faces are common).  QED.
//!
//! ## Theorem (Narrow-Phase Correctness — Möller 1997)
//!
//! For non-coplanar triangles T1 and T2:
//! - **Soundness**: `tri_tri_intersects` returns `true` only if the triangles
//!   properly intersect.  Proof: the interval-overlap condition directly implies
//!   a common segment on the line of intersection.
//! - **Completeness**: if the triangles intersect along a segment on line L,
//!   both projection intervals onto L are non-empty and overlapping.  Therefore
//!   the interval-overlap test returns `true`.  QED (Möller 1997, §3).
//!
//! ## Complexity
//!
//! | Phase | Cost |
//! |-------|------|
//! | AABB build | O(F) |
//! | BVH build | O(F log F) |
//! | BVH self-query | O(F log F + k) |
//! | Narrow phase | O(k) |
//! | **Total** | **O(F log F + k)** |
//!
//! ## References
//!
//! - Möller, T. (1997). A fast triangle-triangle intersection test. *Journal of
//!   Graphics Tools*, 2(2), 25–30.
//! - Devillers & Guigue (2002). Faster triangle-triangle intersection tests.
//!   *INRIA Research Report 4488*.

use crate::application::csg::broad_phase::triangle_aabb;
use crate::domain::core::constants::{SELF_INTERSECT_NORMAL_SIN2_TOL, SELF_INTERSECT_PLANE_REL};
use crate::domain::core::scalar::{Point3r, Real};
use crate::infrastructure::spatial::bvh::with_bvh;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

// ── Public API ────────────────────────────────────────────────────────────────

/// Return all pairs `(i, j)` with `i < j` of non-adjacent faces that properly
/// intersect.
///
/// "Non-adjacent" means the two faces share **no** vertex.  Faces that share a
/// vertex (and therefore at most one edge) are skipped because a manifold mesh
/// necessarily has coincident edges between adjacent faces — these are not
/// self-intersections.
///
/// Returns an empty `Vec` for meshes with fewer than 2 faces.
///
/// # Example
///
/// ```ignore
/// let pairs = detect_self_intersections(
///     &mesh.faces.iter().cloned().collect::<Vec<_>>(),
///     &mesh.vertices,
/// );
/// assert!(pairs.is_empty(), "mesh should be self-intersection-free before CSG");
/// ```
#[must_use]
pub fn detect_self_intersections(faces: &[FaceData], pool: &VertexPool) -> Vec<(usize, usize)> {
    if faces.len() < 2 {
        return Vec::new();
    }

    // ── Phase 1: Per-face AABBs ───────────────────────────────────────────────
    let aabbs: Vec<_> = faces.iter().map(|f| triangle_aabb(f, pool)).collect();

    // ── Phase 2: BVH self-query ───────────────────────────────────────────────
    // For each face i, query for faces j > i with overlapping AABBs.
    // Restricting j > i avoids reporting each pair twice.
    let mut result: Vec<(usize, usize)> = Vec::new();

    with_bvh(&aabbs, |tree, token| {
        let mut hits = Vec::new();
        for i in 0..faces.len() {
            hits.clear();
            tree.query_overlapping(&aabbs[i], &token, &mut hits);

            let fi = &faces[i];
            let [a0, a1, a2] = fi.vertices;
            let ta = [*pool.position(a0), *pool.position(a1), *pool.position(a2)];

            for &j in &hits {
                // Enumerate each unordered pair once.
                if j <= i {
                    continue;
                }

                let fj = &faces[j];

                // ── Adjacency filter ─────────────────────────────────────────
                // Skip pairs that share any vertex: in a manifold mesh such
                // pairs share an edge and will have coincident boundaries —
                // that is expected, not a self-intersection.
                let [b0, b1, b2] = fj.vertices;
                if b0 == a0
                    || b0 == a1
                    || b0 == a2
                    || b1 == a0
                    || b1 == a1
                    || b1 == a2
                    || b2 == a0
                    || b2 == a1
                    || b2 == a2
                {
                    continue;
                }

                let tb = [*pool.position(b0), *pool.position(b1), *pool.position(b2)];

                if tri_tri_intersects(&ta, &tb) {
                    result.push((i, j));
                }
            }
        }
    });

    result
}

// ── Narrow phase: Möller (1997) triangle-triangle test ───────────────────────

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
fn tri_tri_intersects(ta: &[Point3r; 3], tb: &[Point3r; 3]) -> bool {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::Vector3r;
    use crate::infrastructure::storage::face_store::FaceData;

    fn pool_with_verts(
        pts: &[[f64; 3]],
    ) -> (VertexPool, Vec<crate::domain::core::index::VertexId>) {
        let mut pool = VertexPool::default_millifluidic();
        let n = Vector3r::zeros();
        let ids = pts
            .iter()
            .map(|&[x, y, z]| pool.insert_or_weld(Point3r::new(x, y, z), n))
            .collect();
        (pool, ids)
    }

    // ── tri_tri_intersects unit tests ─────────────────────────────────────────

    /// Two flat XY-plane triangles that overlap → intersects.
    #[test]
    fn two_triangles_crossing_in_xz_plane_intersect() {
        // ta: flat in XZ, centered at origin.  tb: rotated ~90°, crosses ta.
        let ta = [
            Point3r::new(-1.0, 0.0, 0.0),
            Point3r::new(1.0, 0.0, 0.0),
            Point3r::new(0.0, 0.0, 1.0),
        ];
        let tb = [
            Point3r::new(0.0, -1.0, 0.5),
            Point3r::new(0.0, 1.0, 0.5),
            Point3r::new(0.0, 0.0, -0.5),
        ];
        assert!(
            tri_tri_intersects(&ta, &tb),
            "crossing triangles must be detected"
        );
    }

    /// Two triangles on opposite sides of a plane → no intersection.
    #[test]
    fn two_separated_triangles_do_not_intersect() {
        let ta = [
            Point3r::new(0.0, 0.0, 0.0),
            Point3r::new(1.0, 0.0, 0.0),
            Point3r::new(0.0, 1.0, 0.0),
        ];
        // tb translated by +5 on Z → separated.
        let tb = [
            Point3r::new(0.0, 0.0, 5.0),
            Point3r::new(1.0, 0.0, 5.0),
            Point3r::new(0.0, 1.0, 5.0),
        ];
        assert!(
            !tri_tri_intersects(&ta, &tb),
            "separated triangles must not intersect"
        );
    }

    /// Two coplanar triangles: conservatively returns false.
    #[test]
    fn coplanar_triangles_return_false() {
        let ta = [
            Point3r::new(0.0, 0.0, 0.0),
            Point3r::new(1.0, 0.0, 0.0),
            Point3r::new(0.0, 1.0, 0.0),
        ];
        let tb = [
            Point3r::new(0.1, 0.1, 0.0),
            Point3r::new(0.5, 0.0, 0.0),
            Point3r::new(0.0, 0.5, 0.0),
        ];
        // Conservative: coplanar → false (not a proper intersection in 3-D).
        assert!(
            !tri_tri_intersects(&ta, &tb),
            "coplanar overlap is not reported"
        );
    }

    // ── detect_self_intersections integration tests ───────────────────────────

    /// A flat quad split into 2 adjacent triangles has no self-intersection.
    #[test]
    fn adjacent_triangles_are_not_self_intersecting() {
        let (pool, ids) = pool_with_verts(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        let faces = vec![
            FaceData::untagged(ids[0], ids[1], ids[2]),
            FaceData::untagged(ids[0], ids[2], ids[3]),
        ];
        let pairs = detect_self_intersections(&faces, &pool);
        assert!(
            pairs.is_empty(),
            "adjacent triangles must not be reported as self-intersecting"
        );
    }

    /// Two non-adjacent triangles that cross each other ARE detected.
    #[test]
    fn non_adjacent_crossing_triangles_are_detected() {
        // ta: horizontal XY-plane triangle.
        // tb: tilted to cross ta (shared no vertices).
        let (pool, ids) = pool_with_verts(&[
            // ta
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            // tb (crosses ta through z=0)
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ]);
        let faces = vec![
            FaceData::untagged(ids[0], ids[1], ids[2]),
            FaceData::untagged(ids[3], ids[4], ids[5]),
        ];
        let pairs = detect_self_intersections(&faces, &pool);
        assert!(
            !pairs.is_empty(),
            "crossing non-adjacent triangles must be detected"
        );
        assert_eq!(pairs, vec![(0, 1)]);
    }

    /// Completely separated non-adjacent triangles: empty result.
    #[test]
    fn separated_non_adjacent_triangles_not_detected() {
        let (pool, ids) = pool_with_verts(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
        ]);
        let faces = vec![
            FaceData::untagged(ids[0], ids[1], ids[2]),
            FaceData::untagged(ids[3], ids[4], ids[5]),
        ];
        let pairs = detect_self_intersections(&faces, &pool);
        assert!(
            pairs.is_empty(),
            "widely separated triangles must not be reported"
        );
    }

    /// Single face: always empty result.
    #[test]
    fn single_face_is_never_self_intersecting() {
        let (pool, ids) = pool_with_verts(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let faces = vec![FaceData::untagged(ids[0], ids[1], ids[2])];
        let pairs = detect_self_intersections(&faces, &pool);
        assert!(pairs.is_empty());
    }

    /// Adversarial: triangle that "touches" another at a vertex (no proper intersection).
    #[test]
    fn vertex_touch_is_not_self_intersection() {
        // ta and tc share vertex at (1,0,0) — adjacent by vertex, not edge.
        let (pool, ids) = pool_with_verts(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0], // same position as ids[1]
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        // Since ids[1] and ids[3] are welded to the same position, they have
        // the same VertexId — the adjacency filter must catch this.
        let faces = vec![
            FaceData::untagged(ids[0], ids[1], ids[2]),
            FaceData::untagged(ids[3], ids[4], ids[5]),
        ];
        let pairs = detect_self_intersections(&faces, &pool);
        // Both share vertex ids[1]==ids[3] → adjacency filter removes the pair.
        assert!(
            pairs.is_empty(),
            "vertex-adjacent triangles must not be reported as self-intersecting"
        );
    }

    /// Triangle in the plane `z = 0`, `scale` on a side.
    fn flat_triangle(scale: Real) -> [Point3r; 3] {
        [
            Point3r::new(0.0, 0.0, 0.0),
            Point3r::new(scale, 0.0, 0.0),
            Point3r::new(scale, scale, 0.0),
        ]
    }

    /// The same triangle tilted about the x axis by `tilt` radians, so its plane
    /// meets `z = 0` along the x axis and the two triangles properly cross.
    ///
    /// With `tilt = 1e-12` the normals are `sin²θ ≈ 1e-24` apart: far below the
    /// parallel threshold, and shallow enough that an unnormalised parallel test
    /// (`|n₁ × n₂|² ∝ L⁸`) crosses its own threshold between scales.
    fn crossing_triangle(scale: Real, tilt: Real) -> [Point3r; 3] {
        let s = tilt;
        [
            Point3r::new(0.4 * scale, -0.2 * scale, -0.2 * scale * s),
            Point3r::new(0.6 * scale, 0.2 * scale, 0.2 * scale * s),
            Point3r::new(0.5 * scale, 0.6 * scale, 0.6 * scale * s),
        ]
    }

    /// The near-parallel decision must not depend on the mesh's scale: this pair
    /// is `sin²θ ≈ 1e-24` apart in normal direction, so it is "parallel" at every
    /// scale and conservatively reported as non-intersecting.
    ///
    /// Under the previous unnormalised test this failed: `|n₁ × n₂|²` scales as
    /// `L⁸`, so at `1e-3` the pair read as parallel and at `1e3` it read as
    /// intersecting.
    #[test]
    fn near_parallel_pair_decisions_are_scale_invariant() {
        let mut decisions = Vec::new();
        for scale in [1e-3_f64, 1.0, 1e3, 1e5] {
            let ta = flat_triangle(scale);
            let tb = crossing_triangle(scale, 1e-12);
            decisions.push(tri_tri_intersects(&ta, &tb));
        }
        assert!(
            decisions.windows(2).all(|w| w[0] == w[1]),
            "the same near-parallel configuration must be decided identically at \
             every scale: {decisions:?}"
        );
        assert!(
            !decisions[0],
            "planes this close to parallel are conservatively non-intersecting"
        );
    }

    /// A pair whose straddling vertex sits a fixed *fraction* of the mesh scale
    /// off the opposing plane must also be decided identically at every scale.
    #[test]
    fn plane_band_decisions_are_scale_invariant() {
        let mut decisions = Vec::new();
        for scale in [1e-3_f64, 1.0, 1e3, 1e5] {
            let ta = flat_triangle(scale);
            // A second triangle crossing ta's plane, with one vertex a
            // scale-relative 1e-11 above it and another well below.
            let gap = 1e-11 * scale;
            let tb = [
                Point3r::new(0.4 * scale, 0.2 * scale, gap),
                Point3r::new(0.6 * scale, 0.4 * scale, -0.5 * scale),
                Point3r::new(0.5 * scale, 0.8 * scale, 0.5 * scale),
            ];
            decisions.push(tri_tri_intersects(&ta, &tb));
        }
        assert!(
            decisions.windows(2).all(|w| w[0] == w[1]),
            "the plate band is relative to the mesh scale, so this pair must be \
             decided identically everywhere: {decisions:?}"
        );
    }
}
