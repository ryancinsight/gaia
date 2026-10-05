//! Shared planar arrangement helpers for CDT-backed CSG subroutines.
//!
//! These utilities consolidate the common "shatter constraints into PSLG
//! sub-edges" workflow used by:
//! - `clip::polygon2d::cdt` (native 2-D polygon Boolean)
//! - `corefine` (3-D face projection -> planar CDT subdivision)

use eunomia::{FloatElement, NumericElement};
use hashbrown::HashMap;

use crate::application::delaunay::{Pslg, PslgVertexId};
use crate::application::welding::GridCell2d;
use crate::domain::core::scalar::{Real, Scalar};

/// Canonical undirected edge key between two point slots.
pub(crate) type PlanarEdgeKey = (usize, usize);

/// Uniform-grid index over planar points for local AABB candidate queries.
/// Accelerates "points near segment" searches from O(E·P) to near-linear.
pub(crate) struct PlanarPointGridIndex {
    inv_cell: Real,
    bins: HashMap<GridCell2d, Vec<usize>>,
}

impl PlanarPointGridIndex {
    #[inline]
    fn point_to_real<T: Scalar>(p: [T; 2]) -> [Real; 2] {
        [p[0].to_f64(), p[1].to_f64()]
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the corridor radius is ceil()'d to an integer number of cells, non-negative, and bounded by practical mesh geometry"
    )]
    fn corridor_radius(tol: Real, cell_size: Real) -> i64 {
        ((tol.max(0.0) / cell_size).ceil() as i64).max(0)
    }

    #[must_use]
    pub(crate) fn new<T: Scalar>(points: &[[T; 2]], cell: T) -> Self {
        let safe_cell = cell.to_f64().max(1e-12);
        let inv_cell = 1.0 / safe_cell;
        let mut bins: HashMap<GridCell2d, Vec<usize>> = HashMap::with_capacity(points.len());
        for (slot, p) in points.iter().enumerate() {
            let point = Self::point_to_real(*p);
            bins.entry(GridCell2d::from_point(&point, inv_cell))
                .or_default()
                .push(slot);
        }
        Self { inv_cell, bins }
    }

    fn cell_of<T: Scalar>(&self, p: [T; 2]) -> GridCell2d {
        let point = Self::point_to_real(p);
        GridCell2d::from_point(&point, self.inv_cell)
    }

    pub(crate) fn collect_aabb_candidates(
        &self,
        min: [Real; 2],
        max: [Real; 2],
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let c_min = self.cell_of(min);
        let c_max = self.cell_of(max);
        let (ix0, iy0) = (c_min.x, c_min.y);
        let (ix1, iy1) = (c_max.x, c_max.y);

        let span_x = u128::try_from((ix1 - ix0 + 1).max(0_i64)).expect("non-negative after max(0)");
        let span_y = u128::try_from((iy1 - iy0 + 1).max(0_i64)).expect("non-negative after max(0)");
        let dense_cell_visits = span_x.saturating_mul(span_y);
        let occupied_bin_count =
            u128::try_from(self.bins.len()).expect("usize always fits in u128");

        if dense_cell_visits > occupied_bin_count.saturating_mul(8) {
            for (&cell, slots) in &self.bins {
                if cell.x >= ix0 && cell.x <= ix1 && cell.y >= iy0 && cell.y <= iy1 {
                    out.extend_from_slice(slots);
                }
            }
            return;
        }

        for ix in ix0..=ix1 {
            for iy in iy0..=iy1 {
                if let Some(slots) = self.bins.get(&GridCell2d { x: ix, y: iy }) {
                    out.extend_from_slice(slots);
                }
            }
        }
    }

    #[inline]
    fn append_cell_slots(&self, ix: i64, iy: i64, out: &mut Vec<usize>) {
        if let Some(slots) = self.bins.get(&GridCell2d { x: ix, y: iy }) {
            out.extend_from_slice(slots);
        }
    }

    fn append_neighborhood_slots(&self, cx: i64, cy: i64, radius: i64, out: &mut Vec<usize>) {
        for ix in (cx - radius)..=(cx + radius) {
            for iy in (cy - radius)..=(cy + radius) {
                self.append_cell_slots(ix, iy, out);
            }
        }
    }

    fn traverse_segment_cells<T: Scalar>(
        &self,
        p1: [T; 2],
        p2: [T; 2],
        mut visit: impl FnMut(i64, i64),
    ) {
        let p1 = Self::point_to_real(p1);
        let p2 = Self::point_to_real(p2);
        let c1 = self.cell_of(p1);
        let c2 = self.cell_of(p2);
        let (mut cx, mut cy) = (c1.x, c1.y);
        let (tx, ty) = (c2.x, c2.y);

        visit(cx, cy);
        if cx == tx && cy == ty {
            return;
        }

        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let cell_size = 1.0 / self.inv_cell;

        let step_x = if dx > 0.0 {
            1
        } else if dx < 0.0 {
            -1
        } else {
            0
        };
        let step_y = if dy > 0.0 {
            1
        } else if dy < 0.0 {
            -1
        } else {
            0
        };

        let mut t_max_x = Real::INFINITY;
        let mut t_max_y = Real::INFINITY;
        let mut t_delta_x = Real::INFINITY;
        let mut t_delta_y = Real::INFINITY;

        if step_x != 0 {
            let next_boundary_x = if step_x > 0 {
                Real::from_integer(cx + 1) * cell_size
            } else {
                Real::from_integer(cx) * cell_size
            };
            t_max_x = ((next_boundary_x - p1[0]) / dx).max(0.0);
            t_delta_x = cell_size / dx.abs();
        }

        if step_y != 0 {
            let next_boundary_y = if step_y > 0 {
                Real::from_integer(cy + 1) * cell_size
            } else {
                Real::from_integer(cy) * cell_size
            };
            t_max_y = ((next_boundary_y - p1[1]) / dy).max(0.0);
            t_delta_y = cell_size / dy.abs();
        }

        while cx != tx || cy != ty {
            if t_max_x < t_max_y {
                cx += step_x;
                t_max_x += t_delta_x;
            } else if t_max_y < t_max_x {
                cy += step_y;
                t_max_y += t_delta_y;
            } else {
                if step_x != 0 {
                    cx += step_x;
                    t_max_x += t_delta_x;
                }
                if step_y != 0 {
                    cy += step_y;
                    t_max_y += t_delta_y;
                }
            }
            visit(cx, cy);
        }
    }

    /// Collect point slots from grid cells intersecting a segment corridor.
    ///
    /// Uses an adaptive strategy:
    /// - fast path: DDA traversal over cells intersected by the segment with
    ///   Chebyshev-neighborhood expansion radius `r = ceil(tol / cell_size)`,
    /// - fallback: sparse occupied-bin AABB filtering when DDA step count is
    ///   disproportionate to occupied bins.
    ///
    /// # Theorem — Adaptive corridor completeness
    ///
    /// In DDA mode, the segment-corridor completeness theorem holds directly:
    /// every point within tolerance lies in an `r`-expanded traversed cell.
    /// In sparse-fallback mode, AABB-filter completeness holds: every accepted
    /// point lies in the segment AABB expanded by `tol`. Since dispatch chooses
    /// one of these complete supersets, candidate retrieval is complete. ∎
    pub(crate) fn collect_segment_corridor_candidates<T: Scalar>(
        &self,
        p1: [T; 2],
        p2: [T; 2],
        tol: T,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let p1_f = Self::point_to_real(p1);
        let p2_f = Self::point_to_real(p2);
        let tol = tol.to_f64();
        let cell_size = 1.0 / self.inv_cell;
        let radius = Self::corridor_radius(tol, cell_size);
        let c_start = self.cell_of(p1_f);
        let c_end = self.cell_of(p2_f);
        let (sx, sy) = (c_start.x, c_start.y);
        let (tx, ty) = (c_end.x, c_end.y);

        let steps = u128::from(sx.abs_diff(tx).max(sy.abs_diff(ty))).saturating_add(1);
        let radius_u = u128::try_from(radius).unwrap_or(0);
        let diameter = radius_u.saturating_mul(2).saturating_add(1);
        let neighborhood = diameter.saturating_mul(diameter);
        let dda_visits = steps.saturating_mul(neighborhood);
        let occupied_bin_count =
            u128::try_from(self.bins.len()).expect("usize always fits in u128");

        // Guard against pathological tiny-cell / long-segment traversals.
        // In those cases, scanning occupied bins in a segment AABB corridor
        // is asymptotically better than stepping every crossed grid cell.
        if dda_visits > occupied_bin_count.saturating_mul(8) {
            let min = [p1_f[0].min(p2_f[0]) - tol, p1_f[1].min(p2_f[1]) - tol];
            let max = [p1_f[0].max(p2_f[0]) + tol, p1_f[1].max(p2_f[1]) + tol];
            self.collect_aabb_candidates(min, max, out);
            return;
        }

        self.traverse_segment_cells(p1_f, p2_f, |cx, cy| {
            self.append_neighborhood_slots(cx, cy, radius, out);
        });
        out.sort_unstable();
        out.dedup();
    }
}

/// Collect candidate `(t, slot)` pairs that lie on a segment interior (plus
/// explicit endpoints) using raw line projection.
///
/// This variant is used by the native 2-D polygon clipping path where polygon
/// loop endpoints are known explicitly and interior intersection points are
/// added from the merged point set.
pub(crate) fn collect_points_on_segment_interior_to_buf<T: Scalar>(
    unique_pts: &[[T; 2]],
    p1: [T; 2],
    p2: [T; 2],
    endpoint_slots: (usize, usize),
    t_eps: T,
    dist_sq_tol: T,
    out: &mut Vec<(T, usize)>,
) {
    let (ri, rj) = endpoint_slots;
    let p1_f = PlanarPointGridIndex::point_to_real(p1);
    let p2_f = PlanarPointGridIndex::point_to_real(p2);
    let dx = p2_f[0] - p1_f[0];
    let dy = p2_f[1] - p1_f[1];
    let l2 = dx * dx + dy * dy;
    let t_eps_f = t_eps.to_f64();
    let dist_sq_tol_f = dist_sq_tol.to_f64();

    out.clear();
    out.push((<T as NumericElement>::ZERO, ri));
    out.push((<T as NumericElement>::ONE, rj));
    if l2 < 1e-24 {
        return;
    }

    for (slot, upt) in unique_pts.iter().enumerate() {
        if slot == ri || slot == rj {
            continue;
        }

        let upt_f = PlanarPointGridIndex::point_to_real(*upt);
        let t = ((upt_f[0] - p1_f[0]) * dx + (upt_f[1] - p1_f[1]) * dy) / l2;
        if t < t_eps_f || t > 1.0 - t_eps_f {
            continue;
        }

        let fx = p1_f[0] + t * dx;
        let fy = p1_f[1] + t * dy;
        let ex = upt_f[0] - fx;
        let ey = upt_f[1] - fy;
        let d2 = ex * ex + ey * ey;
        if d2 < dist_sq_tol_f {
            out.push((<T as Scalar>::from_f64(t), slot));
        }
    }
}

pub(crate) fn collect_points_on_segment_interior<T: Scalar>(
    unique_pts: &[[T; 2]],
    p1: [T; 2],
    p2: [T; 2],
    endpoint_slots: (usize, usize),
    t_eps: T,
    dist_sq_tol: T,
) -> Vec<(T, usize)> {
    // PERF-NOTE: this function is called once per constraint edge; the Vec allocation
    // is proportional to the segment's local point density and not a bottleneck.
    let mut out = Vec::new();
    collect_points_on_segment_interior_to_buf(
        unique_pts,
        p1,
        p2,
        endpoint_slots,
        t_eps,
        dist_sq_tol,
        &mut out,
    );
    out
}

/// Indexed variant of [`collect_points_on_segment_interior`].
///
/// # Theorem — DDA-corridor completeness
///
/// Candidate retrieval via
/// [`PlanarPointGridIndex::collect_segment_corridor_candidates`] is complete
/// for all points within the distance tolerance corridor of the segment. Since
/// final acceptance still uses exact projection and distance checks, this index
/// stage can only change performance, not geometric correctness. ∎
#[expect(
    clippy::too_many_arguments,
    reason = "segment-point query requires both endpoints, tolerances, a spatial index, a scratch buffer, and an output buffer — splitting would force extra allocations"
)]
pub(crate) fn collect_points_on_segment_interior_indexed<T: Scalar>(
    unique_pts: &[[T; 2]],
    point_index: &PlanarPointGridIndex,
    p1: [T; 2],
    p2: [T; 2],
    endpoint_slots: (usize, usize),
    t_eps: T,
    dist_sq_tol: T,
    candidate_slots: &mut Vec<usize>,
    out: &mut Vec<(T, usize)>,
) {
    let (ri, rj) = endpoint_slots;
    let p1_f = PlanarPointGridIndex::point_to_real(p1);
    let p2_f = PlanarPointGridIndex::point_to_real(p2);
    let dx = p2_f[0] - p1_f[0];
    let dy = p2_f[1] - p1_f[1];
    let l2 = dx * dx + dy * dy;
    let t_eps_f = t_eps.to_f64();
    let dist_sq_tol_f = dist_sq_tol.to_f64();

    out.clear();
    out.push((<T as NumericElement>::ZERO, ri));
    out.push((<T as NumericElement>::ONE, rj));
    if l2 < 1e-24 {
        return;
    }

    let tol = dist_sq_tol.sqrt();
    point_index.collect_segment_corridor_candidates(p1, p2, tol, candidate_slots);

    for &slot in candidate_slots.iter() {
        if slot == ri || slot == rj {
            continue;
        }
        let upt = unique_pts[slot];
        let upt_f = PlanarPointGridIndex::point_to_real(upt);
        let t = ((upt_f[0] - p1_f[0]) * dx + (upt_f[1] - p1_f[1]) * dy) / l2;
        if t < t_eps_f || t > 1.0 - t_eps_f {
            continue;
        }

        let fx = p1_f[0] + t * dx;
        let fy = p1_f[1] + t * dy;
        let ex = upt_f[0] - fx;
        let ey = upt_f[1] - fy;
        let d2 = ex * ex + ey * ey;
        if d2 < dist_sq_tol_f {
            out.push((<T as Scalar>::from_f64(t), slot));
        }
    }
}

/// Sort a `(t, slot)` point list, deduplicate slots, and emit shattered
/// consecutive sub-edges into `edges`.
pub(crate) fn insert_shattered_subedges<T: Scalar>(
    on_seg: &mut Vec<(T, usize)>,
    edges: &mut Vec<PlanarEdgeKey>,
) {
    on_seg.sort_by(|a, b| a.0.total_cmp(&b.0));
    on_seg.dedup_by_key(|x| x.1);

    for w in on_seg.windows(2) {
        let (a, b) = (w[0].1, w[1].1);
        if a == b {
            continue;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        edges.push(key);
    }
}

/// Build a PSLG from planar points and deduplicated undirected edge keys.
pub(crate) fn build_pslg_from_points_and_edges<T: Scalar>(
    points: &[[T; 2]],
    edges: &[PlanarEdgeKey],
) -> Pslg {
    let mut pslg = Pslg::with_capacity(points.len(), edges.len());
    for &p in points {
        pslg.add_vertex(p[0].to_f64(), p[1].to_f64());
    }
    for &(a, b) in edges {
        if a == b {
            continue;
        }
        let id_a = PslgVertexId::from_usize(a);
        let id_b = PslgVertexId::from_usize(b);
        pslg.add_segment(id_a, id_b);
    }
    pslg
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::HashSet;
    use proptest::prelude::*;

    #[test]
    fn indexed_segment_collection_finds_all_collinear_points() {
        let pts = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [2.5, 0.0],
            [5.0, 0.0],
            [7.5, 0.0],
            [5.0, 1e-7],
        ];
        let index = PlanarPointGridIndex::new(&pts, 0.1);
        let mut candidates = Vec::new();
        let mut got = Vec::new();
        collect_points_on_segment_interior_indexed(
            &pts,
            &index,
            pts[0],
            pts[1],
            (0, 1),
            1e-8,
            1e-12,
            &mut candidates,
            &mut got,
        );
        let slots: HashSet<usize> = got.into_iter().map(|(_, slot)| slot).collect();
        assert!(slots.contains(&0));
        assert!(slots.contains(&1));
        assert!(slots.contains(&2));
        assert!(slots.contains(&3));
        assert!(slots.contains(&4));
    }

    proptest! {
        #[test]
        fn segment_corridor_candidates_cover_exact_accepts(
            pts in prop::collection::vec((-120_i16..120_i16, -120_i16..120_i16), 6..100),
        ) {
            let unique_pts: Vec<[Real; 2]> = pts
                .into_iter()
                .map(|(x, y)| [Real::from(x) * 0.05, Real::from(y) * 0.05])
                .collect();

            let p1 = unique_pts[0];
            let p2 = unique_pts[1];
            let endpoint_slots = (0usize, 1usize);
            let t_eps = 1e-8;
            let dist_sq_tol = 1e-10;

            let exact = collect_points_on_segment_interior(
                &unique_pts,
                p1,
                p2,
                endpoint_slots,
                t_eps,
                dist_sq_tol,
            );
            let exact_slots: HashSet<usize> = exact.into_iter().map(|(_, slot)| slot).collect();

            let index = PlanarPointGridIndex::new(&unique_pts, dist_sq_tol.sqrt());
            let mut candidates = Vec::new();
            index.collect_segment_corridor_candidates(p1, p2, dist_sq_tol.sqrt(), &mut candidates);
            let candidate_slots: HashSet<usize> = candidates.into_iter().collect();

            for slot in exact_slots {
                prop_assert!(
                    candidate_slots.contains(&slot),
                    "candidate retrieval missed exact-accepted slot {slot}"
                );
            }
        }

        #[test]
        fn indexed_segment_collection_matches_bruteforce(
            pts in prop::collection::vec((-100_i16..100_i16, -100_i16..100_i16), 6..80),
        ) {
            let unique_pts: Vec<[Real; 2]> = pts
                .into_iter()
                .map(|(x, y)| [Real::from(x) * 0.05, Real::from(y) * 0.05])
                .collect();

            let p1 = unique_pts[0];
            let p2 = unique_pts[1];
            let endpoint_slots = (0usize, 1usize);
            let t_eps = 1e-8;
            let dist_sq_tol = 1e-10;

            let brute = collect_points_on_segment_interior(
                &unique_pts,
                p1,
                p2,
                endpoint_slots,
                t_eps,
                dist_sq_tol,
            );

            let index = PlanarPointGridIndex::new(&unique_pts, dist_sq_tol.sqrt());
            let mut candidates = Vec::new();
            let mut fast = Vec::new();
            collect_points_on_segment_interior_indexed(
                &unique_pts,
                &index,
                p1,
                p2,
                endpoint_slots,
                t_eps,
                dist_sq_tol,
                &mut candidates,
                &mut fast,
            );

            let brute_slots: HashSet<usize> = brute.into_iter().map(|(_, slot)| slot).collect();
            let fast_slots: HashSet<usize> = fast.into_iter().map(|(_, slot)| slot).collect();
            prop_assert_eq!(fast_slots, brute_slots);
        }
    }
}
