//! Coplanar Boolean Operations Kernel
//!
//! ## Algorithmic Invariants
//!
//! ### `process_triangle` Complexity
//!
//! For **Intersection** (`want_inside = true`), complexity is O(n) in the number
//! of candidate opposing triangles — each emits at most one clipped polygon piece.
//!
//! For **Difference** (`want_inside = false`), the naive approach accumulates a
//! `remaining` fragment list that can grow as each opposing triangle is subtracted.
//! The fix uses per-fragment AABB pre-screening:
//!
//! ```text
//! For each candidate ci:
//!   aabb_ci = candidate triangle AABB
//!   For each remaining fragment frag:
//!     if aabb_of(frag) ∩ aabb_ci = ∅ → skip (O(1) guard)
//!     else → boolean_clip(Difference)
//! ```
//!
//! For a circular cross-section with N cap triangles, the number of triangles
//! that overlap any given fragment AABB is O(1) (adjacent sectors only), so the
//! total work is O(N) rather than O(N·|remaining|).
//!
//! Packed triangle-coordinate and AABB buffers use `leto::Array` as Gaia's
//! Atlas-owned contiguous numeric storage boundary. The exact clipping kernels
//! borrow slices from those arrays, so this changes storage ownership without
//! copying during the hot query loops.
//!

use super::basis::PlaneBasis;
use super::geometry2d::{
    aabb2, aabb_overlaps, point_in_tri_2d_exact, point_in_union_2d_exact_indexed,
};
use crate::application::csg::boolean::BooleanOp;
use crate::application::csg::clip::{
    boolean_clip, clip_polygon_to_triangle, fan_triangulate, ClipOp,
};
use crate::domain::core::scalar::{Point3r, Real};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use leto::Storage;

fn emit_one(
    p0: Point3r,
    p1: Point3r,
    p2: Point3r,
    basis: &PlaneBasis,
    region: crate::domain::core::index::RegionId,
    result: &mut Vec<FaceData>,
    pool: &mut VertexPool,
) {
    let ab = p1 - p0;
    let ac = p2 - p0;
    let fn_ = ab.cross(ac);
    if fn_.norm() < 1e-20 {
        return;
    }
    let flip = fn_.dot(basis.normal) < 0.0;
    let (o0, o1, o2) = if flip { (p0, p2, p1) } else { (p0, p1, p2) };
    let v0 = pool.insert_or_weld(o0, basis.normal);
    let v1 = pool.insert_or_weld(o1, basis.normal);
    let v2 = pool.insert_or_weld(o2, basis.normal);
    if v0 != v1 && v1 != v2 && v0 != v2 {
        result.push(FaceData::new(v0, v1, v2, region));
    }
}

fn emit_poly2d(
    poly2d: &[[Real; 2]],
    basis: &PlaneBasis,
    region: crate::domain::core::index::RegionId,
    result: &mut Vec<FaceData>,
    pool: &mut VertexPool,
) {
    let poly3d: Vec<Point3r> = poly2d.iter().map(|&[u, v]| basis.lift(u, v)).collect();
    for [t0, t1, t2] in fan_triangulate(&poly3d) {
        emit_one(t0, t1, t2, basis, region, result, pool);
    }
}

/// Compute the 2-D AABB `[min_u, min_v, max_u, max_v]` of an arbitrary polygon.
///
/// Returns `None` for degenerate (empty/point/degenerate) input.
#[inline]
fn aabb2_of_poly(poly: &[[Real; 2]]) -> Option<[Real; 4]> {
    if poly.len() < 3 {
        return None;
    }
    let mut min_u = Real::MAX;
    let mut min_v = Real::MAX;
    let mut max_u = Real::MIN;
    let mut max_v = Real::MIN;
    for &[u, v] in poly {
        if u < min_u {
            min_u = u;
        }
        if u > max_u {
            max_u = u;
        }
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    if max_u - min_u < 1e-20 && max_v - min_v < 1e-20 {
        return None; // degenerate point
    }
    Some([min_u, min_v, max_u, max_v])
}

// ── Broad phase index ────────────────────────────────────────────────────────

/// 2-D sweep-and-prune broad-phase index over triangle AABBs.
/// Completeness follows from filtering all entries with `min_u <= src.max_u`
/// using exact `aabb_overlaps` checks.
struct SweepAabbIndex2d {
    by_min_u: Vec<usize>,
}

impl SweepAabbIndex2d {
    fn build(aabbs: &[[Real; 4]]) -> Self {
        let mut by_min_u: Vec<usize> = (0..aabbs.len()).collect();
        by_min_u.sort_unstable_by(|&i, &j| aabbs[i][0].total_cmp(&aabbs[j][0]));
        Self { by_min_u }
    }

    /// Query overlapping opposing AABBs into `out` (allocation-free).
    fn query_overlaps(&self, src_aabb: &[Real; 4], aabbs: &[[Real; 4]], out: &mut Vec<usize>) {
        out.clear();
        let max_u = src_aabb[2];
        let limit = self.by_min_u.partition_point(|&idx| aabbs[idx][0] <= max_u);
        for &idx in &self.by_min_u[..limit] {
            let aabb = &aabbs[idx];
            if aabb[2] < src_aabb[0] {
                continue;
            }
            if aabb_overlaps(src_aabb, aabb) {
                out.push(idx);
            }
        }
    }
}

// ── Pre-computed triangle data ─────────────────────────────────────────────────

struct TriData {
    coords2d: [Real; 6],   // [ax,ay, bx,by, cx,cy] for point-in-union and clipping
    aabb2d: [Real; 4],     // [min_u, min_v, max_u, max_v]
    verts3d: [Point3r; 3], // 3-D positions (needed to emit original triangles)
}

struct CoplanarBuffers {
    tris: leto::Array<[Real; 6], leto::VecStorage<[Real; 6]>, 1>,
    aabbs: leto::Array<[Real; 4], leto::VecStorage<[Real; 4]>, 1>,
}

impl CoplanarBuffers {
    fn from_tri_data(data: &[TriData]) -> Self {
        let tris = data.iter().map(|tri| tri.coords2d).collect();
        let aabbs = data.iter().map(|tri| tri.aabb2d).collect();
        Self {
            tris: leto::Array::from_shape_vec([data.len()], tris)
                .expect("invariant: collected one coordinate row per triangle"),
            aabbs: leto::Array::from_shape_vec([data.len()], aabbs)
                .expect("invariant: collected one AABB row per triangle"),
        }
    }

    #[inline]
    fn tris(&self) -> &[[Real; 6]] {
        self.tris.storage().as_slice()
    }

    #[inline]
    fn aabbs(&self) -> &[[Real; 4]] {
        self.aabbs.storage().as_slice()
    }
}

fn build_tri_data(faces: &[FaceData], pool: &VertexPool, basis: &PlaneBasis) -> Vec<TriData> {
    faces
        .iter()
        .map(|f| {
            let p = *pool.position(f.vertices[0]);
            let q = *pool.position(f.vertices[1]);
            let r = *pool.position(f.vertices[2]);
            let [px, py] = basis.project(&p);
            let [qx, qy] = basis.project(&q);
            let [rx, ry] = basis.project(&r);
            TriData {
                coords2d: [px, py, qx, qy, rx, ry],
                aabb2d: aabb2(px, py, qx, qy, rx, ry),
                verts3d: [p, q, r],
            }
        })
        .collect()
}

// ── Core: process one source triangle against opposing triangles ───────────────

/// Process one source triangle (in 2-D) against opposing triangles.
///
/// `want_inside = true`  → emit src ∩ (∪ opp)   (Intersection)
/// `want_inside = false` → emit src \ (∪ opp)   (Difference / Union B\A)
fn process_triangle(
    src: &[Real; 6],       // [ax,ay,bx,by,cx,cy] of source in 2-D
    src_3d: &[Point3r; 3], // 3-D positions for fast-path emit
    src_aabb: &[Real; 4],
    aabb_index: &SweepAabbIndex2d,
    opp: &[TriData],
    opp_tris: &[[Real; 6]], // 2-D coords of ALL opposing triangles
    opp_aabbs: &[[Real; 4]],
    want_inside: bool,
    basis: &PlaneBasis,
    region: crate::domain::core::index::RegionId,
    result: &mut Vec<FaceData>,
    pool: &mut VertexPool,
    candidates: &mut Vec<usize>,
) {
    aabb_index.query_overlaps(src_aabb, opp_aabbs, candidates);

    if candidates.is_empty() {
        if !want_inside {
            emit_one(src_3d[0], src_3d[1], src_3d[2], basis, region, result, pool);
        }
        return;
    }

    let [ax, ay, bx, by, cx, cy] = *src;

    let va = point_in_union_2d_exact_indexed(ax, ay, opp_tris, candidates);
    let vb = point_in_union_2d_exact_indexed(bx, by, opp_tris, candidates);
    let vc = point_in_union_2d_exact_indexed(cx, cy, opp_tris, candidates);

    let all_in = va && vb && vc;
    let all_out_vertices = !va && !vb && !vc;
    let all_out = all_out_vertices
        && candidates.iter().all(|&i| {
            let [dx, dy, ex, ey, fx, fy] = opp_tris[i];
            !point_in_tri_2d_exact(dx, dy, ax, ay, bx, by, cx, cy)
                && !point_in_tri_2d_exact(ex, ey, ax, ay, bx, by, cx, cy)
                && !point_in_tri_2d_exact(fx, fy, ax, ay, bx, by, cx, cy)
        });

    if all_in {
        if want_inside {
            emit_one(src_3d[0], src_3d[1], src_3d[2], basis, region, result, pool);
        }
        return;
    }
    if all_out && want_inside {
        return;
    }

    let src_poly: Vec<[Real; 2]> = vec![[ax, ay], [bx, by], [cx, cy]];

    if want_inside {
        for &ci in candidates.iter() {
            let [dx, dy, ex, ey, fx, fy] = opp[ci].coords2d;
            let inside = clip_polygon_to_triangle(&src_poly, dx, dy, ex, ey, fx, fy);
            if inside.len() >= 3 {
                emit_poly2d(&inside, basis, region, result, pool);
            }
        }
    } else {
        // ── Difference path: src \ (∪ candidates) ────────────────────────────
        // Maintain a list of remaining polygon fragments representing the
        // portion of `src` not yet subtracted by any candidate.
        //
        // AABB pre-screening: skip `boolean_clip` when fragment/candidate
        // boxes do not overlap.
        // For a circular cross-section, each fragment only overlaps O(1)
        // adjacent sector triangles, reducing the total work from O(N·|rem|)
        // to O(N) for N candidate triangles.
        //
        // **Invariant**: at all times, `remaining` exactly partitions the
        // portion of `src` lying outside all subtracted candidates so far.
        let mut remaining: Vec<Vec<[Real; 2]>> = vec![src_poly];

        for &ci in candidates.iter() {
            let [dx, dy, ex, ey, fx, fy] = opp[ci].coords2d;
            let cand_aabb = opp_aabbs[ci];
            let b_poly = [[dx, dy], [ex, ey], [fx, fy]];

            let mut new_rem: Vec<Vec<[Real; 2]>> = Vec::with_capacity(remaining.len());

            for poly in remaining {
                // AABB guard: if this fragment cannot possibly overlap the
                // candidate triangle, skip the CDT call entirely.
                match aabb2_of_poly(&poly) {
                    None => {} // degenerate fragment — discard
                    Some(frag_aabb) if !aabb_overlaps(&frag_aabb, &cand_aabb) => {
                        new_rem.push(poly); // no overlap — fragment unchanged
                    }
                    Some(_) => {
                        // Actual clip: fragment minus candidate triangle.
                        let pieces = boolean_clip(&poly, &b_poly, ClipOp::Difference);
                        for piece in pieces {
                            if piece.len() >= 3 {
                                new_rem.push(piece);
                            }
                        }
                    }
                }
            }
            remaining = new_rem;

            // Early exit: nothing left to subtract from.
            if remaining.is_empty() {
                return;
            }
        }

        for poly in &remaining {
            emit_poly2d(poly, basis, region, result, pool);
        }
    }
}

pub(crate) fn boolean_coplanar(
    op: BooleanOp,
    faces_a: &[FaceData],
    faces_b: &[FaceData],
    pool: &mut VertexPool,
    basis: &PlaneBasis,
) -> Vec<FaceData> {
    let mut result: Vec<FaceData> = Vec::new();

    let b_data = build_tri_data(faces_b, pool, basis);
    let a_data = build_tri_data(faces_a, pool, basis);

    let b_buffers = CoplanarBuffers::from_tri_data(&b_data);
    let a_buffers = CoplanarBuffers::from_tri_data(&a_data);
    let b_tris = b_buffers.tris();
    let a_tris = a_buffers.tris();
    let b_aabbs = b_buffers.aabbs();
    let a_aabbs = a_buffers.aabbs();
    let b_index = SweepAabbIndex2d::build(b_aabbs);
    let a_index = SweepAabbIndex2d::build(a_aabbs);
    let mut candidate_buf_ab: Vec<usize> = Vec::new();
    let mut candidate_buf_ba: Vec<usize> = Vec::new();

    for (ai, fa) in faces_a.iter().enumerate() {
        let src = &a_tris[ai];
        let src_3d = &a_data[ai].verts3d;
        let aabb = &a_aabbs[ai];

        match op {
            BooleanOp::Union => {
                process_triangle(
                    src,
                    src_3d,
                    aabb,
                    &b_index,
                    &b_data,
                    b_tris,
                    b_aabbs,
                    false,
                    basis,
                    fa.region,
                    &mut result,
                    pool,
                    &mut candidate_buf_ab,
                );
                process_triangle(
                    src,
                    src_3d,
                    aabb,
                    &b_index,
                    &b_data,
                    b_tris,
                    b_aabbs,
                    true,
                    basis,
                    fa.region,
                    &mut result,
                    pool,
                    &mut candidate_buf_ab,
                );
            }
            BooleanOp::Intersection => {
                process_triangle(
                    src,
                    src_3d,
                    aabb,
                    &b_index,
                    &b_data,
                    b_tris,
                    b_aabbs,
                    true,
                    basis,
                    fa.region,
                    &mut result,
                    pool,
                    &mut candidate_buf_ab,
                );
            }
            BooleanOp::Difference => {
                process_triangle(
                    src,
                    src_3d,
                    aabb,
                    &b_index,
                    &b_data,
                    b_tris,
                    b_aabbs,
                    false,
                    basis,
                    fa.region,
                    &mut result,
                    pool,
                    &mut candidate_buf_ab,
                );
            }
        }
    }

    if matches!(op, BooleanOp::Union) {
        for (bi, fb) in faces_b.iter().enumerate() {
            process_triangle(
                &b_tris[bi],
                &b_data[bi].verts3d,
                &b_aabbs[bi],
                &a_index,
                &a_data,
                a_tris,
                a_aabbs,
                false,
                basis,
                fb.region,
                &mut result,
                pool,
                &mut candidate_buf_ba,
            );
        }
    }

    result
}

#[cfg(test)]
#[path = "tests_operations.rs"]
mod tests;
