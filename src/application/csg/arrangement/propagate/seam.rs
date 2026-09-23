//! The general seam-vertex pass: propagate Steiner points across shared edges.

use crate::application::csg::intersect::SnapSegment;
use crate::application::csg::predicates3d::point_on_segment_exact;
use crate::domain::core::scalar::{Point3r, Real};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use hashbrown::HashMap;

use super::adjacency::AdjacentFaces;
use super::tolerances::{
    COINCIDENT_LEN_SQ, COLLINEAR_TOL_SQ, DEGENERATE_LEN_SQ, PARAM_DEDUP_TOL, PARAM_MARGIN,
};

/// Smallest usable 2-D determinant for the axis-pair solve, in **length²**.
///
/// The solve below divides by a 2-D cross product of two in-plane lengths, so
/// the determinant it must not be divided by scales as `length²`. A threshold
/// that scales as `length` instead makes the effective rejection *angle* grow
/// as `1 / length`: it rejects more axis pairs on a small mesh than on a
/// scaled-up copy of the same mesh. That is a scale-dependent decision, in a
/// pass whose correctness argument is scale consistency, and it is exactly the
/// failure class `arrangement::scale_robustness_tests` exists to catch.
///
/// The form mirrors the same rejection in `csg::corefine`
/// (`1e-14 * n_sq.sqrt() * (d1.norm() + d2.norm())` — two lengths multiplied),
/// so both passes agree on what "near-parallel" means at any scale.
#[must_use]
pub(super) fn min_axis_determinant(edge_len_sq: Real, seg_len_sq: Real) -> Real {
    let edge_len = edge_len_sq.sqrt();
    1e-14 * edge_len * (edge_len + seg_len_sq.sqrt())
}

/// Ensure that every seam vertex created by CDT co-refinement is injected into
/// all faces that share the face edge on which the seam vertex lies.
///
/// ## Problem — T-junctions at shared edges
///
/// When `intersect_triangles` produces a snap-segment endpoint P that lies
/// on the boundary edge `[Va, Vb]` of a triangle face `f1`, the CDT of `f1`
/// inserts a Steiner vertex at P and produces sub-edges `Va→P` and `P→Vb`.
/// However, the adjacent face `f2` (which shares the undirected edge `{Va,Vb}`)
/// has no snap segment touching P, so its CDT leaves edge `Va→Vb` unsplit.
///
/// In the final mesh, sub-edge `Va→P` appears once (from `f1`'s CDT) with no
/// counterpart from `f2` → open boundary edge → non-manifold output.
///
/// ## Algorithm
///
/// 1. Build an undirected edge adjacency map: `{Va, Vb} → [face_idx, …]`.
/// 2. For each face `f` that has snap segments, collect all seam-endpoint
///    positions from those segments.
/// 3. For each endpoint P and each edge `[Va, Vb]` of face `f`, check if P
///    lies strictly between Va and Vb (collinearity + parameter check).
/// 4. If yes, inject snap segments `Va→P` and `P→Vb` into every OTHER face
///    that shares edge `{Va, Vb}` — propagating the Steiner vertex across the
///    shared edge.
///
/// ## Collinearity threshold
///
/// A point P is considered to lie on edge [Va, Vb] if:
/// - `|(Vb−Va) × (P−Va)|² / |Vb−Va|² < COLLINEAR_TOL_SQ` (1e-6; sub-millimetre)
/// - parameter `t = (P−Va)·(Vb−Va) / |Vb−Va|² ∈ (PARAM_MARGIN, 1−PARAM_MARGIN)`
pub fn propagate_seam_vertices(
    faces: &[FaceData],
    segs: &mut [Vec<SnapSegment>],
    pool: &VertexPool,
) {
    use crate::domain::core::index::VertexId;

    if segs.is_empty() {
        return;
    }

    type EdgeKey = (VertexId, VertexId);
    let mut edge_to_faces: HashMap<EdgeKey, AdjacentFaces> =
        HashMap::with_capacity(faces.len() * 3 / 2);
    for (fi, face) in faces.iter().enumerate() {
        let v = face.vertices;
        for i in 0..3_usize {
            let va = v[i];
            let vb = v[(i + 1) % 3];
            let key = if va < vb { (va, vb) } else { (vb, va) };
            edge_to_faces.entry(key).or_default().push(fi);
        }
    }

    let mut injections = Vec::new();
    let mut t_params = Vec::new();
    let mut pts = Vec::new();
    propagate_seam_vertices_impl(
        faces,
        segs,
        pool,
        &edge_to_faces,
        &mut injections,
        &mut t_params,
        &mut pts,
    );
}

fn propagate_seam_vertices_impl(
    faces: &[FaceData],
    segs: &mut [Vec<SnapSegment>],
    pool: &VertexPool,
    edge_to_faces: &HashMap<
        (
            crate::domain::core::index::VertexId,
            crate::domain::core::index::VertexId,
        ),
        AdjacentFaces,
    >,
    injections: &mut Vec<(usize, SnapSegment)>,
    t_params: &mut Vec<Real>,
    pts: &mut Vec<Point3r>,
) {
    injections.clear();

    for (fi, snap_segs) in segs.iter().enumerate() {
        if snap_segs.is_empty() || fi >= faces.len() {
            continue;
        }
        let face = &faces[fi];
        let v = face.vertices;

        for i in 0..3_usize {
            let va_id = v[i];
            let vb_id = v[(i + 1) % 3];
            let pa = *pool.position(va_id);
            let pb = *pool.position(vb_id);

            let edge_vec = pb - pa;
            let edge_len_sq = edge_vec.dot(edge_vec);
            if edge_len_sq < DEGENERATE_LEN_SQ {
                continue;
            }

            let edge_key = if va_id < vb_id {
                (va_id, vb_id)
            } else {
                (vb_id, va_id)
            };
            let adj_faces = match edge_to_faces.get(&edge_key) {
                Some(f) => f,
                None => continue,
            };
            if !adj_faces.iter().any(|&f| f != fi) {
                continue;
            }

            t_params.clear();

            for seg in snap_segs {
                for &p in &[seg.start, seg.end] {
                    if let Some(t_exact) = point_on_segment_exact(&pa, &pb, &p) {
                        if t_exact > PARAM_MARGIN && t_exact < 1.0 - PARAM_MARGIN {
                            t_params.push(t_exact);
                        }
                        continue;
                    }

                    let sp: leto::geometry::Vector3<f64> = p - pa;
                    let sp_len_sq = sp.norm_squared();
                    if sp_len_sq < COINCIDENT_LEN_SQ {
                        continue;
                    }
                    let cross_v = edge_vec.cross(sp);
                    if cross_v.norm_squared() <= COLLINEAR_TOL_SQ * edge_len_sq * sp_len_sq {
                        let t = sp.dot(edge_vec) / edge_len_sq;
                        if t > PARAM_MARGIN && t < 1.0 - PARAM_MARGIN {
                            t_params.push(t);
                        }
                    }
                }

                let sv = seg.end - seg.start;
                let r_vec = seg.start - pa;
                let pairs: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
                let mut best_det_abs = 0.0_f64;
                let mut best_t = 0.0_f64;
                let mut best_s = 0.0_f64;
                for &(ax, ay) in &pairs {
                    let e0 = edge_vec[ax];
                    let e1 = edge_vec[ay];
                    let s0 = sv[ax];
                    let s1 = sv[ay];
                    let r0 = r_vec[ax];
                    let r1 = r_vec[ay];
                    let det = e0 * (-s1) - e1 * (-s0);
                    if det.abs() > best_det_abs {
                        best_det_abs = det.abs();
                        best_t = (r0 * (-s1) - r1 * (-s0)) / det;
                        best_s = (e0 * r1 - e1 * r0) / det;
                    }
                }
                // Reject a near-parallel axis pair: the 2x2 solve below divides
                // by this determinant. `best_det_abs` is a 2-D determinant of
                // two in-plane lengths, so the threshold must be a length² or
                // the decision stops being scale-equivariant — see
                // [`min_axis_determinant`].
                let min_det = min_axis_determinant(edge_len_sq, sv.norm_squared());
                if best_det_abs < min_det {
                    continue;
                }
                if best_t <= PARAM_MARGIN || best_t >= 1.0 - PARAM_MARGIN {
                    continue;
                }
                if best_s <= PARAM_MARGIN || best_s >= 1.0 - PARAM_MARGIN {
                    continue;
                }
                let x_edge = pa + edge_vec * best_t;
                let x_seg = seg.start + sv * best_s;
                // The 3-D points the parameter pair reconstructs must coincide
                // to within `1e-6` *relative in squared distance* — a relative
                // length of 1e-3, since both sides scale as length².
                if (x_edge).distance_squared(x_seg) > 1e-6 * edge_len_sq {
                    // The parameter pair failed its own verification, so the
                    // crossing is rejected.
                    //
                    // An exact-predicate fallback used to be attempted here,
                    // with its result discarded into `_` — so it never decided
                    // anything, and it computed a projection axis, four
                    // `orient_2d_arr` calls and a 2x2 solve per rejected
                    // candidate for nothing. It was removed rather than wired
                    // up: instrumenting this branch and running the whole suite
                    // shows it is never reached, and the predicate it called
                    // returns `None` whenever any projected orientation is
                    // degenerate — which is precisely the near-parallel
                    // configuration that gets here. See the open item in
                    // `backlog/atlas-gaia-mesh-renderer.md`.
                    continue;
                }
                t_params.push(best_t);
            }

            if t_params.is_empty() {
                continue;
            }

            t_params.sort_by(|a, b| a.total_cmp(b));
            t_params.dedup_by(|a, b| (*a - *b).abs() < PARAM_DEDUP_TOL);

            pts.clear();
            pts.push(pa);
            for &t in &*t_params {
                pts.push(pa + edge_vec * t);
            }
            pts.push(pb);

            for &adj_fi in adj_faces {
                if adj_fi == fi {
                    continue;
                }
                for w in pts.windows(2) {
                    if (w[1] - w[0]).norm_squared() < DEGENERATE_LEN_SQ {
                        continue;
                    }
                    injections.push((
                        adj_fi,
                        SnapSegment {
                            start: w[0],
                            end: w[1],
                        },
                    ));
                }
            }
        }
    }

    for (fi, seg) in injections.drain(..) {
        if fi < segs.len() {
            let sb = (
                seg.start.x.to_bits(),
                seg.start.y.to_bits(),
                seg.start.z.to_bits(),
            );
            let eb = (
                seg.end.x.to_bits(),
                seg.end.y.to_bits(),
                seg.end.z.to_bits(),
            );
            let exists = segs[fi].iter().any(|s| {
                let ssb = (
                    s.start.x.to_bits(),
                    s.start.y.to_bits(),
                    s.start.z.to_bits(),
                );
                let seb = (s.end.x.to_bits(), s.end.y.to_bits(), s.end.z.to_bits());
                (ssb == sb && seb == eb) || (ssb == eb && seb == sb)
            });
            if !exists {
                segs[fi].push(seg);
            }
        }
    }
}

/// Repeatedly apply [`propagate_seam_vertices`] until no new segments are
/// generated, ensuring transitive seam propagation converges.
///
/// ## Algorithm — Fixed-Point Propagation
///
/// Each pass may inject new snap-segments into faces adjacent to already-
/// segmented faces.  Those newly-segmented faces may in turn have neighbours
/// that need propagation.  The loop terminates when a pass adds zero new
/// segments (segment count is monotonically non-decreasing).
///
/// ## Theorem — Termination
///
/// **Statement.**  Propagation converges in at most $D$ passes, where $D$ is
/// the diameter of the face-adjacency graph restricted to faces touched by
/// intersection curves.
///
/// **Proof.**  Each pass extends segments by one adjacency hop.  After pass $k$,
/// every face within $k$ hops of an originally-segmented face has been
/// processed.  Since the affected face set is finite (bounded by the mesh
/// face count $F$) and the segment count is monotonically non-decreasing,
/// the total number of passes is at most $\min(D, F)$.  The guard constant
/// `MAX_PROPAGATION_PASSES = 8` provides an $O(1)$ upper bound for safety.  ∎
pub(crate) fn propagate_seam_vertices_until_stable(
    faces: &[FaceData],
    segs: &mut [Vec<SnapSegment>],
    pool: &VertexPool,
) {
    const MAX_PROPAGATION_PASSES: usize = 8;
    use crate::domain::core::index::VertexId;

    if segs.is_empty() {
        return;
    }

    // Build undirected edge → face-index adjacency ONCE.
    type EdgeKey = (VertexId, VertexId);
    let mut edge_to_faces: HashMap<EdgeKey, AdjacentFaces> =
        HashMap::with_capacity(faces.len() * 3 / 2);
    for (fi, face) in faces.iter().enumerate() {
        let v = face.vertices;
        for i in 0..3_usize {
            let va = v[i];
            let vb = v[(i + 1) % 3];
            let key = if va < vb { (va, vb) } else { (vb, va) };
            edge_to_faces.entry(key).or_default().push(fi);
        }
    }

    let mut injections = Vec::new();
    let mut t_params = Vec::new();
    let mut pts = Vec::new();
    for _ in 0..MAX_PROPAGATION_PASSES {
        let before: usize = segs.iter().map(Vec::len).sum();
        propagate_seam_vertices_impl(
            faces,
            segs,
            pool,
            &edge_to_faces,
            &mut injections,
            &mut t_params,
            &mut pts,
        );
        let after: usize = segs.iter().map(Vec::len).sum();
        if after == before {
            break;
        }
    }
}
