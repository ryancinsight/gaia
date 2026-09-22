//! Coplanar cap-seam injection into barrel rim faces (Phase 2d).

use crate::application::csg::intersect::SnapSegment;
use crate::application::csg::predicates3d::point_on_segment_exact;
use crate::application::welding::snap::GridCell;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use hashbrown::{HashMap, HashSet};

use super::tolerances::{
    COINCIDENT_LEN_SQ, COLLINEAR_TOL_SQ, DEGENERATE_LEN_SQ, MIN_HASH_CELL, PARAM_DEDUP_TOL,
    PARAM_MARGIN, PARAM_MIN_SPAN,
};

/// Inject snap segments into barrel rim faces so they are corefined at every
/// seam vertex produced by `boolean_coplanar`.
///
/// ## Problem
///
/// `boolean_coplanar` clips cap triangles against each other in 2-D, producing
/// NEW intersection vertices (e.g., where an A-cap interior edge crosses a B-cap
/// interior edge).  These vertices appear on the *cap* side of the mesh but NOT
/// on the adjacent barrel face's rim edge, creating T-junctions → boundary edges
/// in the output.
///
/// ## Algorithm (Spatial-Hash Accelerated)
///
/// The naïve O(B × P) loop — for each barrel face, test every seam position — is
/// replaced by an O(R × k + P) algorithm:
///
/// 1. **Pre-filter** barrel faces to "rim faces" (exactly 2 on-plane vertices).
///    Non-rim faces are skipped immediately without any seam-position work.
///    Reduces the outer-loop count from B (all barrel faces) to R ≤ B.
///
/// 2. **Build seam-position spatial hash**: map each seam point into a 1 mm
///    grid cell.  Cost: O(P).
///
/// 3. **Query per rim face**: sample 5 points along the rim edge at
///    t ∈ {0, ¼, ½, ¾, 1}, query the 27-cell neighbourhood of each sample
///    (135 cells total), collect candidate seam indices, deduplicate, then
///    run the collinearity test only on candidates.  Cost: O(R × k) where
///    k = seam positions per cell ≈ 1–3 for millifluidic meshes.
///
/// ## Theorem (Spatial Hash Correctness)
///
/// For any seam position P strictly on rim edge [pa, pb] with |pb − pa| ≤ 8 mm,
/// the nearest sample point is within 1 mm of P.  Proof: samples divide [pa, pb]
/// into 4 equal sub-intervals of length |pb − pa|/4 ≤ 2 mm.  P's distance to the
/// nearest sample is at most |pb − pa|/8 ≤ 1 mm = cell size.  Therefore P lands
/// in the 27-cell neighbourhood (radius 1 = 1 mm) of that sample's GridCell.  For
/// edges > 8 mm, the fallback collinearity test (part of the inner loop) still
/// runs correctly — coverage is only reduced to O(1/cell_size) density.  QED.
///
/// ## Complexity
///
/// | Phase | Cost |
/// |-------|------|
/// | Rim-face pre-filter | O(B) |
/// | Seam-position hash build | O(P) |
/// | Query + collinearity test | O(R × 135 × k) |
/// | **Total** | **O(B + P + R × k)** — vs O(B × P) naïve |
///
/// For millifluidic meshes with k ≈ 1–3: effectively **O(B + P)**.
///
/// ## Collinearity test
///
/// Point `s` lies on segment `pa→pb` iff the cross-product `(pb−pa)×(s−pa)` is
/// the zero vector (collinear) and the dot-product parameter
/// `t = (s−pa)·(pb−pa) / |pb−pa|²` lies in `(PARAM_MARGIN, 1−PARAM_MARGIN)`.
///
/// Uses `COLLINEAR_TOL_SQ` (1e-6 on cross²/edge², i.e., |cross|/|edge| < 1e-3),
/// matching `propagate_seam_vertices` for consistent seam detection.
pub fn inject_cap_seam_into_barrels(
    barrel_faces: &[FaceData],
    coplanar_used: &HashSet<usize>,
    plane_pt: &Point3r,
    plane_n: &Vector3r,
    seam_positions: &[Point3r],
    segs_out: &mut [Vec<SnapSegment>],
    pool: &VertexPool,
) {
    let plane_n_len_sq = plane_n.dot(*plane_n);
    if plane_n_len_sq < DEGENERATE_LEN_SQ || seam_positions.is_empty() {
        return;
    }
    let plane_n_len = plane_n_len_sq.sqrt();

    const ON_TOL: Real = 1e-7; // signed-distance tolerance (relative to normal length)
    let tol = ON_TOL * plane_n_len;

    // ── Phase 1: Pre-filter barrel faces to rim faces ─────────────────────────
    // Rim face: exactly 2 on-plane vertices (the rim edge [pa, pb] lies on the
    // cap plane).  We pre-compute all rim edges once instead of re-detecting
    // them inside the seam-position loop.
    struct RimFace {
        face_idx: usize,
        pa: Point3r,
        pb: Point3r,
    }

    let mut rim_faces: Vec<RimFace> = Vec::with_capacity(barrel_faces.len() / 4);
    for (face_idx, face) in barrel_faces.iter().enumerate() {
        if coplanar_used.contains(&face_idx) {
            continue;
        }
        let v0 = *pool.position(face.vertices[0]);
        let v1 = *pool.position(face.vertices[1]);
        let v2 = *pool.position(face.vertices[2]);
        let d0 = (v0 - plane_pt).dot(*plane_n);
        let d1 = (v1 - plane_pt).dot(*plane_n);
        let d2 = (v2 - plane_pt).dot(*plane_n);
        let on0 = d0.abs() < tol;
        let on1 = d1.abs() < tol;
        let on2 = d2.abs() < tol;
        let on_count = u8::from(on0) + u8::from(on1) + u8::from(on2);
        if on_count != 2 {
            continue;
        }
        let (pa, pb) = match (on0, on1, on2) {
            (true, true, false) => (v0, v1),
            (true, false, true) => (v0, v2),
            (false, true, true) => (v1, v2),
            _ => continue,
        };
        let edge_len_sq = (pb - pa).norm_squared();
        if edge_len_sq < DEGENERATE_LEN_SQ {
            continue;
        }
        rim_faces.push(RimFace { face_idx, pa, pb });
    }

    if rim_faces.is_empty() {
        return;
    }

    // ── Phase 2: Build seam-position spatial hash ─────────────────────────────
    // The sample spacing is edge_len / 4, so any seam point on the rim lies
    // within edge_len / 8 of some sample. Size the hash cells from the longest
    // rim edge so that a 27-cell neighborhood remains complete for every rim
    // face processed in this pass.
    let max_rim_edge_len = rim_faces
        .iter()
        .map(|rim| (rim.pb - rim.pa).norm())
        .fold(0.0_f64, f64::max);
    let hash_cell = (max_rim_edge_len / 8.0).max(MIN_HASH_CELL);
    let inv_cell = 1.0 / hash_cell;

    let mut seam_hash: HashMap<GridCell, Vec<usize>> =
        HashMap::with_capacity(seam_positions.len() * 2);
    for (i, s) in seam_positions.iter().enumerate() {
        seam_hash
            .entry(GridCell::from_point_round(s, inv_cell))
            .or_default()
            .push(i);
    }

    // ── Phase 3: Query & inject ───────────────────────────────────────────────
    let mut candidates: Vec<usize> = Vec::new();
    let mut cut_params: Vec<Real> = Vec::new();
    let mut params: Vec<Real> = Vec::new();

    for rim in &rim_faces {
        let RimFace { face_idx, pa, pb } = *rim;
        let edge = pb - pa;
        let edge_len_sq = edge.norm_squared();

        // 5 sample points at t ∈ {0, ¼, ½, ¾, 1} along the rim edge.
        // Deduplication via sort+dedup on the candidate index list.
        candidates.clear();
        for k in 0..=4_u8 {
            let t = Real::from(k) * 0.25;
            let sample = pa + edge * t;
            let home = GridCell::from_point_round(&sample, inv_cell);
            for cell in home.neighborhood_27() {
                if let Some(idxs) = seam_hash.get(&cell) {
                    candidates.extend_from_slice(idxs);
                }
            }
        }

        if candidates.is_empty() {
            continue;
        }
        candidates.sort_unstable();
        candidates.dedup();

        // Collinearity test for each candidate seam position.
        cut_params.clear();
        for &i in &candidates {
            let s = &seam_positions[i];

            // Guard: s must lie on the cap plane.
            let ds = (*s - plane_pt).dot(*plane_n);
            if ds.abs() > tol * 10.0 {
                continue;
            }

            // Exact-first on-segment detection.
            if let Some(t_exact) = point_on_segment_exact(&pa, &pb, s) {
                if t_exact > PARAM_MARGIN && t_exact < 1.0 - PARAM_MARGIN {
                    cut_params.push(t_exact);
                }
                continue;
            }

            // Tolerance-based collinearity + parameter check.
            // True angular check: sin²(θ) = |cross|² / (|edge|² · |sp|²)
            let sp = *s - pa;
            let sp_len_sq = sp.norm_squared();
            if sp_len_sq < COINCIDENT_LEN_SQ {
                continue; // s ≈ pa, skip (t ≈ 0, not interior)
            }
            let cross = edge.cross(sp);
            if cross.norm_squared() <= COLLINEAR_TOL_SQ * edge_len_sq * sp_len_sq {
                let t = sp.dot(edge) / edge_len_sq;
                if t > PARAM_MARGIN && t < 1.0 - PARAM_MARGIN {
                    cut_params.push(t);
                }
            }
        }

        if cut_params.is_empty() {
            continue;
        }

        // Sort and deduplicate cut parameters, then emit sub-interval SnapSegments.
        cut_params.sort_by(|a, b| a.total_cmp(b));
        cut_params.dedup_by(|a, b| (*a - *b).abs() < PARAM_DEDUP_TOL);

        params.clear();
        params.push(0.0);
        params.extend_from_slice(&cut_params);
        params.push(1.0);

        for w in params.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            if (t1 - t0).abs() < PARAM_MIN_SPAN {
                continue;
            }
            let start_3d = pa + edge * t0;
            let end_3d = pa + edge * t1;
            if (end_3d - start_3d).norm_squared() < DEGENERATE_LEN_SQ {
                continue;
            }
            if face_idx < segs_out.len() {
                segs_out[face_idx].push(SnapSegment {
                    start: start_3d,
                    end: end_3d,
                });
            }
        }
    }
}
