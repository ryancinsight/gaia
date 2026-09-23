//! The co-refinement pass: one face against its snap segments, via a 2-D CDT.

use super::geom::{dominant_normal_axes, inside_triangle, midpoint_subdivide, project_2d};
use super::keys::{canonical_edge_key, canonical_segment_key};
use super::{CorefinerScratch, SeamVertexMap, SegBounds, EDGE_EPS, WELD_TOL_SQ};
use crate::application::csg::intersect::SnapSegment;
use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::application::delaunay::{Cdt, Pslg};
use crate::domain::core::constants::{
    DEGENERATE_NORMAL_REL_SQ, DEGENERATE_SEGMENT_REL_SQ, MAX_STEINER_PER_FACE, SLIVER_AREA2D_REL,
};
use crate::domain::core::index::VertexId;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use hashbrown::HashMap;

// ── Main entry point ─────────────────────────────────────────────────────────

/// Co-refine `face` against `snap_segments` using CDT-based 2-D projection.
///
/// Steiner vertices are inserted via `VertexPool::insert_or_weld`; the 3-D
/// face boundary polygon and snap chords are projected to 2-D (dropping the
/// dominant normal axis), fed to `Cdt::from_pslg` (Shewchuk predicates), then
/// lifted back to 3-D `VertexId`s.
///
/// When `seam_map` is provided (non-empty), edge Steiner vertices are looked up
/// from the global map rather than recomputed from raw snap-segment geometry.
/// This guarantees that adjacent faces sharing an edge produce identical
/// Steiner sequences → matching CDT triangulations → zero non-manifold edges.
pub(crate) fn corefine_face(
    face: &FaceData,
    snap_segments: &[SnapSegment],
    pool: &mut VertexPool,
    seam_map: &SeamVertexMap,
    scratch: &mut CorefinerScratch,
) -> Vec<FaceData> {
    scratch.clear();

    let a = *pool.position(face.vertices[0]);
    let b = *pool.position(face.vertices[1]);
    let c = *pool.position(face.vertices[2]);

    let face_n = (b - a).cross(c - a);
    // Scale-relative degeneracy: ‖n‖² vs ‖edge₁‖²·‖edge₂‖² (see constants.rs).
    let edge1_sq = (b - a).norm_squared();
    let edge2_sq = (c - a).norm_squared();
    if face_n.norm_squared() < DEGENERATE_NORMAL_REL_SQ * edge1_sq * edge2_sq {
        return Vec::new();
    }
    let face_n_unit = face_n / face_n.norm();
    let face_pts = [a, b, c];

    // Exact segment canonicalization:
    // remove degenerate and bit-identical duplicate constraints before O(s²)
    // crossing detection and PSLG resolve_crossings.
    // Scale-relative degenerate threshold: segments shorter than
    // sqrt(DEGENERATE_SEGMENT_REL_SQ) · max_edge are collapsed.
    let max_edge_sq = edge1_sq.max(edge2_sq).max((c - b).norm_squared());
    let seg_degen_sq = DEGENERATE_SEGMENT_REL_SQ * max_edge_sq;
    let dedup_snap_segments = &mut scratch.dedup_snap_segments;
    let seen_snap_segments = &mut scratch.seen_snap_segments;
    seen_snap_segments.reserve(snap_segments.len());
    for seg in snap_segments {
        if (seg.end - seg.start).norm_squared() < seg_degen_sq {
            continue;
        }
        let key = canonical_segment_key(seg);
        if seen_snap_segments.insert(key) {
            dedup_snap_segments.push(*seg);
        }
    }
    dedup_snap_segments.sort_unstable_by_key(canonical_segment_key);

    // ── Step 0: Choose projection axes ───────────────────────────────────────
    let (axis_u, axis_v) = dominant_normal_axes(face_n_unit);

    // ── Step 1: Edge Steiner vertices ─────────────────────────────────────────
    //
    // When a global SeamVertexMap is available, edge Steiners are looked up from
    // the map (keyed by canonical edge pair) rather than recomputed from raw
    // snap-segment geometry.  This guarantees that adjacent faces sharing the
    // same edge receive the exact same Steiner VertexIds.
    let edge_steiners = &mut scratch.edge_steiners;
    for es in edge_steiners.iter_mut() {
        es.reserve(4);
    }

    let use_seam_map = !seam_map.is_empty();

    if use_seam_map {
        // ── Seam-map path: look up pre-registered Steiners per edge ──────
        for ei in 0..3_usize {
            let va_id = face.vertices[ei];
            let vb_id = face.vertices[(ei + 1) % 3];
            let key = canonical_edge_key(va_id, vb_id);
            if let Some(steiners) = seam_map.get(&key) {
                // t-parameters in the map are relative to the canonical
                // direction (min→max).  Convert to face-local direction
                // (va→vb) if needed.
                let flip = va_id > vb_id;
                for &(t_canon, vid) in steiners {
                    let t_local = if flip { 1.0 - t_canon } else { t_canon };
                    edge_steiners[ei].push((t_local, vid));
                }
            }
        }
    }

    // seg_vids[i] = [vid_of_start, vid_of_end]; None if not on face boundary.
    let seg_vids = &mut scratch.seg_vids;
    seg_vids.resize(dedup_snap_segments.len(), [None, None]);
    // Use a flat Vec for O(1) short-array dedup to preserve insertion order without allocation.
    let interior_vids = &mut scratch.interior_vids;
    // O(1) membership test; the Vec preserves insertion order for PSLG registration.
    let interior_vid_set = &mut scratch.interior_vid_set;
    interior_vid_set.reserve(dedup_snap_segments.len());

    for (si, seg) in dedup_snap_segments.iter().enumerate() {
        for (ep, &p3d) in [seg.start, seg.end].iter().enumerate() {
            'edge_search: for ei in 0..3_usize {
                let va = face_pts[ei];
                let vb = face_pts[(ei + 1) % 3];
                let edge = vb - va;
                let edge_sq = edge.norm_squared();
                if edge_sq < DEGENERATE_NORMAL_REL_SQ * max_edge_sq {
                    continue;
                }
                let t = (p3d - va).dot(edge) / edge_sq;
                if t <= EDGE_EPS || t >= 1.0 - EDGE_EPS {
                    continue;
                }
                let interp = va + edge * t;
                if (p3d - interp).norm_squared() > WELD_TOL_SQ * 4.0 {
                    continue;
                }
                // If seam map is in use, the edge Steiner is already
                // pre-registered in the map above.  Just resolve the
                // seg_vid linkage by finding the matching VertexId.
                let vid = if use_seam_map {
                    // Find the nearest pre-registered Steiner.
                    let mut best: Option<VertexId> = None;
                    for &(_, sv) in &edge_steiners[ei] {
                        let sp = *pool.position(sv);
                        if (p3d - sp).norm_squared() < WELD_TOL_SQ * 4.0 {
                            best = Some(sv);
                            break;
                        }
                    }
                    if let Some(v) = best {
                        v
                    } else {
                        // Fallback: insert and add (shouldn't happen
                        // if build_seam_vertex_map was complete).
                        let v = pool.insert_or_weld(p3d, face_n_unit);
                        edge_steiners[ei].push((t, v));
                        v
                    }
                } else {
                    let v = pool.insert_or_weld(p3d, face_n_unit);
                    edge_steiners[ei].push((t, v));
                    v
                };
                seg_vids[si][ep] = Some(vid);
                break 'edge_search;
            }

            // Corner vertex fallback — if snap endpoint is within WELD_TOL_SQ
            // of a face corner, snap it to the corner VertexId.
            if seg_vids[si][ep].is_none() {
                for ci in 0..3_usize {
                    if (p3d - face_pts[ci]).norm_squared() < WELD_TOL_SQ * 4.0 {
                        seg_vids[si][ep] = Some(face.vertices[ci]);
                        break;
                    }
                }
            }

            // Interior endpoint fallback
            if seg_vids[si][ep].is_none() && inside_triangle(p3d, a, b, c, face_n, axis_u, axis_v) {
                let vid = pool.insert_or_weld(p3d, face_n_unit);
                if interior_vid_set.insert(vid) {
                    interior_vids.push(vid);
                }
            }
        }
    }

    // ── Step 2: Interior crossing points ─────────────────────────────────────
    //
    // # Theorem — 1-D Interval Pre-Filter Soundness
    //
    // Two line segments $s_1, s_2$ in the face plane can only intersect in the
    // interior (i.e., at parameter values $t \in (\epsilon, 1-\epsilon)$ and
    // $s \in (\epsilon, 1-\epsilon)$) if their axis-aligned bounding intervals
    // overlap on **both** projected axes $(u, v)$.  A pair whose 1-D intervals
    // are disjoint on either axis is guaranteed to have no interior crossing,
    // so it can be skipped without geometric computation.  This reduces
    // the expected cost from $O(S^2)$ to $O(S^2 \cdot p)$ where $p$ is the
    // fraction of pairs with overlapping bounding boxes — typically $p \ll 1$
    // for well-distributed intersection curves.
    //
    // **Proof.**  If two segments' $u$-intervals $[u_{\min,1}, u_{\max,1}]$ and
    // $[u_{\min,2}, u_{\max,2}]$ are disjoint, then for any point on $s_1$ with
    // $u$-coordinate $u_1$ and any point on $s_2$ with $u$-coordinate $u_2$,
    // $u_1 \neq u_2$.  Two segments can only intersect at a point with the same
    // coordinates on both, so they cannot intersect.  The same argument applies
    // to the $v$-axis.  ∎
    let n_sq = face_n.norm_squared();

    // Pre-compute 1-D bounding intervals for each segment on (axis_u, axis_v).
    let seg_bounds = &mut scratch.seg_bounds;
    for seg in dedup_snap_segments.iter() {
        let su = seg.start[axis_u];
        let eu = seg.end[axis_u];
        let sv = seg.start[axis_v];
        let ev = seg.end[axis_v];
        seg_bounds.push(SegBounds {
            u_min: su.min(eu),
            u_max: su.max(eu),
            v_min: sv.min(ev),
            v_max: sv.max(ev),
        });
    }

    for i in 0..dedup_snap_segments.len() {
        for j in (i + 1)..dedup_snap_segments.len() {
            // 1-D interval overlap pre-filter: skip pairs with disjoint
            // bounding boxes on either projected axis.
            if seg_bounds[i].u_max < seg_bounds[j].u_min
                || seg_bounds[j].u_max < seg_bounds[i].u_min
                || seg_bounds[i].v_max < seg_bounds[j].v_min
                || seg_bounds[j].v_max < seg_bounds[i].v_min
            {
                continue;
            }

            let s1 = &dedup_snap_segments[i];
            let s2 = &dedup_snap_segments[j];
            let d1 = s1.end - s1.start;
            let d2 = s2.end - s2.start;
            let r = s2.start - s1.start;
            let denom = d1.cross(d2).dot(face_n);
            let min_d = 1e-14 * n_sq.sqrt() * (d1.norm() + d2.norm());
            if denom.abs() < min_d {
                continue;
            }
            let t_param = r.cross(d2).dot(face_n) / denom;
            let s_param = r.cross(d1).dot(face_n) / denom;
            if t_param <= EDGE_EPS || t_param >= 1.0 - EDGE_EPS {
                continue;
            }
            if s_param <= EDGE_EPS || s_param >= 1.0 - EDGE_EPS {
                continue;
            }
            let crossing = s1.start + d1 * t_param;
            if !inside_triangle(crossing, a, b, c, face_n, axis_u, axis_v) {
                continue;
            }
            let vid = pool.insert_or_weld(crossing, face_n_unit);
            if interior_vid_set.insert(vid) {
                interior_vids.push(vid);
            }
        }
    }

    // ── Dedup edge steiners before length checks ───────────────────────────────
    // Deferring deduplication avoids O(N^2) linear scan overhead on highly refined edges.
    for es in edge_steiners.iter_mut() {
        es.sort_by(|a, b| a.0.total_cmp(&b.0));
        es.dedup_by_key(|&mut (_, vid)| vid);
    }

    // ── Step 3: Early exit ────────────────────────────────────────────────────
    if edge_steiners.iter().all(std::vec::Vec::is_empty) && interior_vids.is_empty() {
        return vec![*face];
    }

    // ── Steiner count guard ───────────────────────────────────────────────────
    //
    // # Theorem — Steiner Guard Soundness
    //
    // `midpoint_subdivide` produces a fan triangulation that covers the original
    // face and preserves all edge-Steiner vertices on shared boundary edges.
    // No interior constraint is resolved (sacrificed for stability), but no
    // T-junctions are introduced because edge Steiners appear identically on
    // adjacent co-refined faces.  The CDT complexity O(s²) for `s` interior
    // Steiners is bounded: with `s ≤ MAX_STEINER_PER_FACE` the fallback triggers
    // before the CDT receives a pathological input. ∎
    {
        let total_steiner: usize =
            edge_steiners.iter().map(|e| e.len()).sum::<usize>() + interior_vids.len();
        if total_steiner > MAX_STEINER_PER_FACE {
            tracing::warn!(
                total_steiner,
                MAX_STEINER_PER_FACE,
                "corefine_face: Steiner count exceeds limit — falling back to midpoint subdivision"
            );
            return midpoint_subdivide(face, edge_steiners, pool, face_n);
        }
    }

    // ── Step 4: Build ordered boundary polygon ────────────────────────────────
    // Compute exact capacity: 3 corners + sum of Steiners per edge.
    let steiner_count: usize = edge_steiners.iter().map(|e| e.len()).sum();
    let boundary_vids = &mut scratch.boundary_vids;
    boundary_vids.reserve(3 + steiner_count);
    for ei in 0..3_usize {
        boundary_vids.push(face.vertices[ei]);
        for &(_, vid) in &edge_steiners[ei] {
            boundary_vids.push(vid);
        }
    }
    boundary_vids.dedup();
    if boundary_vids.len() < 3 {
        return vec![*face];
    }

    // ── Sliver-face guard ─────────────────────────────────────────────────────
    // If the 2D projected area of the boundary polygon is near-zero, the CDT
    // will be degenerate (all vertices colinear in 2D).  This happens for very
    // thin sliver triangles at shallow-angle elbow-cylinder junctions.
    // In this case, fall back to simple midpoint-subdivision: split the edge
    // containing each edge-Steiner point and return the sub-triangles directly.
    {
        let mut area2 = 0.0_f64;
        let mut perim_sq = 0.0_f64;
        let nb = boundary_vids.len();
        for i in 0..nb {
            let pa = *pool.position(boundary_vids[i]);
            let pb = *pool.position(boundary_vids[(i + 1) % nb]);
            let (pu, pv) = project_2d(pa, axis_u, axis_v);
            let (qu, qv) = project_2d(pb, axis_u, axis_v);
            area2 += pu * qv - qu * pv;
            perim_sq += (qu - pu) * (qu - pu) + (qv - pv) * (qv - pv);
        }
        // Scale-relative: compare area² against ∑edge² (both scale as length²).
        if area2.abs() < SLIVER_AREA2D_REL * perim_sq {
            // Sliver: produce sub-triangles by splitting each Steiner-containing
            // edge and fan-stitching. This guarantees Steiner points appear as
            // vertices on the shared boundary even when CDT would degenerate.
            return midpoint_subdivide(face, edge_steiners, pool, face_n);
        }
    }

    // ── Step 5: Build PSLG from 2-D projections ───────────────────────────────
    // Map: VertexId → PslgVertexId.
    //
    // ## Memory note
    //
    // Previously `vid_to_pslg` was a `Vec<Option<PslgVertexId>>` of length
    // `pool.len() + 1`, which allocated O(pool_size) memory per `corefine_face`
    // call.  With pools of 2 000+ vertices and 50+ faces this became 100 k
    // entries.  A `HashMap` with an exact capacity hint is O(face_vertex_count)
    // and avoids the large up-front allocation entirely.
    //
    // ## Performance note (CW5 audit)
    //
    // Replaced `Vec<(VertexId, PslgVertexId)>` with `HashMap` for O(1) lookup.
    // The previous linear-scan `iter().find()` was O(n) per lookup, making
    // PSLG registration O(n²) in the number of Steiner vertices per face.
    let register_cap = boundary_vids.len() + interior_vids.len();
    let vid_to_pslg = &mut scratch.vid_to_pslg;
    vid_to_pslg.reserve(register_cap);
    let pslg_to_vid = &mut scratch.pslg_to_vid;
    pslg_to_vid.reserve(register_cap);
    let mut pslg = Pslg::new();

    // Helper: register a VertexId into the PSLG if not already there.
    // Returns the PSLG vertex id.
    let register = |vid: VertexId,
                    pslg: &mut Pslg,
                    vid_to_pslg: &mut HashMap<VertexId, PslgVertexId>,
                    pslg_to_vid: &mut Vec<VertexId>,
                    pool: &VertexPool|
     -> PslgVertexId {
        if let Some(&pid) = vid_to_pslg.get(&vid) {
            return pid;
        }
        let pos3d = *pool.position(vid);
        let (u, v) = project_2d(pos3d, axis_u, axis_v);
        let pid = pslg.add_vertex(u, v);
        vid_to_pslg.insert(vid, pid);
        pslg_to_vid.push(vid);
        pid
    };

    // Register boundary vertices in order.
    for &vid in &*boundary_vids {
        register(vid, &mut pslg, vid_to_pslg, pslg_to_vid, pool);
    }
    // Register interior Steiner vertices.
    for &vid in &*interior_vids {
        register(vid, &mut pslg, vid_to_pslg, pslg_to_vid, pool);
    }

    // Extract unique 2D points from PSLG for segment shattering.
    let unique_pts = &mut scratch.unique_pts;
    unique_pts.reserve(pslg.vertices().len());
    for v in pslg.vertices() {
        unique_pts.push([v.x, v.y]);
    }
    let pslg_edges = &mut scratch.pslg_edges;

    // Boundary polygon segments (ring).
    let nb = boundary_vids.len();
    let on_edge = &mut scratch.on_edge;
    for i in 0..nb {
        let va = boundary_vids[i];
        let vb = boundary_vids[(i + 1) % nb];
        let pa = vid_to_pslg[&va];
        let pb = vid_to_pslg[&vb];
        if pa != pb {
            let p1 = unique_pts[pa.idx()];
            let p2 = unique_pts[pb.idx()];
            crate::application::csg::arrangement::planar::collect_points_on_segment_interior_to_buf(
                unique_pts,
                p1,
                p2,
                (pa.idx(), pb.idx()),
                1e-8,
                1e-14,
                on_edge,
            );
            crate::application::csg::arrangement::planar::insert_shattered_subedges(
                on_edge, pslg_edges,
            );
        }
    }

    // Constraint segments from snap-segment endpoints.
    for vids in &*seg_vids {
        if let (Some(v0), Some(v1)) = (vids[0], vids[1]) {
            let p0_opt = vid_to_pslg.get(&v0).copied();
            let p1_opt = vid_to_pslg.get(&v1).copied();
            if let (Some(p0), Some(p1)) = (p0_opt, p1_opt)
                && p0 != p1
            {
                let pa = unique_pts[p0.idx()];
                let pb = unique_pts[p1.idx()];
                crate::application::csg::arrangement::planar::collect_points_on_segment_interior_to_buf(
                        unique_pts, pa, pb, (p0.idx(), p1.idx()), 1e-8, 1e-14, on_edge
                    );
                crate::application::csg::arrangement::planar::insert_shattered_subedges(
                    on_edge, pslg_edges,
                );
            }
        }
    }

    pslg_edges.sort_unstable();
    pslg_edges.dedup();
    for &(a, b) in &*pslg_edges {
        let _ = pslg.add_segment(PslgVertexId::from_usize(a), PslgVertexId::from_usize(b));
    }

    // ── Step 6: Build CDT ────────────────────────────────────────────────────
    // Resolve any interior segment crossings that arise from 3-D seam curves
    // whose 2-D projections cross (e.g. out-of-plane multi-branch junctions).
    // `resolve_crossings` now handles proper crossings, T-intersections, and
    // collinear overlaps, so this error path should be unreachable for all
    // geometrically valid CSG inputs.
    pslg.resolve_crossings();
    let cdt = match Cdt::try_from_pslg(&pslg) {
        Ok(cdt) => cdt,
        // Should not be reached after resolve_crossings handles all illegality
        // classes.  Log for diagnostics and fall back to the original face so
        // the overall mesh remains topologically consistent.
        Err(e) => {
            tracing::warn!(
                "corefine_face: PSLG validation failed after resolve_crossings: {e}. \
                 Returning original face (no CDT refinement applied)."
            );
            return vec![*face];
        }
    };
    let dt = cdt.triangulation();

    // ── Step 7: Emit triangles lifted back to 3-D ─────────────────────────────
    let mut triangles: Vec<FaceData> = Vec::new();
    for (_, tri) in dt.interior_triangles() {
        let [pv0, pv1, pv2] = tri.vertices;
        // Exclude super-triangle vertices (index >= pslg_to_vid.len()).
        if pv0.idx() >= pslg_to_vid.len()
            || pv1.idx() >= pslg_to_vid.len()
            || pv2.idx() >= pslg_to_vid.len()
        {
            continue;
        }
        let v0 = pslg_to_vid[pv0.idx()];
        let v1 = pslg_to_vid[pv1.idx()];
        let v2 = pslg_to_vid[pv2.idx()];
        if crate::infrastructure::storage::face_store::FaceData::untagged(v0, v1, v2)
            .is_degenerate()
        {
            continue;
        }

        let p0 = *pool.position(v0);
        let p1 = *pool.position(v1);
        let p2 = *pool.position(v2);
        let tri_n = (p1 - p0).cross(p2 - p0);
        // Scale-relative: compare ‖n‖² against ‖edge₁‖²·‖edge₂‖².
        let e1sq = (p1 - p0).norm_squared();
        let e2sq = (p2 - p0).norm_squared();
        if tri_n.norm_squared() < DEGENERATE_NORMAL_REL_SQ * e1sq * e2sq {
            continue;
        }
        if tri_n.dot(face_n) >= 0.0 {
            triangles.push(FaceData::new(v0, v1, v2, face.region));
        } else {
            triangles.push(FaceData::new(v0, v2, v1, face.region));
        }
    }

    if triangles.is_empty() {
        vec![*face]
    } else {
        triangles
    }
}
