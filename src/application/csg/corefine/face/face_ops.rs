//! Private CDT co-refinement helpers, extracted from `face.rs` to keep it under 500 lines.
//! Called exclusively by `corefine_face`; all items are `pub(super)`.

use super::super::geom::{inside_triangle, midpoint_subdivide, project_2d};
use super::super::keys::{canonical_edge_key, canonical_segment_key};
use super::super::{SeamVertexMap, SegBounds, EDGE_EPS, WELD_TOL_SQ};
use super::{
    EdgeSteinerBuffers, EdgeSteinerLists, FaceProjection, InteriorVertexBuffers, PslgBuffers,
};
use crate::application::csg::intersect::SnapSegment;
use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::application::delaunay::{Cdt, Pslg};
use crate::domain::core::constants::{
    DEGENERATE_NORMAL_REL_SQ, MAX_STEINER_PER_FACE, SLIVER_AREA2D_REL,
};
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Real, Vector3r};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use hashbrown::{HashMap, HashSet};

/// Removes degenerate snap segments and canonicalizes the survivors by endpoint bits.
pub(super) fn dedup_and_sort_snap_segments(
    snap_segments: &[SnapSegment],
    seg_degen_sq: Real,
    dedup_snap_segments: &mut Vec<SnapSegment>,
    seen_snap_segments: &mut HashSet<([u64; 3], [u64; 3])>,
) {
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
}

/// Collects boundary edge Steiners and interior endpoint vertices from snap segments.
pub(super) fn collect_edge_steiners(
    face: &FaceData,
    snap_segments: &[SnapSegment],
    pool: &mut VertexPool,
    seam_map: &SeamVertexMap,
    projection: &FaceProjection,
    buffers: &mut EdgeSteinerBuffers<'_>,
) {
    for es in buffers.edge_steiners.iter_mut() {
        es.reserve(4);
    }

    let use_seam_map = !seam_map.is_empty();
    if use_seam_map {
        for (ei, edge_steiner) in buffers.edge_steiners.iter_mut().enumerate().take(3_usize) {
            let start_vid = face.vertices[ei];
            let end_vid = face.vertices[(ei + 1) % 3];
            let key = canonical_edge_key(start_vid, end_vid);
            if let Some(steiners) = seam_map.get(&key) {
                let flip = start_vid > end_vid;
                for &(t_canon, vid) in steiners {
                    let t_local = if flip { 1.0 - t_canon } else { t_canon };
                    edge_steiner.push((t_local, vid));
                }
            }
        }
    }

    buffers.seg_vids.resize(snap_segments.len(), [None, None]);
    buffers
        .interior
        .interior_vid_set
        .reserve(snap_segments.len());

    for (si, seg) in snap_segments.iter().enumerate() {
        for (ep, &p3d) in [seg.start, seg.end].iter().enumerate() {
            'edge_search: for ei in 0..3_usize {
                let va = projection.face_pts[ei];
                let vb = projection.face_pts[(ei + 1) % 3];
                let edge = vb - va;
                let edge_sq = edge.norm_squared();
                if edge_sq < DEGENERATE_NORMAL_REL_SQ * projection.max_edge_sq {
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

                let vid = if use_seam_map {
                    let mut best: Option<VertexId> = None;
                    for &(_, sv) in &buffers.edge_steiners[ei] {
                        let sp = *pool.position(sv);
                        if (p3d - sp).norm_squared() < WELD_TOL_SQ * 4.0 {
                            best = Some(sv);
                            break;
                        }
                    }
                    if let Some(v) = best {
                        v
                    } else {
                        let v = pool.insert_or_weld(p3d, projection.face_n_unit);
                        buffers.edge_steiners[ei].push((t, v));
                        v
                    }
                } else {
                    let v = pool.insert_or_weld(p3d, projection.face_n_unit);
                    buffers.edge_steiners[ei].push((t, v));
                    v
                };
                buffers.seg_vids[si][ep] = Some(vid);
                break 'edge_search;
            }

            if buffers.seg_vids[si][ep].is_none() {
                for (ci, &face_pt) in projection.face_pts.iter().enumerate() {
                    if (p3d - face_pt).norm_squared() < WELD_TOL_SQ * 4.0 {
                        buffers.seg_vids[si][ep] = Some(face.vertices[ci]);
                        break;
                    }
                }
            }

            if buffers.seg_vids[si][ep].is_none()
                && inside_triangle(
                    p3d,
                    projection.face_pts[0],
                    projection.face_pts[1],
                    projection.face_pts[2],
                    projection.face_n,
                    projection.axis_u,
                    projection.axis_v,
                )
            {
                let vid = pool.insert_or_weld(p3d, projection.face_n_unit);
                if buffers.interior.interior_vid_set.insert(vid) {
                    buffers.interior.interior_vids.push(vid);
                }
            }
        }
    }
}

/// Inserts interior Steiner vertices at proper crossings of projected snap segments.
pub(super) fn collect_interior_crossings(
    snap_segments: &[SnapSegment],
    pool: &mut VertexPool,
    projection: &FaceProjection,
    seg_bounds: &mut Vec<SegBounds>,
    interior: &mut InteriorVertexBuffers<'_>,
) {
    let n_sq = projection.face_n.norm_squared();
    for seg in snap_segments {
        let su = seg.start[projection.axis_u];
        let eu = seg.end[projection.axis_u];
        let sv = seg.start[projection.axis_v];
        let ev = seg.end[projection.axis_v];
        seg_bounds.push(SegBounds {
            u_min: su.min(eu),
            u_max: su.max(eu),
            v_min: sv.min(ev),
            v_max: sv.max(ev),
        });
    }

    for i in 0..snap_segments.len() {
        for j in (i + 1)..snap_segments.len() {
            if seg_bounds[i].u_max < seg_bounds[j].u_min
                || seg_bounds[j].u_max < seg_bounds[i].u_min
                || seg_bounds[i].v_max < seg_bounds[j].v_min
                || seg_bounds[j].v_max < seg_bounds[i].v_min
            {
                continue;
            }

            let s1 = &snap_segments[i];
            let s2 = &snap_segments[j];
            let d1 = s1.end - s1.start;
            let d2 = s2.end - s2.start;
            let r = s2.start - s1.start;
            let denom = d1.cross(d2).dot(projection.face_n);
            let min_d = 1e-14 * n_sq.sqrt() * (d1.norm() + d2.norm());
            if denom.abs() < min_d {
                continue;
            }

            let t_param = r.cross(d2).dot(projection.face_n) / denom;
            let s_param = r.cross(d1).dot(projection.face_n) / denom;
            if t_param <= EDGE_EPS || t_param >= 1.0 - EDGE_EPS {
                continue;
            }
            if s_param <= EDGE_EPS || s_param >= 1.0 - EDGE_EPS {
                continue;
            }

            let crossing = s1.start + d1 * t_param;
            if !inside_triangle(
                crossing,
                projection.face_pts[0],
                projection.face_pts[1],
                projection.face_pts[2],
                projection.face_n,
                projection.axis_u,
                projection.axis_v,
            ) {
                continue;
            }

            let vid = pool.insert_or_weld(crossing, projection.face_n_unit);
            if interior.interior_vid_set.insert(vid) {
                interior.interior_vids.push(vid);
            }
        }
    }
}

/// Builds the ordered boundary ring from the face corners and sorted edge Steiners.
pub(super) fn build_boundary_polygon(
    face: &FaceData,
    edge_steiners: &EdgeSteinerLists,
    boundary_vids: &mut Vec<VertexId>,
) -> bool {
    let steiner_count: usize = edge_steiners.iter().map(Vec::len).sum();
    boundary_vids.reserve(3 + steiner_count);
    for (ei, edge_steiner) in edge_steiners.iter().enumerate().take(3_usize) {
        boundary_vids.push(face.vertices[ei]);
        for &(_, vid) in edge_steiner {
            boundary_vids.push(vid);
        }
    }
    boundary_vids.dedup();
    boundary_vids.len() >= 3
}

/// Applies the early-return guards and prepares the ordered boundary ring for CDT.
pub(super) fn prepare_boundary_or_fallback(
    face: &FaceData,
    edge_steiners: &EdgeSteinerLists,
    interior_vids: &[VertexId],
    boundary_vids: &mut Vec<VertexId>,
    pool: &mut VertexPool,
    projection: &FaceProjection,
) -> Option<Vec<FaceData>> {
    if edge_steiners.iter().all(std::vec::Vec::is_empty) && interior_vids.is_empty() {
        return Some(vec![*face]);
    }

    let total_steiner: usize =
        edge_steiners.iter().map(Vec::len).sum::<usize>() + interior_vids.len();
    if total_steiner > MAX_STEINER_PER_FACE {
        tracing::warn!(
            total_steiner,
            MAX_STEINER_PER_FACE,
            "corefine_face: Steiner count exceeds limit — falling back to midpoint subdivision"
        );
        return Some(midpoint_subdivide(
            face,
            edge_steiners,
            pool,
            projection.face_n,
        ));
    }

    if !build_boundary_polygon(face, edge_steiners, boundary_vids) {
        return Some(vec![*face]);
    }

    if boundary_polygon_is_sliver(boundary_vids, pool, projection.axis_u, projection.axis_v) {
        return Some(midpoint_subdivide(
            face,
            edge_steiners,
            pool,
            projection.face_n,
        ));
    }

    None
}

/// Tests whether the projected boundary polygon is too thin for a stable CDT.
pub(super) fn boundary_polygon_is_sliver(
    boundary_vids: &[VertexId],
    pool: &VertexPool,
    axis_u: usize,
    axis_v: usize,
) -> bool {
    let mut area2 = 0.0_f64;
    let mut perim_sq = 0.0_f64;
    for i in 0..boundary_vids.len() {
        let pa = *pool.position(boundary_vids[i]);
        let pb = *pool.position(boundary_vids[(i + 1) % boundary_vids.len()]);
        let (pu, pv) = project_2d(pa, axis_u, axis_v);
        let (qu, qv) = project_2d(pb, axis_u, axis_v);
        area2 += pu * qv - qu * pv;
        perim_sq += (qu - pu) * (qu - pu) + (qv - pv) * (qv - pv);
    }

    area2.abs() < SLIVER_AREA2D_REL * perim_sq
}

/// Registers a welded 3-D vertex into the face-local PSLG and returns its PSLG slot.
pub(super) fn register_pslg_vertex(
    vid: VertexId,
    pslg: &mut Pslg,
    vid_to_pslg: &mut HashMap<VertexId, PslgVertexId>,
    pslg_to_vid: &mut Vec<VertexId>,
    pool: &VertexPool,
    axis_u: usize,
    axis_v: usize,
) -> PslgVertexId {
    if let Some(&pid) = vid_to_pslg.get(&vid) {
        return pid;
    }

    let pos3d = *pool.position(vid);
    let (u, v) = project_2d(pos3d, axis_u, axis_v);
    let pid = pslg.add_vertex(u, v);
    vid_to_pslg.insert(vid, pid);
    pslg_to_vid.push(vid);
    pid
}

/// Builds the face PSLG by projecting the boundary ring and shattering snap constraints.
pub(super) fn build_face_pslg(
    boundary_vids: &[VertexId],
    interior_vids: &[VertexId],
    seg_vids: &[[Option<VertexId>; 2]],
    pool: &VertexPool,
    axis_u: usize,
    axis_v: usize,
    buffers: &mut PslgBuffers<'_>,
) -> Pslg {
    let register_cap = boundary_vids.len() + interior_vids.len();
    buffers.vid_to_pslg.reserve(register_cap);
    buffers.pslg_to_vid.reserve(register_cap);

    let mut pslg = Pslg::new();
    for &vid in boundary_vids {
        register_pslg_vertex(
            vid,
            &mut pslg,
            &mut *buffers.vid_to_pslg,
            &mut *buffers.pslg_to_vid,
            pool,
            axis_u,
            axis_v,
        );
    }
    for &vid in interior_vids {
        register_pslg_vertex(
            vid,
            &mut pslg,
            &mut *buffers.vid_to_pslg,
            &mut *buffers.pslg_to_vid,
            pool,
            axis_u,
            axis_v,
        );
    }

    buffers.unique_pts.reserve(pslg.vertices().len());
    for v in pslg.vertices() {
        buffers.unique_pts.push([v.x, v.y]);
    }

    for i in 0..boundary_vids.len() {
        let va = boundary_vids[i];
        let vb = boundary_vids[(i + 1) % boundary_vids.len()];
        let pa = buffers.vid_to_pslg[&va];
        let pb = buffers.vid_to_pslg[&vb];
        if pa == pb {
            continue;
        }

        let p1 = buffers.unique_pts[pa.idx()];
        let p2 = buffers.unique_pts[pb.idx()];
        crate::application::csg::arrangement::planar::collect_points_on_segment_interior_to_buf(
            &*buffers.unique_pts,
            p1,
            p2,
            (pa.idx(), pb.idx()),
            1e-8,
            1e-14,
            &mut *buffers.on_edge,
        );
        crate::application::csg::arrangement::planar::insert_shattered_subedges(
            &mut *buffers.on_edge,
            &mut *buffers.pslg_edges,
        );
    }

    for vids in seg_vids {
        if let (Some(v0), Some(v1)) = (vids[0], vids[1]) {
            let p0_opt = buffers.vid_to_pslg.get(&v0).copied();
            let p1_opt = buffers.vid_to_pslg.get(&v1).copied();
            if let (Some(p0), Some(p1)) = (p0_opt, p1_opt) {
                if p0 == p1 {
                    continue;
                }
                let pa = buffers.unique_pts[p0.idx()];
                let pb = buffers.unique_pts[p1.idx()];
                crate::application::csg::arrangement::planar::collect_points_on_segment_interior_to_buf(
                    &*buffers.unique_pts,
                    pa,
                    pb,
                    (p0.idx(), p1.idx()),
                    1e-8,
                    1e-14,
                    &mut *buffers.on_edge,
                );
                crate::application::csg::arrangement::planar::insert_shattered_subedges(
                    &mut *buffers.on_edge,
                    &mut *buffers.pslg_edges,
                );
            }
        }
    }

    buffers.pslg_edges.sort_unstable();
    buffers.pslg_edges.dedup();
    for &(a, b) in &*buffers.pslg_edges {
        let _ = pslg.add_segment(PslgVertexId::from_usize(a), PslgVertexId::from_usize(b));
    }

    pslg
}

/// Resolves PSLG crossings and constructs the constrained Delaunay triangulation.
pub(super) fn build_face_cdt(pslg: &mut Pslg) -> Option<Cdt> {
    pslg.resolve_crossings();
    match Cdt::try_from_pslg(pslg) {
        Ok(cdt) => Some(cdt),
        Err(e) => {
            tracing::warn!(
                "corefine_face: PSLG validation failed after resolve_crossings: {e}. \
                 Returning original face (no CDT refinement applied)."
            );
            None
        }
    }
}

/// Lifts interior CDT triangles back to 3-D face data while preserving face winding.
pub(super) fn assemble_cdt_faces(
    cdt: &Cdt,
    face: &FaceData,
    face_n: Vector3r,
    pool: &VertexPool,
    pslg_to_vid: &[VertexId],
) -> Vec<FaceData> {
    let mut triangles: Vec<FaceData> = Vec::new();
    for (_, tri) in cdt.triangulation().interior_triangles() {
        let [pv0, pv1, pv2] = tri.vertices;
        if pv0.idx() >= pslg_to_vid.len()
            || pv1.idx() >= pslg_to_vid.len()
            || pv2.idx() >= pslg_to_vid.len()
        {
            continue;
        }

        let v0 = pslg_to_vid[pv0.idx()];
        let v1 = pslg_to_vid[pv1.idx()];
        let v2 = pslg_to_vid[pv2.idx()];
        if FaceData::untagged(v0, v1, v2).is_degenerate() {
            continue;
        }

        let p0 = *pool.position(v0);
        let p1 = *pool.position(v1);
        let p2 = *pool.position(v2);
        let tri_n = (p1 - p0).cross(p2 - p0);
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
