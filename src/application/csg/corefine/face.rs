//! The co-refinement pass: one face against its snap segments, via a 2-D CDT.

pub(super) mod face_ops;
use face_ops::{
    assemble_cdt_faces, build_face_cdt, build_face_pslg, collect_edge_steiners,
    collect_interior_crossings, dedup_and_sort_snap_segments, prepare_boundary_or_fallback,
};

use super::geom::dominant_normal_axes;
use super::{CorefinerScratch, SeamVertexMap};
use crate::application::csg::intersect::SnapSegment;
use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::domain::core::constants::{DEGENERATE_NORMAL_REL_SQ, DEGENERATE_SEGMENT_REL_SQ};
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use hashbrown::{HashMap, HashSet};

pub(super) type EdgeSteiner = (Real, VertexId);
pub(super) type EdgeSteinerLists = [Vec<EdgeSteiner>; 3];

pub(super) struct FaceProjection {
    pub(super) face_pts: [Point3r; 3],
    pub(super) face_n: Vector3r,
    pub(super) face_n_unit: Vector3r,
    pub(super) axis_u: usize,
    pub(super) axis_v: usize,
    pub(super) max_edge_sq: Real,
}

pub(super) struct InteriorVertexBuffers<'a> {
    pub(super) interior_vids: &'a mut Vec<VertexId>,
    pub(super) interior_vid_set: &'a mut HashSet<VertexId>,
}

pub(super) struct EdgeSteinerBuffers<'a> {
    pub(super) edge_steiners: &'a mut EdgeSteinerLists,
    pub(super) seg_vids: &'a mut Vec<[Option<VertexId>; 2]>,
    pub(super) interior: InteriorVertexBuffers<'a>,
}

pub(super) struct PslgBuffers<'a> {
    pub(super) vid_to_pslg: &'a mut HashMap<VertexId, PslgVertexId>,
    pub(super) pslg_to_vid: &'a mut Vec<VertexId>,
    pub(super) unique_pts: &'a mut Vec<[Real; 2]>,
    pub(super) pslg_edges: &'a mut Vec<(usize, usize)>,
    pub(super) on_edge: &'a mut Vec<(Real, usize)>,
}

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
    let projection = FaceProjection {
        face_pts: [a, b, c],
        face_n,
        face_n_unit,
        axis_u: 0,
        axis_v: 0,
        max_edge_sq: edge1_sq.max(edge2_sq).max((c - b).norm_squared()),
    };

    // Exact segment canonicalization:
    // remove degenerate and bit-identical duplicate constraints before O(s²)
    // crossing detection and PSLG resolve_crossings.
    // Scale-relative degenerate threshold: segments shorter than
    // sqrt(DEGENERATE_SEGMENT_REL_SQ) · max_edge are collapsed.
    let seg_degen_sq = DEGENERATE_SEGMENT_REL_SQ * projection.max_edge_sq;
    dedup_and_sort_snap_segments(
        snap_segments,
        seg_degen_sq,
        &mut scratch.dedup_snap_segments,
        &mut scratch.seen_snap_segments,
    );
    let dedup_snap_segments = &scratch.dedup_snap_segments;

    // ── Step 0: Choose projection axes ───────────────────────────────────────
    let (axis_u, axis_v) = dominant_normal_axes(face_n_unit);
    let projection = FaceProjection {
        axis_u,
        axis_v,
        ..projection
    };

    // ── Step 1: Edge Steiner vertices ─────────────────────────────────────────
    //
    // When a global SeamVertexMap is available, edge Steiners are looked up from
    // the map (keyed by canonical edge pair) rather than recomputed from raw
    // snap-segment geometry.  This guarantees that adjacent faces sharing the
    // same edge receive the exact same Steiner VertexIds.
    {
        let mut buffers = EdgeSteinerBuffers {
            edge_steiners: &mut scratch.edge_steiners,
            seg_vids: &mut scratch.seg_vids,
            interior: InteriorVertexBuffers {
                interior_vids: &mut scratch.interior_vids,
                interior_vid_set: &mut scratch.interior_vid_set,
            },
        };
        collect_edge_steiners(
            face,
            dedup_snap_segments,
            pool,
            seam_map,
            &projection,
            &mut buffers,
        );
    }

    // ── Step 2: Interior crossing points ─────────────────────────────────────
    {
        let mut interior = InteriorVertexBuffers {
            interior_vids: &mut scratch.interior_vids,
            interior_vid_set: &mut scratch.interior_vid_set,
        };
        collect_interior_crossings(
            dedup_snap_segments,
            pool,
            &projection,
            &mut scratch.seg_bounds,
            &mut interior,
        );
    }

    // ── Dedup edge steiners before length checks ───────────────────────────────
    // Deferring deduplication avoids O(N^2) linear scan overhead on highly refined edges.
    for es in &mut scratch.edge_steiners {
        es.sort_by(|a, b| a.0.total_cmp(&b.0));
        es.dedup_by_key(|&mut (_, vid)| vid);
    }

    // ── Steps 3-4: Early exits and boundary preparation ──────────────────────
    if let Some(faces) = prepare_boundary_or_fallback(
        face,
        &scratch.edge_steiners,
        &scratch.interior_vids,
        &mut scratch.boundary_vids,
        pool,
        &projection,
    ) {
        return faces;
    }

    // ── Step 5: Build PSLG from 2-D projections ───────────────────────────────
    let mut pslg = {
        let mut buffers = PslgBuffers {
            vid_to_pslg: &mut scratch.vid_to_pslg,
            pslg_to_vid: &mut scratch.pslg_to_vid,
            unique_pts: &mut scratch.unique_pts,
            pslg_edges: &mut scratch.pslg_edges,
            on_edge: &mut scratch.on_edge,
        };
        build_face_pslg(
            &scratch.boundary_vids,
            &scratch.interior_vids,
            &scratch.seg_vids,
            pool,
            axis_u,
            axis_v,
            &mut buffers,
        )
    };

    // ── Step 6: Build CDT ────────────────────────────────────────────────────
    // Resolve any interior segment crossings that arise from 3-D seam curves
    // whose 2-D projections cross (e.g. out-of-plane multi-branch junctions).
    // `resolve_crossings` handles proper crossings, T-intersections, and
    // collinear overlaps when the active-precision construction is
    // representable. `try_from_pslg` validates the result and keeps this
    // failure path observable when a crossing cannot be constructed.
    let Some(cdt) = build_face_cdt(&mut pslg) else {
        return vec![*face];
    };

    // ── Step 7: Emit triangles lifted back to 3-D ─────────────────────────────
    assemble_cdt_faces(&cdt, face, projection.face_n, pool, &scratch.pslg_to_vid)
}
