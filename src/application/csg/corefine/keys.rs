//! Canonical keys and the global seam-vertex map shared across faces.

use super::{PointBits3, EDGE_EPS, WELD_TOL_SQ};
use crate::application::csg::intersect::SnapSegment;
use crate::domain::core::constants::DEGENERATE_NORMAL_REL_SQ;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Point3r, Real};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use hashbrown::HashMap;

fn point_bits3(p: &Point3r) -> PointBits3 {
    [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
}

/// Canonical exact key for an undirected 3-D snap segment.
///
/// Uses raw IEEE-754 bit patterns (no tolerance/quantization), so two segments
/// deduplicate iff both endpoint coordinates are bit-identical up to endpoint
/// ordering.
#[inline]
pub(super) fn canonical_segment_key(seg: &SnapSegment) -> (PointBits3, PointBits3) {
    let a = point_bits3(&seg.start);
    let b = point_bits3(&seg.end);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ── Seam Vertex Map ──────────────────────────────────────────────────────────

/// Canonical undirected edge key: `(min(u,v), max(u,v))`.
#[inline]
pub(super) fn canonical_edge_key(u: VertexId, v: VertexId) -> (VertexId, VertexId) {
    if u <= v {
        (u, v)
    } else {
        (v, u)
    }
}

/// Global map of pre-registered Steiner vertices on each mesh edge.
///
/// Key: canonical undirected edge `(min(u,v), max(u,v))`.
/// Value: sorted list of `(t_parameter, VertexId)` along the edge from `u→v`
///        (or `min→max`).  Every face sharing edge `(u,v)` receives the **exact
///        same** Steiner sequence, guaranteeing seam-consistent CDT
///        triangulation.
///
/// # Theorem — Seam Consistency
///
/// If all faces sharing edge `(u,v)` use the same ordered sequence of Steiner
/// `VertexId`s from this map, their CDT sub-triangulations produce identical
/// edge decompositions along `(u,v)`, eliminating non-manifold edges.
///
/// **Proof sketch.** The CDT boundary polygon for face F includes
/// `[Vᵢ, S₁, S₂, …, Sₖ, Vᵢ₊₁]` along edge `(u,v)`.  Since all faces
/// sharing `(u,v)` use identical `S₁…Sₖ` with identical `VertexId`s, the CDT
/// constraint edges `{Vᵢ,S₁},{S₁,S₂},…,{Sₖ,Vᵢ₊₁}` are identical across
/// all faces.  The CDT honours constraint edges exactly (Shewchuk predicates),
/// so adjacent faces produce matching triangulation along the seam. ∎
pub type SeamVertexMap = HashMap<(VertexId, VertexId), Vec<(Real, VertexId)>>;

/// Build a global [`SeamVertexMap`] from all faces and their snap-segments.
///
/// For every snap-segment endpoint that lies on a face edge `(Vᵢ,Vᵢ₊₁)`, the
/// endpoint is registered via `VertexPool::insert_or_weld` **once** and stored
/// under the canonical edge key.  Subsequent faces sharing that edge receive
/// the same `VertexId` by lookup rather than re-inserting (which could produce
/// a different ID due to differing face normals passed to `insert_or_weld`).
pub fn build_seam_vertex_map(
    faces: &[FaceData],
    snap_segments: &[Vec<SnapSegment>],
    pool: &mut VertexPool,
) -> SeamVertexMap {
    debug_assert_eq!(
        faces.len(),
        snap_segments.len(),
        "every face must have a snap-segment slot"
    );

    let non_empty_count = snap_segments.iter().filter(|s| !s.is_empty()).count();
    let mut map: SeamVertexMap = HashMap::with_capacity(non_empty_count * 2);

    for (fi, face) in faces.iter().enumerate() {
        let face_segs = &snap_segments[fi];
        if face_segs.is_empty() {
            continue;
        }

        let pts: [Point3r; 3] = [
            *pool.position(face.vertices[0]),
            *pool.position(face.vertices[1]),
            *pool.position(face.vertices[2]),
        ];
        let face_n = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        let edge1_sq = (pts[1] - pts[0]).norm_squared();
        let edge2_sq = (pts[2] - pts[0]).norm_squared();
        if face_n.norm_squared() < DEGENERATE_NORMAL_REL_SQ * edge1_sq * edge2_sq {
            continue;
        }
        let face_n_unit = face_n / face_n.norm();
        let max_edge_sq = edge1_sq.max(edge2_sq).max((pts[2] - pts[1]).norm_squared());

        for seg in face_segs {
            for &p3d in &[seg.start, seg.end] {
                // Check each face edge.
                for ei in 0..3_usize {
                    let va_id = face.vertices[ei];
                    let vb_id = face.vertices[(ei + 1) % 3];
                    let va = pts[ei];
                    let vb = pts[(ei + 1) % 3];
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

                    // Register under canonical edge key.
                    let key = canonical_edge_key(va_id, vb_id);
                    let vid = pool.insert_or_weld(p3d, face_n_unit);

                    // Compute t relative to the canonical direction
                    // (min→max).
                    let t_canon = if va_id <= vb_id { t } else { 1.0 - t };

                    let entry = map.entry(key).or_default();
                    if !entry.iter().any(|&(_, v)| v == vid) {
                        entry.push((t_canon, vid));
                    }
                    break; // endpoint matched this edge, done.
                }
            }
        }
    }

    // Sort each edge's Steiners by t-parameter for consistent boundary
    // polygon construction.
    for steiners in map.values_mut() {
        steiners.sort_by(|a, b| a.0.total_cmp(&b.0));
        steiners.dedup_by_key(|entry| entry.1);
    }

    map
}
