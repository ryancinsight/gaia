//! 3-D mesh co-refinement via CDT (Constrained Delaunay Triangulation).
//!
//! ## Algorithm — 7-Step CDT Co-Refinement (per face)
//!
//! ```text
//! face + snap_segments
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 0 — Choose 2-D projection axes                     │
//!    │  dominant_normal_axes(n): drop max-magnitude axis         │
//!    │  Theorem: projected area ≥ true area / √3  (always       │
//!    │         well-conditioned for non-zero normal)             │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 1 — Classify snap endpoint positions               │
//!    │  • On edge → edge Steiner point (t parameter)            │
//!    │  • At corner → existing VertexId                         │
//!    │  • Interior → interior Steiner (HashSet dedup)           │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 2 — Interior crossing points                       │
//!    │  All pairs (si, sj): line-line intersection in face plane │
//!    │  O(s²) where s = |snap_segments|; s≤4 in practice        │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 3 — Early exit                                     │
//!    │  No edge Steiners + no interior → return [face]          │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 4 — Build ordered boundary polygon                 │
//!    │  3 corners + sorted Steiners per edge                    │
//!    │  capacity = 3 + Σ|edge_steiners[i]|  (exact)            │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 5 — Build PSLG (Planar Straight Line Graph)        │
//!    │  HashMap<VertexId, PslgVertexId> (O(face_v) not pool_v)  │
//!    │  Shatter boundary + constraint segments to sub-edges     │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 6 — resolve_crossings + Cdt::try_from_pslg        │
//!    │  Shewchuk exact predicates; fallback: return [face]      │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!    ┌────▼────────────────────────────────────────────────────┐
//!    │  Step 7 — Lift CDT triangles back to 3-D                │
//!    │  Exclude supertriangle vertices, flip to match face_n    │
//!    └────┬────────────────────────────────────────────────────┘
//!         │
//!  sub-triangles replacing original face
//! ```
//!
//! ## Theorem — Projection Correctness (Watertightness Invariant)
//!
//! Two 3-D `VertexId`s that refer to the same welded pool position project to
//! the *same* 2-D coordinate under the dominant-axis-drop projection, and
//! therefore receive the same PSLG index.  Because the CDT uses Shewchuk exact
//! predicates (no epsilon fallbacks), any two adjacent face patches that share
//! a seam edge will produce exactly the same set of CDT edges along that seam,
//! eliminating T-junctions and achieving topological watertightness. ∎
//!
//! ## Memory Note
//!
//! `vid_to_pslg` uses `HashMap<VertexId, PslgVertexId>` with capacity hint
//! `boundary_vids.len() + interior_vids.len()`.  The previous `Vec<Option<_>>`
//! of length `pool.len()` allocated O(pool_size) per call — up to 100 k entries
//! for pools of 2 000+ vertices.  The HashMap reduces per-call allocation from
//! O(pool_size) to O(face_vertex_count) (typically 3–12 entries).

use hashbrown::HashMap;

use super::intersect::SnapSegment;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Real;

use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::domain::core::constants::{COREFINE_EDGE_EPS, COREFINE_WELD_TOL_SQ};

/// Distance tolerance squared for edge Steiner projection.
///
/// Delegates to [`COREFINE_WELD_TOL_SQ`] from the SSOT constants module.
/// Widened from 1e-8 to 1e-6 to handle shallow-angle tangent junctions
/// (e.g. elbow-cylinder V-shape) where floating-point drift can exceed 1e-8.
const WELD_TOL_SQ: Real = COREFINE_WELD_TOL_SQ;
/// Exclude snap endpoints that fall exactly at a face corner.
///
/// Delegates to [`COREFINE_EDGE_EPS`] from the SSOT constants module.
const EDGE_EPS: Real = COREFINE_EDGE_EPS;

type PointBits3 = [u64; 3];

/// 1-D bounding interval of a projected segment.
#[derive(Clone, Copy, Debug)]
struct SegBounds {
    /// Minimum value along the projected U axis.
    u_min: f64,
    /// Maximum value along the projected U axis.
    u_max: f64,
    /// Minimum value along the projected V axis.
    v_min: f64,
    /// Maximum value along the projected V axis.
    v_max: f64,
}

/// Reusable allocations for face co-refinement, eliminating heap allocation churn.
pub(crate) struct CorefinerScratch {
    /// Deduplicated snap segments.
    dedup_snap_segments: Vec<SnapSegment>,
    /// Set of snap segment canonical keys already seen.
    seen_snap_segments: hashbrown::HashSet<(PointBits3, PointBits3)>,
    /// Array of vertex IDs of segment endpoints, indexed by segment.
    seg_vids: Vec<[Option<VertexId>; 2]>,
    /// List of interior Steiner vertex IDs.
    interior_vids: Vec<VertexId>,
    /// Set of interior Steiner vertex IDs.
    interior_vid_set: hashbrown::HashSet<VertexId>,
    /// Pre-computed 1-D bounding intervals of projected segments.
    seg_bounds: Vec<SegBounds>,
    /// Edge Steiner vertices along the 3 face edges, sorted by parameter.
    edge_steiners: [Vec<(Real, VertexId)>; 3],
    /// Ordered list of boundary vertex IDs.
    boundary_vids: Vec<VertexId>,
    /// Mapping from VertexId to PSLG vertex identifier.
    vid_to_pslg: HashMap<VertexId, PslgVertexId>,
    /// Inverse mapping from PSLG vertex identifier index to VertexId.
    pslg_to_vid: Vec<VertexId>,
    /// Unique 2D coordinates of PSLG vertices.
    unique_pts: Vec<[Real; 2]>,
    /// Planar edge keys in the PSLG.
    pslg_edges: Vec<(usize, usize)>,
    /// Points collected along the interior of a segment.
    on_edge: Vec<(Real, usize)>,
}

impl CorefinerScratch {
    /// Creates a new, empty `CorefinerScratch` with default capacities.
    pub(crate) fn new() -> Self {
        Self {
            dedup_snap_segments: Vec::new(),
            seen_snap_segments: hashbrown::HashSet::new(),
            seg_vids: Vec::new(),
            interior_vids: Vec::new(),
            interior_vid_set: hashbrown::HashSet::new(),
            seg_bounds: Vec::new(),
            edge_steiners: [Vec::new(), Vec::new(), Vec::new()],
            boundary_vids: Vec::new(),
            vid_to_pslg: HashMap::new(),
            pslg_to_vid: Vec::new(),
            unique_pts: Vec::new(),
            pslg_edges: Vec::new(),
            on_edge: Vec::new(),
        }
    }

    /// Clears all buffers inside the scratchpad while retaining their allocated capacities.
    fn clear(&mut self) {
        self.dedup_snap_segments.clear();
        self.seen_snap_segments.clear();
        self.seg_vids.clear();
        self.interior_vids.clear();
        self.interior_vid_set.clear();
        self.seg_bounds.clear();
        for es in &mut self.edge_steiners {
            es.clear();
        }
        self.boundary_vids.clear();
        self.vid_to_pslg.clear();
        self.pslg_to_vid.clear();
        self.unique_pts.clear();
        self.pslg_edges.clear();
        self.on_edge.clear();
    }
}

mod face;
mod geom;
mod keys;

pub(crate) use face::corefine_face;
pub use keys::{build_seam_vertex_map, SeamVertexMap};

#[cfg(test)]
mod tests;
