//! Bowyer-Watson constructors: point-set builds and super-triangle setup.

use super::DelaunayTriangulation;
use crate::application::delaunay::dim2::pslg::vertex::{PslgVertex, PslgVertexId};
use crate::application::delaunay::dim2::triangulation::locate::Location;
use crate::application::delaunay::dim2::triangulation::ordering::hilbert_order;
use crate::application::delaunay::dim2::triangulation::triangle::{
    Triangle, TriangleId, GHOST_TRIANGLE,
};
use crate::domain::core::scalar::Real;

impl DelaunayTriangulation {
    /// Build a Delaunay triangulation from a set of 2-D points.
    ///
    /// The points are inserted incrementally via Bowyer-Watson.
    #[must_use]
    pub fn from_points(points: &[(Real, Real)]) -> Self {
        let vertices: Vec<PslgVertex> =
            points.iter().map(|&(x, y)| PslgVertex::new(x, y)).collect();
        Self::from_vertices(&vertices)
    }

    /// Build from an existing vertex array.
    ///
    /// Uses Hilbert-curve ordering (BRIO-inspired) for optimal O(n log n)
    /// insertion performance.  By sorting vertices along a space-filling
    /// curve, consecutive insertions are spatially adjacent, reducing the
    /// Lawson walk distance from O(√n) to near-O(1) per insertion.
    #[must_use]
    pub fn from_vertices(real_vertices: &[PslgVertex]) -> Self {
        let n = real_vertices.len();
        let mut dt = Self::init_with_super_triangle(real_vertices);

        if n == 0 {
            return dt;
        }

        // Compute Hilbert-curve insertion order.
        let order = hilbert_order(real_vertices);

        for &orig_idx in &order {
            let vid = PslgVertexId::from_usize(orig_idx);
            dt.insert_vertex(vid);
        }

        dt.num_real_vertices = n;
        dt
    }

    /// Initialize the triangulation with a super-triangle that encloses all
    /// input vertices.
    fn init_with_super_triangle(real_vertices: &[PslgVertex]) -> Self {
        // Compute bounding box and inflate it significantly.
        let (mut min_x, mut min_y) = (Real::INFINITY, Real::INFINITY);
        let (mut max_x, mut max_y) = (Real::NEG_INFINITY, Real::NEG_INFINITY);

        for v in real_vertices {
            if v.x < min_x {
                min_x = v.x;
            }
            if v.y < min_y {
                min_y = v.y;
            }
            if v.x > max_x {
                max_x = v.x;
            }
            if v.y > max_y {
                max_y = v.y;
            }
        }

        // Handle degenerate case (0 or 1 point).
        if !min_x.is_finite() {
            min_x = -1.0;
            min_y = -1.0;
            max_x = 1.0;
            max_y = 1.0;
        }

        let dx = (max_x - min_x).max(1e-10);
        let dy = (max_y - min_y).max(1e-10);
        let dmax = dx.max(dy);
        let cx = (min_x + max_x) * 0.5;
        let cy = (min_y + max_y) * 0.5;

        // Super-triangle vertices far outside the bounding box.
        let margin = 20.0 * dmax;
        let sv0 = PslgVertex::new(cx - margin, cy - margin);
        let sv1 = PslgVertex::new(cx + margin, cy - margin);
        let sv2 = PslgVertex::new(cx, cy + margin);

        let n = real_vertices.len();
        let sv0_id = PslgVertexId::from_usize(n);
        let sv1_id = PslgVertexId::from_usize(n + 1);
        let sv2_id = PslgVertexId::from_usize(n + 2);

        let mut vertices: Vec<PslgVertex> = Vec::with_capacity(n + 3);
        vertices.extend_from_slice(real_vertices);
        vertices.push(sv0);
        vertices.push(sv1);
        vertices.push(sv2);

        let super_tri = Triangle::new(sv0_id, sv1_id, sv2_id);
        let triangles = vec![super_tri];
        let super_tid = TriangleId::new(0);

        // Initialize vert_to_tri: super-triangle vertices point to t0,
        // real vertices start as GHOST (updated on insertion).
        let mut vert_to_tri = vec![GHOST_TRIANGLE; n + 3];
        vert_to_tri[sv0_id.idx()] = super_tid;
        vert_to_tri[sv1_id.idx()] = super_tid;
        vert_to_tri[sv2_id.idx()] = super_tid;

        Self {
            vertices,
            triangles,
            last_triangle: TriangleId::new(0),
            num_real_vertices: 0,
            super_verts: [sv0_id, sv1_id, sv2_id],
            vert_to_tri,
            locate_epoch: vec![0; 1],
            locate_gen: 0,
        }
    }

    /// Locate the triangle containing `(qx, qy)` via Lawson's oriented walk
    /// with epoch-based cycle detection.
    ///
    /// # Theorem — Epoch-Stamped Visited Tracking
    ///
    /// **Statement**: A per-triangle epoch vector `E[0..T]` and a global
    /// generation counter `g` provide amortised $O(1)$ visited-set
    /// operations with $O(T)$ memory allocated once, eliminating the
    /// per-call `HashSet` allocation incurred by the cold-path
    /// [`locate()`](crate::application::delaunay::dim2::triangulation::locate::locate) free function.
    ///
    /// **Proof**: Each call increments `g` and marks visited triangles by
    /// setting `E[tid] = g`.  A triangle is "visited in this call" iff
    /// `E[tid] == g`.  Since `g` is unique per call, all previous marks
    /// are implicitly invalidated without clearing `E`.  The `Vec` grows
    /// monotonically (amortised $O(1)$ push); no other allocation occurs.
    /// When `g` wraps around $2^{32}$ we reset `E` to zero — this happens
    /// once per ~4 billion locate calls and costs $O(T)$.  ∎
    pub(crate) fn locate_point(
        &mut self,
        start: TriangleId,
        qx: Real,
        qy: Real,
    ) -> Option<Location> {
        self.locate_gen = self.locate_gen.wrapping_add(1);
        if self.locate_gen == 0 {
            // Epoch wrapped — clear all stamps.
            self.locate_epoch.fill(0);
            self.locate_gen = 1;
        }
        let r#gen = self.locate_gen;

        // Ensure epoch vec covers all triangles.
        if self.locate_epoch.len() < self.triangles.len() {
            self.locate_epoch.resize(self.triangles.len(), 0);
        }
        let vertices = &self.vertices;
        let triangles = &self.triangles;
        let locate_epoch = &mut self.locate_epoch;
        crate::application::delaunay::dim2::triangulation::locate::locate_with_visited(
            vertices,
            triangles,
            start,
            qx,
            qy,
            |idx| {
                let was_visited = locate_epoch[idx] == r#gen;
                locate_epoch[idx] = r#gen;
                was_visited
            },
        )
    }

    /// Insert a vertex into the triangulation via Bowyer-Watson.
    pub(crate) fn insert_vertex(&mut self, vid: PslgVertexId) {
        let v = self.vertices[vid.idx()];

        // 1. Locate the containing triangle (epoch-based, zero-alloc).
        let loc = if let Some(l) = self.locate_point(self.last_triangle, v.x, v.y) {
            l
        } else {
            // Fallback: linear scan for a valid starting triangle.
            let start = self
                .triangles
                .iter()
                .position(|t| t.alive)
                .map_or(TriangleId::new(0), TriangleId::from_usize);

            match self.locate_point(start, v.x, v.y) {
                Some(l) => l,
                None => return, // Point outside super-triangle — skip.
            }
        };

        match loc {
            Location::OnVertex(_, _) => {
                // Duplicate point — skip insertion.
            }
            Location::Inside(tid) => {
                self.insert_in_triangle(vid, tid);
            }
            Location::OnEdge(tid, edge) => {
                self.insert_on_edge(vid, tid, edge);
            }
        }
    }
}
