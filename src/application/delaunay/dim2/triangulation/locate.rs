//! Point location via Lawson's oriented walk.
//!
//! # Theorem — Lawson Walk Correctness
//!
//! **Statement**: Starting from any triangle in a Delaunay triangulation and
//! walking toward the query point by crossing edges where `orient_2d` indicates
//! the point lies on the opposite side, the walk reaches the containing
//! triangle in $O(\sqrt{n})$ expected steps for uniformly distributed points.
//!
//! **Proof sketch**: At each step, the signed area of the walk triangle with
//! respect to the query point strictly decreases.  Since the triangulation has
//! a finite number of triangles and the walk never revisits a triangle, it
//! terminates.  The $O(\sqrt{n})$ bound follows from a geometric argument
//! showing that the walk crosses at most $O(\sqrt{n})$ Delaunay edges on
//! average for random point distributions (Devroye, Lemaire, Moreau 2004).
//!
//! # Cycle Detection
//!
//! In pathological geometries (near-degenerate configurations, floating-point
//! perturbations) the oriented walk can cycle between 2–3 triangles.  We track
//! visited triangles via a dense `Vec<bool>` bitflag array indexed by
//! `TriangleId`, giving $O(1)$ lookup/insert with no hashing overhead.
//! The safety bound `max_steps = 3 * T` is retained as a hard cap.
//!
//! For the hot path (per-vertex insertion), use
//! `DelaunayTriangulation::locate_point` which amortises allocation via an
//! epoch-stamped buffer.
//!
//! # Algorithm
//!
//! ```text
//! LOCATE(q, start_triangle):
//!   t ← start_triangle
//!   visited ← ∅
//!   LOOP:
//!     if t ∈ visited: RETURN NONE      // cycle detected
//!     visited ← visited ∪ {t}
//!     for each edge i of t:
//!       let (a, b) = edge vertices of t.edge[i]
//!       if orient_2d(a, b, q) < 0:  // q is on the wrong side
//!         t ← t.adj[i]
//!         CONTINUE LOOP
//!     RETURN t  // q is inside t (or on its boundary)
//! ```

use crate::domain::core::scalar::Real;
use crate::domain::geometry::predicates::{orient_2d, Orientation};
use leto::geometry::Point2;

use super::triangle::{Triangle, TriangleId, GHOST_TRIANGLE};
use crate::application::delaunay::dim2::pslg::vertex::PslgVertex;

/// Result of a point location query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    /// The query point lies strictly inside triangle `tid`.
    Inside(TriangleId),
    /// The query point lies on edge `edge` of triangle `tid`.
    OnEdge(TriangleId, usize),
    /// The query point coincides with vertex at local index `vi` of triangle `tid`.
    OnVertex(TriangleId, usize),
}

enum WalkStep {
    Cross(TriangleId),
    Found(Location),
}

#[inline]
fn classify_triangle_location(
    vertices: &[PslgVertex],
    tri: &Triangle,
    tid: TriangleId,
    q: &Point2<Real>,
) -> WalkStep {
    let v0 = &vertices[tri.vertices[0].idx()];
    let v1 = &vertices[tri.vertices[1].idx()];
    let v2 = &vertices[tri.vertices[2].idx()];

    let p0 = Point2::new(v0.x, v0.y);
    let p1 = Point2::new(v1.x, v1.y);
    let p2 = Point2::new(v2.x, v2.y);

    let o0 = orient_2d(&p1, &p2, q);
    if o0 == Orientation::Negative {
        return WalkStep::Cross(tri.adj[0]);
    }

    let o1 = orient_2d(&p2, &p0, q);
    if o1 == Orientation::Negative {
        return WalkStep::Cross(tri.adj[1]);
    }

    let o2 = orient_2d(&p0, &p1, q);
    if o2 == Orientation::Negative {
        return WalkStep::Cross(tri.adj[2]);
    }

    if o1 == Orientation::Degenerate && o2 == Orientation::Degenerate {
        return WalkStep::Found(Location::OnVertex(tid, 0));
    }
    if o0 == Orientation::Degenerate && o2 == Orientation::Degenerate {
        return WalkStep::Found(Location::OnVertex(tid, 1));
    }
    if o0 == Orientation::Degenerate && o1 == Orientation::Degenerate {
        return WalkStep::Found(Location::OnVertex(tid, 2));
    }

    if o0 == Orientation::Degenerate {
        return WalkStep::Found(Location::OnEdge(tid, 0));
    }
    if o1 == Orientation::Degenerate {
        return WalkStep::Found(Location::OnEdge(tid, 1));
    }
    if o2 == Orientation::Degenerate {
        return WalkStep::Found(Location::OnEdge(tid, 2));
    }

    WalkStep::Found(Location::Inside(tid))
}

pub(in crate::application::delaunay::dim2::triangulation) fn locate_with_visited<F>(
    vertices: &[PslgVertex],
    triangles: &[Triangle],
    start: TriangleId,
    qx: Real,
    qy: Real,
    mut mark_visited: F,
) -> Option<Location>
where
    F: FnMut(usize) -> bool,
{
    let q = Point2::new(qx, qy);
    let mut tid = start;
    let max_steps = triangles.len() * 3;

    for _ in 0..max_steps {
        if tid == GHOST_TRIANGLE {
            return None;
        }
        let idx = tid.idx();
        if idx >= triangles.len() || mark_visited(idx) {
            return None;
        }
        let tri = &triangles[idx];
        if !tri.alive {
            return None;
        }

        match classify_triangle_location(vertices, tri, tid, &q) {
            WalkStep::Cross(next) => tid = next,
            WalkStep::Found(location) => return Some(location),
        }
    }

    None
}

/// Locate the triangle containing point `(qx, qy)` using Lawson's oriented walk.
///
/// # Arguments
///
/// - `vertices`: the vertex pool
/// - `triangles`: the triangle pool (may contain tombstoned entries)
/// - `start`: the triangle to begin the walk from
/// - `qx`, `qy`: query point coordinates
///
/// # Returns
///
/// A [`Location`] describing where the query point lies relative to the
/// triangulation.  Returns `None` if the point is outside the convex hull
/// (walked off the boundary).
#[must_use]
pub fn locate(
    vertices: &[PslgVertex],
    triangles: &[Triangle],
    start: TriangleId,
    qx: Real,
    qy: Real,
) -> Option<Location> {
    let mut visited = vec![false; triangles.len()];
    locate_with_visited(vertices, triangles, start, qx, qy, |idx| {
        let was_visited = visited[idx];
        visited[idx] = true;
        was_visited
    })
}
