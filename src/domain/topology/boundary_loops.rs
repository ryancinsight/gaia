//! Closed-loop extraction from directed boundary edges.
//!
//! A mesh with holes describes each hole as a set of *directed* boundary edges:
//! every edge carries the winding the missing face would have had, so the rim of
//! a hole is a set of closed loops over vertex ids. This module recovers those
//! loops.
//!
//! It is a pure graph operation over [`VertexId`] pairs — no mesh, no positions,
//! no tolerance — which is why it lives in the domain layer rather than in
//! either of the two application callers that need it.
//!
//! ## Why it is not private to either caller
//!
//! Hole sealing (`application::watertight::seal`) and seam repair
//! (`application::csg::arrangement::stitch`) each carried a private copy of this
//! walk, and the copies had already diverged: the sealing copy bounded the
//! *walk* but not the loops it returned, while the seam-repair copy bounded
//! both. Every bound is a parameter here, so a caller states its own policy
//! instead of inheriting one by accident of which copy it was cut from.
//!
//! ## Traversal
//!
//! Greedy DFS from the lowest unused vertex id, visiting successors in ascending
//! order, so the result does not depend on input order. Each directed edge is
//! consumed at most once, so a rim that revisits an edge cannot spin.
//!
//! A *figure-8* rim — one that touches itself at a vertex rather than closing —
//! would otherwise be returned as a single self-intersecting loop that no fill
//! can triangulate. The walk detects re-entry and cuts back to the repeated
//! vertex, emitting the lobe it just completed as its own loop.
//!
//! ## Storage
//!
//! The directed edges are sorted and deduplicated once, which makes them a CSR
//! successor table: the rows are the runs of equal `from` vertex, and a row is
//! found by binary search over the distinct `from` vertices. Every step of the
//! walk consumes the *lowest* unused successor of its vertex, so the consumed
//! edges of a row are always a prefix of that row, and one cursor per row
//! replaces a hash set of used edges. Loops are appended to one value buffer
//! with an offset table ([`PackedRows`]) instead of one `Vec` per loop.

use crate::domain::core::index::VertexId;
use crate::domain::topology::PackedRows;

/// Trace closed loops from directed boundary edges.
///
/// `boundary` holds `(from, to)` pairs; each distinct pair is consumed at most
/// once.
///
/// - `max_path_len` bounds the vertices visited while walking a *single* loop
///   before that walk is abandoned, so a malformed adjacency cannot spin.
/// - `max_loop_len` bounds the length of a loop that is *returned*; a longer one
///   is dropped. Pass [`usize::MAX`] to return whatever the walk closed.
///
/// Only loops of at least three vertices are returned: a shorter cycle encloses
/// no area and cannot be filled.
///
/// # Example
///
/// ```rust,ignore
/// // trace_loops is pub(crate) — used internally; example shown for illustration.
/// let rim = [(v0, v1), (v1, v2), (v2, v0)];
/// let loops = trace_loops(&rim, 4096, usize::MAX);
/// assert_eq!(loops.len(), 1);
/// assert_eq!(loops[0].len(), 3);
/// ```
#[must_use]
pub(crate) fn trace_loops(
    boundary: &[(VertexId, VertexId)],
    max_path_len: usize,
    max_loop_len: usize,
) -> PackedRows<VertexId> {
    let mut edges = boundary.to_vec();
    edges.sort_unstable();
    edges.dedup();

    // Row `r` is the successor run of `froms[r]`: `edges[row_start[r]..row_start[r + 1]]`.
    let mut froms: Vec<VertexId> = Vec::new();
    let mut row_start: Vec<usize> = Vec::new();
    for (index, &(from, _)) in edges.iter().enumerate() {
        if froms.last() != Some(&from) {
            froms.push(from);
            row_start.push(index);
        }
    }
    row_start.push(edges.len());
    // `cursor[r]` is the first unconsumed edge of row `r`.
    let mut cursor: Vec<usize> = row_start[..froms.len()].to_vec();

    // Consume the lowest unused successor of `vertex`, if any remain.
    let mut take_successor = |vertex: VertexId| -> Option<VertexId> {
        let row = froms.binary_search(&vertex).ok()?;
        let edge = cursor[row];
        (edge < row_start[row + 1]).then(|| {
            cursor[row] = edge + 1;
            edges[edge].1
        })
    };

    let mut loop_vertices: Vec<VertexId> = Vec::new();
    let mut loop_offsets = vec![0usize];
    let mut emit = |lobe: &[VertexId]| {
        if lobe.len() >= 3 && lobe.len() <= max_loop_len {
            loop_vertices.extend_from_slice(lobe);
            loop_offsets.push(loop_vertices.len());
        }
    };

    let mut path: Vec<VertexId> = Vec::new();
    for &start in &froms {
        while let Some(first_next) = take_successor(start) {
            path.clear();
            path.extend([start, first_next]);
            let mut cur = first_next;
            let mut closed = false;

            while path.len() <= max_path_len {
                let Some(next) = take_successor(cur) else {
                    break;
                };
                if next == start {
                    closed = true;
                    break;
                }
                // Figure-8: the walk has re-entered a vertex it already
                // holds, so the lobe between the two visits is itself a
                // closed loop. Emit it and cut the walk back to the
                // repeated vertex.
                if let Some(reentry) = path.iter().position(|&v| v == next) {
                    emit(&path[reentry..]);
                    path.truncate(reentry + 1);
                } else {
                    path.push(next);
                }
                cur = next;
            }

            if closed {
                emit(&path);
            }
        }
    }

    PackedRows::from_parts(loop_offsets, loop_vertices)
}

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: u32) -> VertexId {
        VertexId::new(n)
    }

    /// Loops as sorted vertex lists, sorted between loops, so a test can assert
    /// on the *set* of loops without depending on traversal order.
    fn normalised(loops: &PackedRows<VertexId>) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = loops
            .iter()
            .map(|lobe| {
                let mut ids: Vec<u32> = lobe.iter().map(|id| id.raw()).collect();
                ids.sort_unstable();
                ids
            })
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn triangle_rim_is_one_loop() {
        let rim = [(v(0), v(1)), (v(1), v(2)), (v(2), v(0))];
        let loops = trace_loops(&rim, 4096, usize::MAX);
        assert_eq!(normalised(&loops), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn disjoint_rims_are_separate_loops() {
        let rim = [
            (v(0), v(1)),
            (v(1), v(2)),
            (v(2), v(0)),
            (v(3), v(4)),
            (v(4), v(5)),
            (v(5), v(3)),
        ];
        let loops = trace_loops(&rim, 4096, usize::MAX);
        assert_eq!(normalised(&loops), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    /// A rim that touches itself at a vertex must be split, or the result is a
    /// single self-intersecting loop no fill can triangulate.
    #[test]
    fn figure_eight_rim_is_split_into_lobes() {
        let rim = [
            (v(0), v(1)),
            (v(1), v(2)),
            (v(2), v(3)),
            (v(3), v(4)),
            (v(4), v(2)), // re-enters 2, closing the 2-3-4 lobe
            (v(2), v(0)), // closes the 0-1-2 lobe
        ];
        let loops = trace_loops(&rim, 4096, usize::MAX);
        assert_eq!(normalised(&loops), vec![vec![0, 1, 2], vec![2, 3, 4]]);
    }

    /// The bounds are the whole reason this is shared: each caller states its own
    /// policy instead of inheriting the copy it was cut from.
    #[test]
    fn max_loop_len_drops_loops_it_will_not_fill() {
        let quad = [(v(0), v(1)), (v(1), v(2)), (v(2), v(3)), (v(3), v(0))];
        assert_eq!(trace_loops(&quad, 4096, 3).len(), 0);
        assert_eq!(trace_loops(&quad, 4096, 4).len(), 1);
    }

    /// An unclosed chain is not a hole and must yield nothing.
    #[test]
    fn open_chain_yields_no_loop() {
        let chain = [(v(0), v(1)), (v(1), v(2)), (v(2), v(3))];
        assert!(trace_loops(&chain, 4096, usize::MAX).is_empty());
    }

    /// `max_path_len` abandons a walk that will not close in budget.
    #[test]
    fn max_path_len_abandons_a_long_walk() {
        let long: Vec<(VertexId, VertexId)> = (0..100).map(|i| (v(i), v(i + 1))).collect();
        assert!(trace_loops(&long, 10, usize::MAX).is_empty());
        // With no path bound the same chain still does not close, so it is
        // still not a loop — the bound only changes how far the walk gets.
        assert!(trace_loops(&long, usize::MAX, usize::MAX).is_empty());
    }

    /// Characterization of the exact rows, not just the loop set: the fill
    /// callers consume loops in this order and each loop in this winding. A
    /// re-entry lobe is emitted before the loop that encloses it, and a
    /// duplicated edge is walked once.
    #[test]
    fn rows_follow_walk_order_and_duplicates_walk_once() {
        let rim = [
            (v(0), v(1)),
            (v(1), v(2)),
            (v(2), v(3)),
            (v(2), v(3)),
            (v(3), v(1)), // re-enters 1, emitting the 1-2-3 lobe first
            (v(1), v(4)),
            (v(4), v(0)),
        ];
        let loops = trace_loops(&rim, 4096, usize::MAX);
        let rows: Vec<Vec<u32>> = loops
            .iter()
            .map(|row| row.iter().map(|id| id.raw()).collect())
            .collect();
        assert_eq!(rows, vec![vec![1, 2, 3], vec![0, 1, 4]]);
    }

    #[test]
    fn no_edges_yields_no_loops() {
        assert!(trace_loops(&[], 4096, usize::MAX).is_empty());
    }

    /// Output must not depend on hash iteration order, so the same rim given in
    /// a different order yields the same loops.
    #[test]
    fn result_is_independent_of_input_order() {
        let forward = [
            (v(0), v(1)),
            (v(1), v(2)),
            (v(2), v(3)),
            (v(3), v(4)),
            (v(4), v(2)),
            (v(2), v(0)),
        ];
        let mut reversed = forward;
        reversed.reverse();
        assert_eq!(
            normalised(&trace_loops(&forward, 4096, usize::MAX)),
            normalised(&trace_loops(&reversed, 4096, usize::MAX))
        );
    }
}
