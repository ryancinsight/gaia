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
//! order, so the result does not depend on hash iteration order. Each directed
//! edge is consumed at most once, so a rim that revisits an edge cannot spin.
//!
//! A *figure-8* rim — one that touches itself at a vertex rather than closing —
//! would otherwise be returned as a single self-intersecting loop that no fill
//! can triangulate. The walk detects re-entry and cuts back to the repeated
//! vertex, emitting the lobe it just completed as its own loop.

use hashbrown::{HashMap, HashSet};

use crate::domain::core::index::VertexId;

/// Trace closed loops from directed boundary edges.
///
/// `boundary` holds `(from, to)` pairs; each pair is consumed at most once.
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
) -> Vec<Vec<VertexId>> {
    // Successors per vertex. Capacity 2 is the manifold case (each rim vertex
    // has one incoming and one outgoing edge); a non-manifold vertex grows.
    let mut adj: HashMap<VertexId, Vec<VertexId>> = HashMap::with_capacity(boundary.len());
    for &(vi, vj) in boundary {
        adj.entry(vi)
            .or_insert_with(|| Vec::with_capacity(2))
            .push(vj);
    }
    // Ascending successors, plus ascending starts below, make the output
    // independent of hash iteration order.
    for successors in adj.values_mut() {
        successors.sort_unstable();
    }

    let mut used: HashSet<(VertexId, VertexId)> = HashSet::new();
    let mut loops: Vec<Vec<VertexId>> = Vec::new();
    let mut starts: Vec<VertexId> = adj.keys().copied().collect();
    starts.sort_unstable();

    for start in starts {
        let Some(first_successors) = adj.get(&start) else {
            continue;
        };
        // `adj` is only read below, so this borrow is held across the walk
        // rather than cloned.
        for &first_next in first_successors {
            if used.contains(&(start, first_next)) {
                continue;
            }
            let mut path: Vec<VertexId> = vec![start, first_next];
            used.insert((start, first_next));
            let mut cur = first_next;
            let mut closed = false;

            loop {
                if path.len() > max_path_len {
                    break;
                }
                let Some(successors) = adj.get(&cur) else {
                    break;
                };
                let mut advanced = false;
                for &next in successors {
                    if used.contains(&(cur, next)) {
                        continue;
                    }
                    used.insert((cur, next));
                    if next == start {
                        closed = true;
                        advanced = true;
                        break;
                    }
                    // Figure-8: the walk has re-entered a vertex it already
                    // holds, so the lobe between the two visits is itself a
                    // closed loop. Emit it and cut the walk back to the
                    // repeated vertex.
                    if let Some(reentry) = path.iter().position(|&v| v == next) {
                        let lobe = path[reentry..].to_vec();
                        if lobe.len() >= 3 && lobe.len() <= max_loop_len {
                            loops.push(lobe);
                        }
                        path.truncate(reentry + 1);
                        cur = next;
                        advanced = true;
                        break;
                    }
                    path.push(next);
                    cur = next;
                    advanced = true;
                    break;
                }
                if !advanced || closed {
                    break;
                }
            }

            if closed && path.len() >= 3 && path.len() <= max_loop_len {
                loops.push(path);
            }
        }
    }

    loops
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
    fn normalised(loops: &[Vec<VertexId>]) -> Vec<Vec<u32>> {
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
