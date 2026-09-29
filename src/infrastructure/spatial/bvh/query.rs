//! Iterative BVH traversal using a fixed-size stack.
//!
//! Avoids recursion and heap allocation during queries.  Node AABBs are read
//! without a token (pure spatial culling data); connectivity is read through a
//! [`TokenAccess`] permit (branded structural data).

use super::node::{BvhNodeKind, MAX_STACK_DEPTH};
use crate::domain::geometry::aabb::Aabb;
use crate::infrastructure::permission::{PermissionedArena, TokenAccess};

/// Iterative AABB-overlap query over a flat-arena BVH.
///
/// This is the single traversal body shared by both public entry points on
/// [`super::BvhTree`] — the exclusive (`&GhostToken`) and shared
/// (`SharedGhostToken`) queries — with `access` resolving to
/// `PermissionedArena::get` or `::get_shared` through [`TokenAccess`].  It
/// traverses with a `[u32; MAX_STACK_DEPTH]` stack; exact per-primitive checks
/// at leaves eliminate false positives from conservative union AABBs.
///
/// # Arguments
///
/// - `node_aabbs`  — node bounding boxes (token-free, pure geometry)
/// - `node_kinds`  — node connectivity (requires `access`)
/// - `indices`     — permuted primitive index table
/// - `prim_aabbs`  — source primitive AABBs for exact leaf-level checks
/// - `query`       — the AABB to test against
/// - `access`      — branded permit matching `node_kinds`
/// - `out`         — accumulator; not cleared before appending
pub(super) fn query_overlapping_generic<'brand, A: TokenAccess<'brand>>(
    node_aabbs: &[Aabb],
    node_kinds: &PermissionedArena<'brand, BvhNodeKind>,
    indices: &[usize],
    prim_aabbs: &[Aabb],
    query: &Aabb,
    access: A,
    out: &mut Vec<usize>,
) {
    if node_aabbs.is_empty() {
        return;
    }

    let mut stack = [0u32; MAX_STACK_DEPTH];
    let mut top: usize = 0;
    stack[top] = 0;
    top += 1;

    while top > 0 {
        top -= 1;
        let node_idx = stack[top] as usize;

        // Token-free AABB cull — ends with no borrow held.
        if !node_aabbs[node_idx].intersects(query) {
            continue;
        }

        // `Copy` clone is a 12-byte register copy; the arena borrow ends
        // before we access `indices` or `prim_aabbs` below.
        let kind = *access.get(node_kinds, node_idx);

        match kind {
            BvhNodeKind::Leaf { start, end } => {
                for &prim_idx in indices.iter().take(end as usize).skip(start as usize) {
                    // Exact per-primitive check; eliminates false positives
                    // from the conservative union AABB of the leaf node.
                    if prim_aabbs[prim_idx].intersects(query) {
                        out.push(prim_idx);
                    }
                }
            }
            BvhNodeKind::Inner { left, right } => {
                // Push right first so left is popped first (DFS order).
                stack[top] = right;
                top += 1;
                stack[top] = left;
                top += 1;
            }
        }
    }
}
