//! Flat vertex-to-face incidence chains for the topology welder.
//!
//! One slot per (vertex, face) incidence, chained per vertex: the counting
//! build keeps each vertex's run contiguous, and a merge splices the absorbed
//! vertex's whole chain onto the survivor's tail in constant time.

use crate::infrastructure::storage::face_store::FaceStore;

/// Sentinel for "no slot" / "no face" in the incidence chains.
const EMPTY: u32 = u32::MAX;

/// Flat vertex-to-face adjacency: one slot per (vertex, face) incidence,
/// chained per vertex.
///
/// The scattered predecessor (`Vec<Vec<u32>>`) cost one heap allocation per
/// vertex plus a linear uniqueness scan per absorbed face. Here the initial
/// build is a counting pass plus a vertex-major fill, so each vertex's run is
/// contiguous, and a merge splices the absorbed vertex's whole chain onto the
/// survivor's tail in constant time — `is_safe_to_merge`'s shared-face
/// condition guarantees the two chains are disjoint, so the splice needs no
/// uniqueness scan at all.
pub(super) struct VertexFaceIncidence {
    /// First incidence slot of each vertex, or [`EMPTY`].
    head: Vec<u32>,
    /// Last incidence slot of each vertex, or [`EMPTY`].
    tail: Vec<u32>,
    /// Face id carried by each slot.
    face: Vec<u32>,
    /// Next slot in the same vertex's chain, or [`EMPTY`].
    next: Vec<u32>,
}

impl VertexFaceIncidence {
    /// Build the adjacency and the per-face vertex snapshot from a
    /// `FaceStore`.
    ///
    /// Returns `(incidence, cur_face_verts)` where vertex `i`'s incident
    /// face ids read as one chain, and `cur_face_verts[f]` is the current
    /// (post-merge-update) triple of raw vertex IDs for face `f`. A face
    /// listing the same vertex twice (a degenerate corner) contributes one
    /// incidence, matching the scattered predecessor's build-time
    /// uniqueness.
    pub(super) fn build(n: usize, face_store: &FaceStore) -> (Self, Vec<[u32; 3]>) {
        let mut counts = vec![0_usize; n];
        for (_, face) in face_store.iter_enumerated() {
            let vertices = face.vertices;
            for (i, v) in vertices.iter().enumerate() {
                if !vertices[..i].contains(v) {
                    counts[usize::try_from(v.raw()).expect("vertex index fits usize")] += 1;
                }
            }
        }

        let total: usize = counts.iter().copied().sum();
        let mut head = vec![EMPTY; n];
        let mut tail = vec![EMPTY; n];
        let mut face = vec![0_u32; total];
        let mut next = vec![EMPTY; total];
        let mut start_of_run = 0_usize;
        for (v, count) in counts.iter().copied().enumerate() {
            if count != 0 {
                head[v] = u32::try_from(start_of_run).expect("incidence count fits u32");
                tail[v] =
                    u32::try_from(start_of_run + count - 1).expect("incidence count fits u32");
            }
            start_of_run += count;
        }

        // Fill each vertex's run in face-arrival order: the cursor walks the
        // preallocated run, so every vertex's incidences stay contiguous.
        let mut cursor = head.clone();
        for (f_id, face_data) in face_store.iter_enumerated() {
            let vertices = face_data.vertices;
            for (i, v) in vertices.iter().enumerate() {
                if vertices[..i].contains(v) {
                    continue;
                }
                let index = usize::try_from(v.raw()).expect("vertex index fits usize");
                let slot = usize::try_from(cursor[index]).expect("incidence slot fits usize");
                let run_end = usize::try_from(tail[index]).expect("incidence slot fits usize");
                face[slot] = f_id.0;
                next[slot] = if slot == run_end {
                    EMPTY
                } else {
                    u32::try_from(slot + 1).expect("incidence count fits u32")
                };
                cursor[index] = u32::try_from(slot + 1).expect("incidence count fits u32");
            }
        }
        debug_assert_eq!(start_of_run, total);

        let mut cur_face_verts: Vec<[u32; 3]> = vec![[0; 3]; face_store.len()];
        for (f_id, face_data) in face_store.iter_enumerated() {
            cur_face_verts[usize::try_from(f_id.0).expect("face index fits usize")] = [
                face_data.vertices[0].raw(),
                face_data.vertices[1].raw(),
                face_data.vertices[2].raw(),
            ];
        }
        (
            Self {
                head,
                tail,
                face,
                next,
            },
            cur_face_verts,
        )
    }

    /// Whether `vertex` currently carries no incident face.
    pub(super) fn is_empty(&self, vertex: u32) -> bool {
        self.head[usize::try_from(vertex).expect("vertex index fits usize")] == EMPTY
    }

    /// Number of vertices carrying at least one incident face.
    pub(super) fn active_count(&self) -> usize {
        self.head.iter().filter(|&&h| h != EMPTY).count()
    }

    /// The incident face ids of `vertex`, oldest first.
    pub(super) fn faces(&self, vertex: u32) -> FaceChain<'_> {
        FaceChain {
            incidence: self,
            slot: self.head[usize::try_from(vertex).expect("vertex index fits usize")],
        }
    }

    /// Splice every incidence of `src` onto the tail of `dst`, leaving `src`
    /// empty. Constant time: the caller has already established (via the
    /// shared-face condition) that the two chains are disjoint, so no
    /// uniqueness scan is needed.
    pub(super) fn absorb(&mut self, dst: u32, src: u32) {
        let dst = usize::try_from(dst).expect("vertex index fits usize");
        let src = usize::try_from(src).expect("vertex index fits usize");
        if self.head[src] == EMPTY {
            return;
        }
        if self.head[dst] == EMPTY {
            self.head[dst] = self.head[src];
        } else {
            self.next[usize::try_from(self.tail[dst]).expect("incidence slot fits usize")] =
                self.head[src];
        }
        self.tail[dst] = self.tail[src];
        self.head[src] = EMPTY;
        self.tail[src] = EMPTY;
    }
}

/// A vertex's incident-face chain, read oldest first.
pub(super) struct FaceChain<'a> {
    incidence: &'a VertexFaceIncidence,
    slot: u32,
}

impl Iterator for FaceChain<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.slot == EMPTY {
            return None;
        }

        let slot = usize::try_from(self.slot).expect("incidence slot fits usize");
        self.slot = self.incidence.next[slot];
        Some(self.incidence.face[slot])
    }
}
