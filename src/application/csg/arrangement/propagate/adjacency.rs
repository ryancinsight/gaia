//! Edge → incident-face map used by both propagation passes.
//!
//! The two-inline/rest-heap representation keeps the overwhelmingly common
//! manifold case (an edge shared by two triangles) allocation-free while
//! still holding non-manifold edges with any number of incident faces.

/// Faces incident to one undirected edge, inline for the first two.
#[derive(Clone, Default)]
pub(super) struct AdjacentFaces {
    count: usize,
    inline: [usize; 2],
    heap: Option<Vec<usize>>,
}

impl AdjacentFaces {
    pub(super) fn push(&mut self, face_idx: usize) {
        if self.count < 2 {
            self.inline[self.count] = face_idx;
            self.count += 1;
        } else {
            let mut v = self.heap.take().unwrap_or_default();
            v.push(face_idx);
            self.heap = Some(v);
            self.count += 1;
        }
    }

    pub(super) fn iter(&self) -> AdjacentFacesIter<'_> {
        AdjacentFacesIter {
            adj: self,
            index: 0,
        }
    }
}

pub(super) struct AdjacentFacesIter<'a> {
    adj: &'a AdjacentFaces,
    index: usize,
}

impl<'a> Iterator for AdjacentFacesIter<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < 2 && self.index < self.adj.count {
            let item = &self.adj.inline[self.index];
            self.index += 1;
            Some(item)
        } else if self.index < self.adj.count {
            let heap_idx = self.index - 2;
            let heap_vec = self
                .adj
                .heap
                .as_ref()
                .expect("invariant: heap storage exists for the third adjacent face");
            let item = &heap_vec[heap_idx];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a AdjacentFaces {
    type Item = &'a usize;
    type IntoIter = AdjacentFacesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
