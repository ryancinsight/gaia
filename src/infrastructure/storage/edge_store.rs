//! Edge storage with half-edge connectivity.
//!
//! Each edge is stored as a canonical `(min_vertex, max_vertex)` pair with
//! references to adjacent faces. This replaces csgrs's on-demand adjacency
//! rebuilding with a persistent, incrementally-maintained structure.

use hashbrown::HashMap;

use crate::domain::core::index::{EdgeId, FaceId, VertexId};
use crate::infrastructure::storage::face_store::FaceData;

/// Stack-allocated, inline representation for edge-adjacent faces.
///
/// Avoids heap allocations for boundary (valence 1) and manifold (valence 2) edges,
/// falling back to a boxed slice only for non-manifold edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdjacentFaces {
    /// No adjacent faces (e.g. during construction/clear).
    Zero,
    /// Shared by exactly 1 face (boundary edge).
    One(FaceId),
    /// Shared by exactly 2 faces (manifold edge).
    Two(FaceId, FaceId),
    /// Shared by >2 faces (non-manifold edge).
    Many(Box<[FaceId]>),
}

impl AdjacentFaces {
    /// Create an empty list of adjacent faces.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::Zero
    }

    /// Valence: number of adjacent faces.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
            Self::Many(slice) => slice.len(),
        }
    }

    /// Is it empty?
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Zero)
    }

    /// Push a face ID to the list of adjacent faces.
    pub fn push(&mut self, face: FaceId) {
        let current = std::mem::replace(self, Self::Zero);
        *self = match current {
            Self::Zero => Self::One(face),
            Self::One(f0) => Self::Two(f0, face),
            Self::Two(f0, f1) => {
                let v = vec![f0, f1, face];
                Self::Many(v.into_boxed_slice())
            }
            Self::Many(slice) => {
                let mut v = slice.into_vec();
                v.push(face);
                Self::Many(v.into_boxed_slice())
            }
        };
    }

    /// Iterate over adjacent face IDs.
    #[inline]
    pub fn iter(&self) -> AdjacentFacesIter<'_> {
        AdjacentFacesIter {
            faces: self,
            index: 0,
        }
    }
}

impl Default for AdjacentFaces {
    #[inline]
    fn default() -> Self {
        Self::Zero
    }
}

/// Iterator over `AdjacentFaces`.
pub struct AdjacentFacesIter<'a> {
    faces: &'a AdjacentFaces,
    index: usize,
}

impl<'a> Iterator for AdjacentFacesIter<'a> {
    type Item = &'a FaceId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.faces {
            AdjacentFaces::Zero => None,
            AdjacentFaces::One(f0) => {
                if self.index == 0 {
                    self.index += 1;
                    Some(f0)
                } else {
                    None
                }
            }
            AdjacentFaces::Two(f0, f1) => {
                if self.index == 0 {
                    self.index += 1;
                    Some(f0)
                } else if self.index == 1 {
                    self.index += 1;
                    Some(f1)
                } else {
                    None
                }
            }
            AdjacentFaces::Many(slice) => {
                if self.index < slice.len() {
                    let item = &slice[self.index];
                    self.index += 1;
                    Some(item)
                } else {
                    None
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.faces.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AdjacentFacesIter<'_> {}

impl<'a> IntoIterator for &'a AdjacentFaces {
    type Item = &'a FaceId;
    type IntoIter = AdjacentFacesIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl std::ops::Index<usize> for AdjacentFaces {
    type Output = FaceId;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            AdjacentFaces::Zero => panic!("index out of bounds: 0 for empty"),
            AdjacentFaces::One(f0) => {
                if index == 0 {
                    f0
                } else {
                    panic!("index out of bounds: {index} for length 1")
                }
            }
            AdjacentFaces::Two(f0, f1) => {
                if index == 0 {
                    f0
                } else if index == 1 {
                    f1
                } else {
                    panic!("index out of bounds: {index} for length 2")
                }
            }
            AdjacentFaces::Many(slice) => &slice[index],
        }
    }
}

/// Data stored per edge.
#[derive(Clone, Debug)]
pub struct EdgeData {
    /// The two endpoint vertex IDs (canonical: v0 < v1).
    pub vertices: (VertexId, VertexId),
    /// Faces sharing this edge (0 = boundary, 1 = boundary, 2 = manifold, >2 = non-manifold).
    pub faces: AdjacentFaces,
}

impl EdgeData {
    /// Is this a boundary edge (shared by exactly 1 face)?
    #[inline]
    #[must_use]
    pub fn is_boundary(&self) -> bool {
        self.faces.len() == 1
    }

    /// Is this a manifold interior edge (shared by exactly 2 faces)?
    #[inline]
    #[must_use]
    pub fn is_manifold(&self) -> bool {
        self.faces.len() == 2
    }

    /// Is this a non-manifold edge (shared by >2 faces)?
    #[inline]
    #[must_use]
    pub fn is_non_manifold(&self) -> bool {
        self.faces.len() > 2
    }

    /// Valence: number of adjacent faces.
    #[inline]
    #[must_use]
    pub fn valence(&self) -> usize {
        self.faces.len()
    }
}

/// Storage for edges, built from faces.
///
/// Edges are identified by their canonical vertex pair `(min, max)`.
#[derive(Clone)]
pub struct EdgeStore {
    /// Edge data indexed by `EdgeId`.
    edges: Vec<EdgeData>,
    /// Lookup: canonical vertex pair → edge ID.
    edge_map: HashMap<(VertexId, VertexId), EdgeId>,
}

impl EdgeStore {
    /// Create an empty edge store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            edge_map: HashMap::new(),
        }
    }

    /// Build the edge store from a slice of faces.
    ///
    /// This scans all face edges and constructs the edge adjacency in O(F)
    /// where F = number of faces.
    ///
    /// Capacity hint: for a closed manifold triangle mesh, E = 3F/2 by
    /// the Euler relation.  Pre-allocating both the `edges` vec and the
    /// `edge_map` hash avoids incremental rehashing during construction.
    #[must_use]
    pub fn from_faces(faces: &[(FaceId, &FaceData)]) -> Self {
        let cap = faces.len().saturating_mul(3) / 2;
        let mut store = Self {
            edges: Vec::with_capacity(cap),
            edge_map: HashMap::with_capacity(cap),
        };

        for &(face_id, face) in faces {
            for (a, b) in face.edges_canonical() {
                store.register_edge(a, b, face_id);
            }
        }

        store
    }

    /// Build from a face store directly.
    ///
    /// Drives the inner registration loop without an intermediate `Vec` allocation.
    #[must_use]
    pub fn from_face_store(
        face_store: &crate::infrastructure::storage::face_store::FaceStore,
    ) -> Self {
        let cap = face_store.len().saturating_mul(3) / 2;
        let mut store = Self {
            edges: Vec::with_capacity(cap),
            edge_map: HashMap::with_capacity(cap),
        };
        for (face_id, face) in face_store.iter_enumerated() {
            for (a, b) in face.edges_canonical() {
                store.register_edge(a, b, face_id);
            }
        }
        store
    }

    /// Register an edge between `a` and `b` as belonging to `face`.
    #[inline]
    fn register_edge(&mut self, a: VertexId, b: VertexId, face: FaceId) {
        let key = if a.0 <= b.0 { (a, b) } else { (b, a) };

        if let Some(&edge_id) = self.edge_map.get(&key) {
            self.edges[edge_id.as_usize()].faces.push(face);
        } else {
            let edge_id = EdgeId::from_usize(self.edges.len());
            let mut faces = AdjacentFaces::new();
            faces.push(face);
            self.edges.push(EdgeData {
                vertices: key,
                faces,
            });
            self.edge_map.insert(key, edge_id);
        }
    }

    /// Number of edges.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Is the store empty?
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Get edge data by ID.
    #[inline]
    #[must_use]
    pub fn get(&self, id: EdgeId) -> &EdgeData {
        &self.edges[id.as_usize()]
    }

    /// Look up an edge by its canonical vertex pair.
    #[inline]
    #[must_use]
    pub fn find_edge(&self, a: VertexId, b: VertexId) -> Option<EdgeId> {
        let key = if a.0 <= b.0 { (a, b) } else { (b, a) };
        self.edge_map.get(&key).copied()
    }

    /// Iterate over all edges.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &EdgeData> {
        self.edges.iter()
    }

    /// Iterate with IDs.
    #[inline]
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (EdgeId, &EdgeData)> {
        self.edges
            .iter()
            .enumerate()
            .map(|(i, e)| (EdgeId::from_usize(i), e))
    }

    /// All boundary edges (valence == 1).
    #[must_use]
    pub fn boundary_edges(&self) -> Vec<EdgeId> {
        self.boundary_edges_iter().collect()
    }

    /// Iterate over boundary edges without allocating a `Vec`.
    ///
    /// Prefer this over [`boundary_edges`] when only iteration is needed.
    ///
    /// [`boundary_edges`]: Self::boundary_edges
    pub fn boundary_edges_iter(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_boundary())
            .map(|(i, _)| EdgeId::from_usize(i))
    }

    /// All non-manifold edges (valence > 2).
    #[must_use]
    pub fn non_manifold_edges(&self) -> Vec<EdgeId> {
        self.non_manifold_edges_iter().collect()
    }

    /// Iterate over non-manifold edges without allocating a `Vec`.
    ///
    /// Prefer this over [`non_manifold_edges`] when only iteration is needed.
    ///
    /// [`non_manifold_edges`]: Self::non_manifold_edges
    pub fn non_manifold_edges_iter(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_non_manifold())
            .map(|(i, _)| EdgeId::from_usize(i))
    }

    /// Count boundary edges.
    #[must_use]
    pub fn boundary_edge_count(&self) -> usize {
        self.edges.iter().filter(|e| e.is_boundary()).count()
    }

    /// Count non-manifold edges.
    #[must_use]
    pub fn non_manifold_edge_count(&self) -> usize {
        self.edges.iter().filter(|e| e.is_non_manifold()).count()
    }

    /// Clear all edges.
    pub fn clear(&mut self) {
        self.edges.clear();
        self.edge_map.clear();
    }
}

impl Default for EdgeStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests_edge_store.rs"]
mod tests;
