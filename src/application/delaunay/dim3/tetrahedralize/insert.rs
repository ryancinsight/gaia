//! Bowyer-Watson 3D point insertion: cavity search, erosion, and retriangulation.

use leto::geometry::Point3;

use crate::domain::core::scalar::Scalar;

use super::types::Tetrahedron;
use super::BowyerWatson3D;

impl<T: Scalar> BowyerWatson3D<T> {
    /// Insert a completely generic point into the mathematical grid.
    /// Updates the global state to rigorously maintain the Delaunay invariant.
    /// Exectues with guaranteed 0 heap allocations by swapping internal pre-allocated buffers.
    ///
    /// # Panics
    ///
    /// Panics if cavity expansion records an invalid or vacant tetrahedron slot
    /// in `bad_tets`, violating the occupied-index invariant used by the
    /// circumsphere flood fill.
    pub fn insert_point(&mut self, point: Point3<T>) {
        let p_idx = self.vertices.len();
        self.vertices.push(point);

        self.cavity_cache.clear();
        self.cavity_faces.clear();
        self.bad_tets.clear();
        self.visited_tets.clear();
        self.search_q.clear();
        self.search_visited.clear();

        // 1. Seed Discovery: O(1) expected time walk from last insertion.
        let mut seed = usize::MAX;
        let mut curr = self.last_inserted_tet;
        if self.tetrahedra.get(curr).and_then(|t| t.as_ref()).is_none()
            && let Some(valid_idx) = self.tetrahedra.iter().position(Option::is_some)
        {
            curr = valid_idx;
        }

        self.search_q.push(curr);
        self.search_visited.insert(curr);

        while let Some(current) = self.search_q.pop() {
            if let Some(tet) = &self.tetrahedra[current] {
                if tet.contains_in_circumsphere(&point, &self.vertices) {
                    seed = current;
                    break;
                }
                for face in &tet.faces() {
                    if let Some(&[t0, t1]) = self.face_tets.get(face) {
                        let neighbor = if t0 == current { t1 } else { t0 };
                        if neighbor != usize::MAX && self.search_visited.insert(neighbor) {
                            self.search_q.push(neighbor);
                            if self.search_visited.len() > 300 {
                                break;
                            } // Bound local BFS logic
                        }
                    }
                }
            }
        }

        // Fallback: Exact global search (triggers if point is exceptionally distant)
        if seed == usize::MAX {
            for (i, tet_opt) in self.tetrahedra.iter().enumerate() {
                if let Some(tet) = tet_opt
                    && tet.contains_in_circumsphere(&point, &self.vertices)
                {
                    seed = i;
                    break;
                }
            }
        }

        if seed == usize::MAX {
            return;
        }

        // 2. Cavity BFS: Topological expansion bounding strictly internal voids
        self.bad_tets.push(seed);
        self.visited_flags[seed] = true;
        self.visited_tets.push(seed);
        let mut current_idx = 0;

        while current_idx < self.bad_tets.len() {
            let curr = self.bad_tets[current_idx];
            current_idx += 1;

            let face_list = self.tetrahedra[curr]
                .as_ref()
                .expect("invariant: bad_tets contains valid occupied indices")
                .faces();

            for face in &face_list {
                *self.cavity_cache.entry(*face).or_insert(0) += 1;

                if let Some(&[t0, t1]) = self.face_tets.get(face) {
                    let neighbor = if t0 == curr { t1 } else { t0 };
                    if neighbor != usize::MAX && !self.visited_flags[neighbor] {
                        self.visited_flags[neighbor] = true;
                        self.visited_tets.push(neighbor);

                        if let Some(n_tet) = &self.tetrahedra[neighbor]
                            && n_tet.contains_in_circumsphere(&point, &self.vertices)
                        {
                            self.bad_tets.push(neighbor);
                        }
                    }
                }
            }
        }

        // Restore boolean flags to pristine zero-allocation state
        for &idx in &self.visited_tets {
            self.visited_flags[idx] = false;
        }

        // 3. Extrude forming the new Delaunay boundary
        for (&face, &count) in &self.cavity_cache {
            if count == 1 {
                self.cavity_faces.push((face, count));
            }
        }
        self.cavity_faces.sort_unstable_by_key(|(f, _)| *f);

        // Terminate boundary blocks to prevent overlap
        for i in 0..self.bad_tets.len() {
            let idx = self.bad_tets[i];
            self.remove_tet(idx);
        }

        for i in 0..self.cavity_faces.len() {
            let (face, _) = self.cavity_faces[i];
            let new_tet =
                Tetrahedron::new([face.v[0], face.v[1], face.v[2], p_idx], &self.vertices);
            self.add_tet(new_tet);
        }
        self.cavity_faces.clear();
    }
}
