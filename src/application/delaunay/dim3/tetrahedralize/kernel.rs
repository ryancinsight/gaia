//! Bowyer-Watson 3D kernel: construction, pool management, finalization.

use hashbrown::{HashMap, HashSet};
use leto::geometry::{Point3, Vector3};

use crate::domain::core::scalar::Scalar;

use super::types::Tetrahedron;
use super::BowyerWatson3D;

impl<T: Scalar> BowyerWatson3D<T> {
    /// Initialize the state engine with an AABB bounding domain.
    pub fn new(min_bound: Point3<T>, max_bound: Point3<T>) -> Self {
        let mut engine = Self {
            vertices: Vec::new(),
            tetrahedra: Vec::new(),
            free_list: Vec::new(),
            face_tets: HashMap::with_capacity(1024),
            last_inserted_tet: 0,
            cavity_cache: HashMap::with_capacity(1024),
            cavity_faces: Vec::with_capacity(1024),
            bad_tets: Vec::with_capacity(1024),
            visited_tets: Vec::with_capacity(1024),
            visited_flags: Vec::new(),
            search_q: Vec::with_capacity(256),
            search_visited: HashSet::with_capacity(256),
            super_idx: 0,
        };
        engine.inject_super_tetrahedron(min_bound, max_bound);
        engine
    }

    /// Initialize the state engine with pre-allocated storage based on an a-priori
    /// geometric point count heuristic.
    pub fn with_capacity(
        min_bound: Point3<T>,
        max_bound: Point3<T>,
        point_capacity: usize,
    ) -> Self {
        let mut engine = Self {
            vertices: Vec::with_capacity(point_capacity + 4),
            tetrahedra: Vec::with_capacity(point_capacity * 6),
            free_list: Vec::with_capacity(1024),
            face_tets: HashMap::with_capacity(point_capacity * 12),
            last_inserted_tet: 0,
            cavity_cache: HashMap::with_capacity(1024),
            cavity_faces: Vec::with_capacity(1024),
            bad_tets: Vec::with_capacity(1024),
            visited_tets: Vec::with_capacity(1024),
            visited_flags: Vec::with_capacity(point_capacity * 6),
            search_q: Vec::with_capacity(256),
            search_visited: HashSet::with_capacity(256),
            super_idx: 0,
        };
        engine.inject_super_tetrahedron(min_bound, max_bound);
        engine
    }

    /// Construct a super-tetrahedron spanning the minimum bounding box.
    /// A minimum multiplier of 5.0 ensures boundary cavity retriangulations
    /// do not hit the degenerate corners.
    fn inject_super_tetrahedron(&mut self, min: Point3<T>, max: Point3<T>) {
        let d = max - min;
        let d_max = (d.x).max_scalar(d.y).max_scalar(d.z) * T::from_int(5);
        let center = min + d / T::from_int(2);

        let p0 = center
            + Vector3::new(
                <T as eunomia::NumericElement>::ZERO,
                d_max,
                -d_max / T::from_int(3),
            );
        let p1 = center
            + Vector3::new(
                d_max * (<T as Scalar>::from_f64(std::f64::consts::FRAC_PI_3)).sin(),
                -d_max / T::from_int(2),
                -d_max / T::from_int(3),
            );
        let p2 = center
            + Vector3::new(
                -d_max * (<T as Scalar>::from_f64(std::f64::consts::FRAC_PI_3)).sin(),
                -d_max / T::from_int(2),
                -d_max / T::from_int(3),
            );
        // The peak point goes upwards to enclose +Z, completing the regular tetrahedron mathematically
        let p3 = center
            + Vector3::new(
                <T as eunomia::NumericElement>::ZERO,
                <T as eunomia::NumericElement>::ZERO,
                d_max,
            );

        // Record the anchor index so we can delete super-vertices in O(1) time
        self.super_idx = self.vertices.len();
        self.vertices.push(p0);
        self.vertices.push(p1);
        self.vertices.push(p2);
        self.vertices.push(p3);

        let tet = Tetrahedron::new(
            [
                self.super_idx,
                self.super_idx + 1,
                self.super_idx + 2,
                self.super_idx + 3,
            ],
            &self.vertices,
        );
        self.add_tet(tet);
    }

    /// O(1) stable index registry for adjacency tracking.
    pub(super) fn add_tet(&mut self, tet: Tetrahedron<T>) -> usize {
        let idx = if let Some(i) = self.free_list.pop() {
            self.tetrahedra[i] = Some(tet);
            i
        } else {
            let i = self.tetrahedra.len();
            self.tetrahedra.push(Some(tet));
            if self.visited_flags.len() <= i {
                self.visited_flags.push(false);
            }
            i
        };

        for face in &self.tetrahedra[idx]
            .as_ref()
            .expect("invariant: just inserted tetrahedron")
            .faces()
        {
            let entry = self
                .face_tets
                .entry(*face)
                .or_insert([usize::MAX, usize::MAX]);
            if entry[0] == usize::MAX {
                entry[0] = idx;
            } else {
                entry[1] = idx;
            }
        }

        self.last_inserted_tet = idx;
        idx
    }

    /// Exact memory freeing mechanism for Delaunay retesselation.
    pub(super) fn remove_tet(&mut self, idx: usize) -> Tetrahedron<T> {
        let tet = self.tetrahedra[idx]
            .take()
            .expect("invariant: tetrahedron index to remove must be occupied");
        self.free_list.push(idx);

        for face in &tet.faces() {
            if let Some(entry) = self.face_tets.get_mut(face) {
                if entry[0] == idx {
                    entry[0] = usize::MAX;
                } else if entry[1] == idx {
                    entry[1] = usize::MAX;
                }

                if entry[0] == usize::MAX && entry[1] == usize::MAX {
                    self.face_tets.remove(face);
                } else if entry[0] == usize::MAX {
                    entry[0] = entry[1];
                    entry[1] = usize::MAX;
                }
            }
        }
        tet
    }

    /// Terminate and consolidate the finalized unstructured scalar mesh.
    ///
    /// Strips the 4 super-tetrahedron anchor vertices and remaps all retained
    /// tetrahedron indices into a contiguous vertex array. This prevents
    /// phantom super-vertices from contaminating downstream boundary extraction.
    #[must_use]
    pub fn finalize(self) -> (Vec<Point3<T>>, Vec<[usize; 4]>) {
        // Collect tets that do not reference any super-vertex.
        let mut raw_tets: Vec<[usize; 4]> = self
            .tetrahedra
            .into_iter()
            .flatten()
            .filter(|tet| !tet.shares_vertex_with_super(self.super_idx))
            .map(|t| t.v)
            .collect();

        // Determine which vertices are actually referenced by surviving tets.
        let n = self.vertices.len();
        let mut referenced = vec![false; n];
        for tet in &raw_tets {
            for &idx in tet {
                referenced[idx] = true;
            }
        }

        // Build old→new index remap, excluding exactly the 4 super-vertices
        // at indices [super_idx .. super_idx+4).
        let super_end = self.super_idx + 4;
        let mut old_to_new = vec![usize::MAX; n];
        let mut clean_points = Vec::with_capacity(n);
        for (old_idx, point) in self.vertices.into_iter().enumerate() {
            if old_idx >= self.super_idx && old_idx < super_end {
                continue; // Discard super-tetrahedron anchors
            }
            if referenced[old_idx] {
                old_to_new[old_idx] = clean_points.len();
                clean_points.push(point);
            }
        }

        // Remap tet indices.
        for tet in &mut raw_tets {
            for idx in tet.iter_mut() {
                *idx = old_to_new[*idx];
            }
        }

        (clean_points, raw_tets)
    }
}
