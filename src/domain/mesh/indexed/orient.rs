//! Outward orientation repair by manifold BFS from an extremal face.

use super::IndexedMesh;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::aabb::Aabb;
use crate::domain::topology::PackedRows;
use crate::infrastructure::storage::face_store::FaceData;
use hashbrown::HashMap;
use leto::geometry::Vector3;
use std::collections::VecDeque;

/// Compute per-face normals and X-centroid ordering keys for orientation repair.
fn face_geometry<T: Scalar>(
    mesh: &IndexedMesh<T>,
    face_list: &[FaceData],
) -> (Vec<Option<Vector3<T>>>, Vec<T>) {
    use crate::domain::geometry::normal::triangle_normal;

    let mut face_normals: Vec<Option<Vector3<T>>> = Vec::with_capacity(face_list.len());
    let mut centroid_x: Vec<T> = Vec::with_capacity(face_list.len());
    let third = T::from_int(3);
    for face in face_list {
        let a = mesh.vertices.position(face.vertices[0]);
        let b = mesh.vertices.position(face.vertices[1]);
        let c = mesh.vertices.position(face.vertices[2]);
        face_normals.push(triangle_normal(a, b, c));
        centroid_x.push((a.x + b.x + c.x) / third);
    }
    (face_normals, centroid_x)
}

/// Build the deterministic extremal-face seed order used across all components.
fn build_seed_order<T: Scalar>(
    face_normals: &[Option<Vector3<T>>],
    centroid_x: &[T],
) -> Vec<usize> {
    let neg_inf = <T as eunomia::RealField>::neg_infinity();
    let mut seed_order: Vec<usize> = (0..face_normals.len())
        .filter(|&fi| face_normals[fi].is_some() && centroid_x[fi] > neg_inf)
        .collect();
    seed_order.sort_unstable_by(|&a, &b| {
        let (xa, xb) = (centroid_x[a], centroid_x[b]);
        xb.partial_cmp(&xa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    seed_order
}

/// Build the undirected face adjacency map keyed by canonicalized edges.
fn build_undirected_edge_adjacency(
    face_list: &[FaceData],
) -> HashMap<(VertexId, VertexId), [usize; 2]> {
    let mut edges: HashMap<(VertexId, VertexId), [usize; 2]> =
        HashMap::with_capacity(face_list.len() * 3);
    for (fi, face) in face_list.iter().enumerate() {
        let vertices = face.vertices;
        for k in 0..3 {
            let j = (k + 1) % 3;
            let mut va = vertices[k];
            let mut vb = vertices[j];
            if va > vb {
                std::mem::swap(&mut va, &mut vb);
            }
            let entry = edges.entry((va, vb)).or_insert([usize::MAX, usize::MAX]);
            if entry[0] == usize::MAX {
                entry[0] = fi;
            } else {
                entry[1] = fi;
            }
        }
    }
    edges
}

/// Propagate a seed face orientation through one connected component by BFS.
#[expect(
    clippy::too_many_arguments,
    reason = "the BFS operates on multiple precomputed shared buffers so the main orientation repair can stay linear and allocation-free"
)]
fn bfs_orient_component<T: Scalar>(
    face_list: &[FaceData],
    face_normals: &[Option<Vector3<T>>],
    edges: &HashMap<(VertexId, VertexId), [usize; 2]>,
    seed_fi: usize,
    orientation: &mut [Option<bool>],
    component_id: &mut [usize],
    component_id_val: usize,
    queue: &mut VecDeque<usize>,
) {
    queue.push_back(seed_fi);
    while let Some(fi) = queue.pop_front() {
        let is_outward = orientation[fi]
            .expect("invariant: orientation is set before fi is pushed into the BFS queue");
        let vertices = face_list[fi].vertices;
        for k in 0..3 {
            let j = (k + 1) % 3;
            let mut va = vertices[k];
            let mut vb = vertices[j];
            if va > vb {
                std::mem::swap(&mut va, &mut vb);
            }

            let Some(&[f0, f1]) = edges.get(&(va, vb)) else {
                continue;
            };
            let neighbor = if f0 == fi { f1 } else { f0 };
            if neighbor == usize::MAX
                || orientation[neighbor].is_some()
                || face_normals[neighbor].is_none()
            {
                continue;
            }

            let neighbor_vertices = face_list[neighbor].vertices;
            let mut neighbor_is_reverse = false;
            for nk in 0..3 {
                let nj = (nk + 1) % 3;
                if neighbor_vertices[nk] == vertices[j] && neighbor_vertices[nj] == vertices[k] {
                    neighbor_is_reverse = true;
                    break;
                }
            }

            orientation[neighbor] = Some(if neighbor_is_reverse {
                is_outward
            } else {
                !is_outward
            });
            component_id[neighbor] = component_id_val;
            queue.push_back(neighbor);
        }
    }
}

/// Partition faces by connected component and accumulate their AABBs.
fn build_component_face_rows<T: Scalar>(
    mesh: &IndexedMesh<T>,
    face_list: &[FaceData],
    face_normals: &[Option<Vector3<T>>],
    component_id: &[usize],
    component_count: usize,
) -> (PackedRows<u32>, Vec<Aabb<T>>) {
    let mut component_face_counts = vec![0usize; component_count];
    for fi in 0..face_list.len() {
        let comp = component_id[fi];
        if comp != usize::MAX && face_normals[fi].is_some() {
            component_face_counts[comp] += 1;
        }
    }

    let (mut component_faces, mut face_cursors) =
        PackedRows::<u32>::from_counts(component_face_counts);
    let mut component_aabbs: Vec<Aabb<T>> = vec![Aabb::empty(); component_count];
    for (fi, face) in face_list.iter().enumerate() {
        let comp = component_id[fi];
        if comp == usize::MAX || face_normals[fi].is_none() {
            continue;
        }
        component_aabbs[comp].expand(mesh.vertices.position(face.vertices[0]));
        component_aabbs[comp].expand(mesh.vertices.position(face.vertices[1]));
        component_aabbs[comp].expand(mesh.vertices.position(face.vertices[2]));
        component_faces.write(
            &mut face_cursors,
            comp,
            u32::try_from(fi).expect("face index fits in u32"),
        );
    }
    (component_faces, component_aabbs)
}

/// Encode corrected face windings into prepared triangles for nesting tests.
fn build_prepared_component_faces<T: Scalar>(
    mesh: &IndexedMesh<T>,
    face_list: &[FaceData],
    orientation: &[Option<bool>],
    component_faces: &PackedRows<u32>,
) -> Vec<crate::application::csg::arrangement::gwn::PreparedFace> {
    use crate::application::csg::arrangement::gwn::PreparedFace;

    let mut prepared: Vec<PreparedFace> = Vec::with_capacity(component_faces.values().len());
    for &fi in component_faces.values() {
        let fi = usize::try_from(fi).expect("packed face index fits in usize");
        let vertices = face_list[fi].vertices;
        let (i0, i1, i2) = if orientation[fi] == Some(false) {
            (vertices[0], vertices[2], vertices[1])
        } else {
            (vertices[0], vertices[1], vertices[2])
        };
        let a = mesh.vertices.position(i0);
        let b = mesh.vertices.position(i1);
        let c = mesh.vertices.position(i2);
        let a = crate::domain::core::scalar::Point3r::new(a.x.to_f64(), a.y.to_f64(), a.z.to_f64());
        let b = crate::domain::core::scalar::Point3r::new(b.x.to_f64(), b.y.to_f64(), b.z.to_f64());
        let c = crate::domain::core::scalar::Point3r::new(c.x.to_f64(), c.y.to_f64(), c.z.to_f64());
        let ab = b - a;
        let ac = c - a;
        let normal = ab.cross(ac);
        prepared.push(PreparedFace {
            a,
            b,
            c,
            centroid: crate::domain::core::scalar::Point3r::new(
                (a.x + b.x + c.x) / 3.0,
                (a.y + b.y + c.y) / 3.0,
                (a.z + b.z + c.z) / 3.0,
            ),
            normal,
            area: 0.5 * normal.norm(),
        });
    }
    prepared
}

/// Detect components that must be flipped because they are nested inside another shell.
fn detect_nested_components<T: Scalar>(
    mesh: &IndexedMesh<T>,
    face_list: &[FaceData],
    face_normals: &[Option<Vector3<T>>],
    orientation: &[Option<bool>],
    component_id: &[usize],
    component_seeds: &[usize],
    component_count: usize,
) -> Vec<bool> {
    use crate::application::csg::arrangement::classify::classify_fragment_prepared;
    use crate::application::csg::arrangement::tiebreaker::FragmentClass;

    if component_count <= 1 {
        return Vec::new();
    }

    let (component_faces, component_aabbs) =
        build_component_face_rows(mesh, face_list, face_normals, component_id, component_count);
    let prepared = build_prepared_component_faces(mesh, face_list, orientation, &component_faces);
    let component_offsets = component_faces.offsets();
    let mut flip_component = vec![false; component_count];

    for comp in 0..component_count {
        let seed_fi = component_seeds[comp];
        let Some(seed_normal) = face_normals[seed_fi] else {
            continue;
        };
        let corrected_normal = if orientation[seed_fi] == Some(false) {
            -seed_normal
        } else {
            seed_normal
        };
        let normal_len = corrected_normal.norm();
        if normal_len <= <T as eunomia::NumericElement>::ZERO {
            continue;
        }

        let seed_face = face_list[seed_fi];
        let a = mesh.vertices.position(seed_face.vertices[0]);
        let b = mesh.vertices.position(seed_face.vertices[1]);
        let c = mesh.vertices.position(seed_face.vertices[2]);
        let centroid = leto::geometry::Point3::new(
            (a.x + b.x + c.x) / T::from_int(3),
            (a.y + b.y + c.y) / T::from_int(3),
            (a.z + b.z + c.z) / T::from_int(3),
        );

        let diag = (component_aabbs[comp].max - component_aabbs[comp].min)
            .norm()
            .to_f64();
        let probe_eps = (diag * 1.0e-6).max(T::tolerance().to_f64() * 10.0);
        let inward = corrected_normal / normal_len;
        let probe = crate::domain::core::scalar::Point3r::new(
            centroid.x.to_f64() - inward.x.to_f64() * probe_eps,
            centroid.y.to_f64() - inward.y.to_f64() * probe_eps,
            centroid.z.to_f64() - inward.z.to_f64() * probe_eps,
        );
        let probe_normal = crate::domain::core::scalar::Vector3r::new(
            inward.x.to_f64(),
            inward.y.to_f64(),
            inward.z.to_f64(),
        );

        let mut nesting_depth = 0usize;
        for other in 0..component_count {
            let (other_start, other_end) = (component_offsets[other], component_offsets[other + 1]);
            if other == comp || other_start == other_end {
                continue;
            }
            let contains_probe =
                component_aabbs[other].contains_point(&leto::geometry::Point3::new(
                    <T as Scalar>::from_f64(probe.x),
                    <T as Scalar>::from_f64(probe.y),
                    <T as Scalar>::from_f64(probe.z),
                ));
            if !contains_probe {
                continue;
            }
            if matches!(
                classify_fragment_prepared(
                    &probe,
                    &probe_normal,
                    &prepared[other_start..other_end]
                ),
                FragmentClass::Inside
            ) {
                nesting_depth += 1;
            }
        }

        if nesting_depth % 2 == 1 {
            flip_component[comp] = true;
        }
    }

    flip_component
}

impl<T: Scalar> IndexedMesh<T> {
    /// Repair any inconsistent face windings so that all normals point outward.
    ///
    /// Uses a manifold BFS flood from the extremal face to determine globally
    /// consistent outward orientation, then flips any inward-facing face's
    /// winding in-place (`v1 ↔ v2`). Finally calls [`Self::recompute_normals`] to
    /// synchronise vertex normals with the repaired geometry.
    ///
    /// ## Theorem basis
    ///
    /// For any closed orientable 2-manifold M embedded in ℝ³, the face with
    /// the vertex carrying the highest X coordinate must have an outward normal
    /// with `n_x ≥ 0` (Jordan-Brouwer separation theorem applied to the +X
    /// axis half-space). BFS propagation from this seed via the half-edge
    /// adjacency graph assigns a globally consistent orientation label to every
    /// reachable face. Flipping only the faces labelled *inward* corrects the
    /// minority misclassified by Phase-4 GWN seam ambiguity in the CSG
    /// pipeline (Turk & Levoy 1994; Zhou et al. 2016 "Mesh Arrangements for
    /// Solid Geometry") without disturbing the majority that are already correct.
    ///
    /// ## Properties
    ///
    /// - **Topology-preserving**: only vertex ordering within each face changes;
    ///   no vertices or edges are created, deleted, or repositioned.
    /// - **Watertightness-preserving**: the undirected edge graph is unchanged.
    /// - **O(F + E)** time and space (same as the half-edge BFS).
    /// - **No-op** on meshes already all-outward; safe to call unconditionally.
    ///
    /// ## Disconnected components
    ///
    /// Each connected component is re-seeded from its own extremal face, so
    /// multi-component meshes (e.g. a chip body with separate channel voids)
    /// are handled correctly.
    ///
    /// ## Nested-shell correction (Jordan–Brouwer nesting)
    ///
    /// After the BFS phase, a ray-casting parity test detects nested shells
    /// (e.g., CSG difference cavities).  Interior shells at odd nesting depth
    /// have their orientation toggled so that their normals point inward,
    /// producing the correct signed-volume divergence-theorem integral:
    /// `V_total = V_outer − Σ V_cavities`.  This ensures that `orient_outward`
    /// is safe to call on CSG difference results that contain cavities.
    ///
    /// # Panics
    ///
    /// Panics if the extremal seed face selected for orientation repair does
    /// not have a cached normal after the face-normal prepass.
    pub fn orient_outward(&mut self) {
        let face_list: Vec<FaceData> = self.faces.iter().copied().collect();
        if face_list.is_empty() {
            return;
        }

        let (face_normals, centroid_x) = face_geometry(self, &face_list);
        let seed_order = build_seed_order(&face_normals, &centroid_x);
        let edges = build_undirected_edge_adjacency(&face_list);
        let mut orientation: Vec<Option<bool>> = vec![None; face_list.len()];
        let mut component_id: Vec<usize> = vec![usize::MAX; face_list.len()];
        let mut component_seeds: Vec<usize> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(face_list.len());
        let mut component_count = 0usize;
        let mut seed_cursor = 0usize;

        while seed_cursor < seed_order.len() {
            while seed_cursor < seed_order.len() && orientation[seed_order[seed_cursor]].is_some() {
                seed_cursor += 1;
            }
            if seed_cursor >= seed_order.len() {
                break;
            }
            let seed_fi = seed_order[seed_cursor];
            let seed_normal = face_normals[seed_fi].expect(
                "invariant: seed_fi is the extremal face whose normal was computed in the face_normals pass",
            );
            orientation[seed_fi] = Some(seed_normal.x >= <T as eunomia::NumericElement>::ZERO);
            component_id[seed_fi] = component_count;
            component_seeds.push(seed_fi);
            bfs_orient_component(
                &face_list,
                &face_normals,
                &edges,
                seed_fi,
                &mut orientation,
                &mut component_id,
                component_count,
                &mut queue,
            );
            component_count += 1;
        }

        let flip_component = detect_nested_components(
            self,
            &face_list,
            &face_normals,
            &orientation,
            &component_id,
            &component_seeds,
            component_count,
        );

        // Flip inward faces in-place (swap v1 ↔ v2).  The nesting verdict is
        // applied here, in one pass over the faces, instead of being written
        // back into `orientation` per component.
        for (fi, face) in self.faces.iter_mut().enumerate() {
            let comp = component_id[fi];
            let outward = if flip_component.get(comp).copied().unwrap_or(false) {
                orientation[fi].map(|state| !state)
            } else {
                orientation[fi]
            };
            if outward == Some(false) {
                face.flip();
            }
        }

        // ── Signed-volume verification ────────────────────────────────
        //
        // The max-centroid-X seed heuristic assumes the extreme face's
        // outward normal has non-negative X.  This fails for concave
        // geometries (e.g., N-ary Intersection/Difference producing small
        // pocket-like shapes).  The signed-volume test is the definitive
        // orientation check for a closed manifold: by the divergence
        // theorem, a correctly outward-oriented surface always encloses
        // positive signed volume.  If negative, flip every face.
        let signed_vol = crate::domain::geometry::measure::total_signed_volume(
            self.faces.iter_enumerated().map(|(_, face)| {
                (
                    self.vertices.position(face.vertices[0]),
                    self.vertices.position(face.vertices[1]),
                    self.vertices.position(face.vertices[2]),
                )
            }),
        );
        if signed_vol < <T as eunomia::NumericElement>::ZERO {
            for face in self.faces.iter_mut() {
                face.flip();
            }
        }

        self.edges = None;

        // Synchronise vertex normals with the repaired winding.
        self.recompute_normals();
    }
}
