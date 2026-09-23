//! Outward orientation repair by manifold BFS from an extremal face.

use super::IndexedMesh;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::aabb::Aabb;
use crate::domain::topology::PackedRows;
use leto::geometry::Vector3;

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
    pub fn orient_outward(&mut self) {
        use crate::application::csg::arrangement::classify::classify_fragment_prepared;
        use crate::application::csg::arrangement::gwn::PreparedFace;
        use crate::application::csg::arrangement::tiebreaker::FragmentClass;
        use crate::domain::geometry::normal::triangle_normal;
        use std::collections::VecDeque;

        // Collect an owned copy of all face data so the immutable borrow ends
        // before the mutable `self.faces.iter_mut()` pass below.
        use crate::infrastructure::storage::face_store::FaceData;
        let face_list: Vec<FaceData> = self.faces.iter().copied().collect();
        let n_faces: usize = face_list.len();
        if n_faces == 0 {
            return;
        }

        // Per-face normals (None = degenerate) and centroid X.  Both are read
        // once here rather than re-derived per component: the seed search
        // below runs once per component, so per-face work placed inside it is
        // multiplied by the component count.
        let mut face_normals: Vec<Option<Vector3<T>>> = Vec::with_capacity(n_faces);
        let mut centroid_x: Vec<T> = Vec::with_capacity(n_faces);
        let third = <T as Scalar>::from_f64(3.0);
        for face in &face_list {
            let a = self.vertices.position(face.vertices[0]);
            let b = self.vertices.position(face.vertices[1]);
            let c = self.vertices.position(face.vertices[2]);
            face_normals.push(triangle_normal(a, b, c));
            centroid_x.push((a.x + b.x + c.x) / third);
        }

        // Seed order: the faces a BFS may legally start from, highest centroid
        // X first, ties broken by ascending face index.  This reproduces the
        // "unvisited non-degenerate face with the maximum centroid X" rule
        // exactly, but computes the order once — O(n log n) — instead of
        // rescanning every face for every component, which is O(components x
        // faces) and is what made a many-island mesh quadratic.
        //
        // `centroid_x > -inf` is the exact selectability predicate: the scan it
        // replaces began at `neg_infinity()`, so a face with a NaN or -infinite
        // centroid could never win it and must not be selectable here either.
        let neg_inf = <T as eunomia::RealField>::neg_infinity();
        let mut seed_order: Vec<u32> = (0..n_faces as u32)
            .filter(|&fi| {
                let fi = fi as usize;
                face_normals[fi].is_some() && centroid_x[fi] > neg_inf
            })
            .collect();
        seed_order.sort_unstable_by(|&a, &b| {
            let (xa, xb) = (centroid_x[a as usize], centroid_x[b as usize]);
            xb.partial_cmp(&xa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        // Undirected edge -> adjacent face indices.
        // In a valid 2-manifold boundary mesh, every edge is shared by exactly 2 faces.
        // We use this exact property to build a flawless traversal graph, rather than
        // overwriting directed half-edges which randomly collide on unoriented meshes.
        let half_edge_cap: usize = n_faces * 3;
        let mut edges: hashbrown::HashMap<(VertexId, VertexId), [usize; 2]> =
            hashbrown::HashMap::with_capacity(half_edge_cap);
        for (fi, face) in face_list.iter().enumerate() {
            let v = face.vertices;
            for k in 0..3 {
                let j = (k + 1) % 3;
                let mut va = v[k];
                let mut vb = v[j];
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

        // BFS orientation labels: Some(true) = outward, Some(false) = inward.
        let mut orientation: Vec<Option<bool>> = vec![None; n_faces];

        // Track connected-component membership for nesting detection.
        let mut component_id: Vec<usize> = vec![usize::MAX; n_faces];
        let mut component_seeds: Vec<usize> = Vec::new();
        let mut current_component: usize = 0;

        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n_faces);

        // Outer loop handles disconnected components — each gets its own seed.
        //
        // `seed_cursor` walks `seed_order` once for the whole traversal.  A
        // face's orientation is only ever set, never cleared, so an entry the
        // cursor has passed can never become selectable again and the walk is
        // amortised O(n) in total rather than O(n) per component.
        let mut seed_cursor = 0usize;
        loop {
            while seed_cursor < seed_order.len()
                && orientation[seed_order[seed_cursor] as usize].is_some()
            {
                seed_cursor += 1;
            }
            if seed_cursor >= seed_order.len() {
                break;
            }
            // The unvisited non-degenerate face with the maximum centroid X:
            // the highest such entry the cursor has not yet passed.
            let seed_fi = seed_order[seed_cursor] as usize;

            component_seeds.push(seed_fi);

            // Seed orientation: the outward normal of the extremal face must
            // have a non-negative X component.
            let seed_normal = face_normals[seed_fi].expect("invariant: seed_fi is the extremal face whose normal was computed in the face_normals pass");
            orientation[seed_fi] = Some(seed_normal.x >= <T as eunomia::NumericElement>::ZERO);
            component_id[seed_fi] = current_component;

            queue.push_back(seed_fi);

            while let Some(fi) = queue.pop_front() {
                let is_outward = orientation[fi]
                    .expect("invariant: orientation is set before fi is pushed into the BFS queue");
                let v = face_list[fi].vertices;
                for k in 0..3 {
                    let j = (k + 1) % 3;
                    let mut va = v[k];
                    let mut vb = v[j];
                    if va > vb {
                        std::mem::swap(&mut va, &mut vb);
                    }

                    if let Some(&[f0, f1]) = edges.get(&(va, vb)) {
                        let nfi = if f0 == fi { f1 } else { f0 };
                        if nfi != usize::MAX
                            && orientation[nfi].is_none()
                            && face_normals[nfi].is_some()
                        {
                            // Determine if neighbor's current winding correctly opposes ours
                            let nv = face_list[nfi].vertices;
                            let mut neighbor_is_reverse = false;
                            for nk in 0..3 {
                                let nj = (nk + 1) % 3;
                                if nv[nk] == v[j] && nv[nj] == v[k] {
                                    neighbor_is_reverse = true;
                                    break;
                                }
                            }

                            // If the neighbor already opposes our edge, it shares our orientation state.
                            // If it aligns (flows identically), we must flip it to maintain manifold parity.
                            let next_outward = if neighbor_is_reverse {
                                is_outward
                            } else {
                                !is_outward
                            };

                            orientation[nfi] = Some(next_outward);
                            component_id[nfi] = current_component;
                            queue.push_back(nfi);
                        }
                    }
                }
            }

            current_component += 1;
        }

        // Nesting verdict per component, applied once at the end.  Carrying a
        // flag is what lets the nesting loop below stop rewriting
        // `orientation` face-by-face, which was a second O(components x faces)
        // scan; a flag costs one byte per component.
        let mut flip_component: Vec<bool> = Vec::new();

        if current_component > 1 {
            let component_count = current_component;
            let mut component_face_counts = vec![0usize; component_count];
            for fi in 0..n_faces {
                let comp = component_id[fi];
                if comp != usize::MAX && face_normals[fi].is_some() {
                    component_face_counts[comp] += 1;
                }
            }
            // Per-component face grouping in the CSR shape `AdjacencyGraph`
            // already owns: one offset table plus one contiguous buffer rather
            // than one `Vec` per component.  The count pass above and the fill
            // pass below are a counting sort over `component_id`, which the BFS
            // has already populated.
            let (mut component_faces, mut face_cursors) =
                PackedRows::<u32>::from_counts(component_face_counts);
            let mut component_aabbs: Vec<Aabb<T>> = vec![Aabb::empty(); component_count];

            for (fi, face) in face_list.iter().enumerate() {
                let comp = component_id[fi];
                if comp == usize::MAX || face_normals[fi].is_none() {
                    continue;
                }

                // The AABB spans the same three vertices whichever way the
                // face is wound, so it is expanded from the face as read rather
                // than from a corrected copy.
                component_aabbs[comp].expand(self.vertices.position(face.vertices[0]));
                component_aabbs[comp].expand(self.vertices.position(face.vertices[1]));
                component_aabbs[comp].expand(self.vertices.position(face.vertices[2]));
                component_faces.write(&mut face_cursors, comp, fi as u32);
            }

            // Prepared geometry, laid out in the partition's own order, so one
            // component is a slice of this buffer taken at the partition's
            // offsets.  Winding is corrected here, which is the only place it
            // matters: `normal` is built from the corrected vertex order.
            let mut prepared: Vec<PreparedFace> =
                Vec::with_capacity(component_faces.values().len());
            for &fi in component_faces.values() {
                let fi = fi as usize;
                let v = face_list[fi].vertices;
                let (i0, i1, i2) = if orientation[fi] == Some(false) {
                    (v[0], v[2], v[1])
                } else {
                    (v[0], v[1], v[2])
                };
                let a = self.vertices.position(i0);
                let b = self.vertices.position(i1);
                let c = self.vertices.position(i2);
                let a = crate::domain::core::scalar::Point3r::new(
                    a.x.to_f64(),
                    a.y.to_f64(),
                    a.z.to_f64(),
                );
                let b = crate::domain::core::scalar::Point3r::new(
                    b.x.to_f64(),
                    b.y.to_f64(),
                    b.z.to_f64(),
                );
                let c = crate::domain::core::scalar::Point3r::new(
                    c.x.to_f64(),
                    c.y.to_f64(),
                    c.z.to_f64(),
                );
                let ab = b - a;
                let ac = c - a;
                let normal = ab.cross(ac);
                let centroid = crate::domain::core::scalar::Point3r::new(
                    (a.x + b.x + c.x) / 3.0,
                    (a.y + b.y + c.y) / 3.0,
                    (a.z + b.z + c.z) / 3.0,
                );
                prepared.push(PreparedFace {
                    a,
                    b,
                    c,
                    centroid,
                    normal,
                    area: 0.5 * normal.norm(),
                });
            }

            let component_offsets = component_faces.offsets();
            flip_component = vec![false; component_count];

            for comp in 0..component_count {
                let seed_fi = component_seeds[comp];
                let seed_face = face_list[seed_fi];
                let corrected_normal = match face_normals[seed_fi] {
                    Some(normal) if orientation[seed_fi] == Some(false) => -normal,
                    Some(normal) => normal,
                    None => continue,
                };

                let normal_len = corrected_normal.norm();
                if normal_len <= <T as eunomia::NumericElement>::ZERO {
                    continue;
                }

                let a = self.vertices.position(seed_face.vertices[0]);
                let b = self.vertices.position(seed_face.vertices[1]);
                let c = self.vertices.position(seed_face.vertices[2]);
                let centroid = leto::geometry::Point3::new(
                    (a.x + b.x + c.x) / <T as Scalar>::from_f64(3.0),
                    (a.y + b.y + c.y) / <T as Scalar>::from_f64(3.0),
                    (a.z + b.z + c.z) / <T as Scalar>::from_f64(3.0),
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
                    let (other_start, other_end) =
                        (component_offsets[other], component_offsets[other + 1]);
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
        }

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
