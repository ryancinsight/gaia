//! Body-Centered Cubic (BCC) Lattice Seeding and SDF Volumetric Meshing.
//!
//! Generates an unstructured `IndexedMesh<T>` conforming to an implicit `Sdf3D` surface
//! using gradient descent and the robust-predicate `BowyerWatson3D`
//! tetrahedralizer.

use crate::application::delaunay::dim3::sdf::Sdf3D;
use crate::application::delaunay::dim3::tetrahedralize::BowyerWatson3D;
use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::indexed::IndexedMesh;
use hashbrown::{HashMap, HashSet};
use leto::geometry::{Point3, Vector3};

/// An implicit-to-explicit tetrahedral mesh generator.
pub struct SdfMesher<T: Scalar> {
    /// Nominal edge length for the internal BCC lattice.
    pub cell_size: T,
    /// Number of gradient descent steps for boundary projection.
    pub snap_iterations: usize,
    /// Distance threshold normalized to `cell_size` for points that undergo snapping.
    pub snap_radius: T,
}

impl<T: Scalar> SdfMesher<T> {
    /// Create a new volumetric mesher with a target characteristic element edge length.
    pub fn new(cell_size: T) -> Self {
        Self {
            cell_size,
            snap_iterations: 15,
            snap_radius: <T as Scalar>::from_f64(1.5),
        }
    }

    /// Embed the boundary geometry by evaluating the input `Sdf3D`, inserting
    /// conforming seed points, and tetrahedralizing into an `IndexedMesh<T>`.
    ///
    /// The generated interior uses the Delaunay empty-circumsphere criterion;
    /// that establishes connectivity, not a quality optimum. Boundary
    /// projection follows the `Sdf3D` implementation's gradient contract.
    ///
    /// # Panics
    ///
    /// Panics if the derived lattice dimensions or signed loop indices exceed
    /// the representable host integer ranges used for internal capacities and
    /// scalar conversion.
    pub fn build_volume<S: Sdf3D<T>>(&self, sdf: &S) -> IndexedMesh<T> {
        let (min, max) = sdf.bounds();
        let h = self.cell_size;
        let raw_points =
            generate_bcc_points(sdf, min, max, h, self.snap_iterations, self.snap_radius);
        let total_capacity = raw_points.len().max(16);
        let mut delaunay = BowyerWatson3D::with_capacity(min, max, total_capacity);
        let unique_points = weld_and_order_points(raw_points, h);

        for p in unique_points {
            delaunay.insert_point(p);
        }

        let (points, tetrahedra) = delaunay.finalize();
        let keep = carve_tetrahedra(sdf, &points, &tetrahedra, h);
        let mut mesh = assemble_tetrahedral_mesh(points, keep);
        let b_faces = mesh.boundary_faces();
        orient_boundary_faces(&mut mesh, &b_faces);
        relax_boundary_vertices(&mut mesh, sdf, &b_faces, h);

        tracing::debug!(
            "Delaunay generated {} tets out of {} final points",
            mesh.cell_count(),
            mesh.vertex_count()
        );
        mesh.recompute_normals();

        mesh
    }
}

/// Truncate a floating-point ceil/floor result after the caller has constrained its range.
#[expect(
    clippy::cast_possible_truncation,
    reason = "ceil/floor values are range-checked via the checked integer conversion immediately afterward"
)]
fn truncate_floor_to_int(value: f64) -> i64 {
    value as i64
}

/// Convert a floating-point upper lattice extent into an integral loop bound.
fn ceil_to_isize(value: f64) -> isize {
    isize::try_from(truncate_floor_to_int(value.ceil())).expect("grid dimension fits in isize")
}

/// Convert a floating-point grid coordinate into the corresponding hash-cell index.
fn floor_to_isize(value: f64) -> isize {
    isize::try_from(truncate_floor_to_int(value.floor())).expect("grid coordinate fits in isize")
}

/// Compress a host integer lattice coordinate into the deterministic jitter hash domain.
fn compress_axis_coord(value: isize) -> i32 {
    i32::try_from(value).expect("lattice coordinate fits in i32")
}

/// Generate one deterministic pseudo-random jitter component in `[-1, 1]`.
fn hash_jitter<T: Scalar>(ix: i32, iy: i32, iz: i32, seed: i32) -> T {
    let hash_seed = ix.wrapping_mul(73_856_093)
        ^ iy.wrapping_mul(19_349_663)
        ^ iz.wrapping_mul(83_492_791)
        ^ seed.wrapping_mul(41_293_819);
    let mut h_val = u32::from_ne_bytes(hash_seed.to_ne_bytes());
    h_val ^= h_val >> 16;
    h_val = h_val.wrapping_mul(0x85EB_CA6B);
    h_val ^= h_val >> 13;
    h_val = h_val.wrapping_mul(0xC2B2_AE35);
    h_val ^= h_val >> 16;
    let fract = f64::from(h_val) / f64::from(u32::MAX) * 2.0 - 1.0;
    <T as Scalar>::from_f64(fract)
}

/// Snap one candidate lattice point toward the SDF boundary and discard deep exterior samples.
fn snap_lattice_candidate<T: Scalar, S: Sdf3D<T>>(
    sdf: &S,
    mut point: Point3<T>,
    h: T,
    snap_iterations: usize,
    snap_radius: T,
) -> Option<Point3<T>> {
    let sr = snap_radius * h;
    let mut dist = sdf.eval(&point);
    if dist > sr {
        return None;
    }

    if eunomia::NumericElement::abs(dist) < sr {
        for _ in 0..snap_iterations {
            let grad = sdf.gradient(&point);
            if grad.norm_squared() > <T as Scalar>::from_f64(1e-12) {
                point -= grad * dist;
            }
            dist = sdf.eval(&point);
            if eunomia::NumericElement::abs(dist) < <T as Scalar>::from_f64(1e-6) * h {
                break;
            }
        }
    }

    if dist <= <T as Scalar>::from_f64(1e-5) * h {
        Some(point)
    } else {
        None
    }
}

/// Generate the snapped BCC seeding lattice inside the SDF bounds.
fn generate_bcc_points<T: Scalar, S: Sdf3D<T>>(
    sdf: &S,
    min: Point3<T>,
    max: Point3<T>,
    h: T,
    snap_iterations: usize,
    snap_radius: T,
) -> Vec<Point3<T>> {
    let half_h = h / T::from_int(2);
    let w_x = eunomia::NumericElement::to_f64((max.x - min.x) / h);
    let w_y = eunomia::NumericElement::to_f64((max.y - min.y) / h);
    let w_z = eunomia::NumericElement::to_f64((max.z - min.z) / h);
    let num_x = ceil_to_isize(w_x) + 2;
    let num_y = ceil_to_isize(w_y) + 2;
    let num_z = ceil_to_isize(w_z) + 2;
    let total_capacity = usize::try_from(2 * (num_x + 2) * (num_y + 2) * (num_z + 2))
        .expect("non-negative BCC capacity");
    let jitter_mag = <T as Scalar>::from_f64(1e-5) * h;
    let mut raw_points = Vec::with_capacity(total_capacity);

    for i in -1..=num_x {
        for j in -1..=num_y {
            for k in -1..=num_z {
                let ix = compress_axis_coord(i);
                let jy = compress_axis_coord(j);
                let kz = compress_axis_coord(k);
                let i_scalar = T::from_integer(i64::try_from(i).expect("i fits in i64"));
                let j_scalar = T::from_integer(i64::try_from(j).expect("j fits in i64"));
                let k_scalar = T::from_integer(i64::try_from(k).expect("k fits in i64"));

                let p_a = min
                    + Vector3::new(
                        i_scalar * h + hash_jitter::<T>(ix, jy, kz, 0) * jitter_mag,
                        j_scalar * h + hash_jitter::<T>(ix, jy, kz, 1) * jitter_mag,
                        k_scalar * h + hash_jitter::<T>(ix, jy, kz, 2) * jitter_mag,
                    );
                let p_b = min
                    + Vector3::new(
                        i_scalar * h + half_h + hash_jitter::<T>(ix, jy, kz, 3) * jitter_mag,
                        j_scalar * h + half_h + hash_jitter::<T>(ix, jy, kz, 4) * jitter_mag,
                        k_scalar * h + half_h + hash_jitter::<T>(ix, jy, kz, 5) * jitter_mag,
                    );

                for point in [p_a, p_b] {
                    if let Some(snapped) =
                        snap_lattice_candidate(sdf, point, h, snap_iterations, snap_radius)
                    {
                        raw_points.push(snapped);
                    }
                }
            }
        }
    }

    raw_points
}

/// Derive a deterministic shuffle seed from the meshing bounds and lattice spacing.
fn deterministic_point_seed<T: Scalar>(points: &[Point3<T>], h: T) -> u64 {
    if points.is_empty() {
        return eunomia::NumericElement::to_f64(h).to_bits();
    }

    let seed_component = |value: T| eunomia::NumericElement::to_f64(value).to_bits();
    let min = points.iter().fold(points[0], |acc, p| {
        Point3::new(acc.x.min(p.x), acc.y.min(p.y), acc.z.min(p.z))
    });
    let max = points.iter().fold(points[0], |acc, p| {
        Point3::new(acc.x.max(p.x), acc.y.max(p.y), acc.z.max(p.z))
    });
    seed_component(min.x)
        ^ seed_component(min.y).rotate_left(7)
        ^ seed_component(min.z).rotate_left(13)
        ^ seed_component(max.x).rotate_left(19)
        ^ seed_component(max.y).rotate_left(29)
        ^ seed_component(max.z).rotate_left(37)
        ^ seed_component(h).rotate_left(43)
}

/// Deduplicate BCC seeds, restore spatial locality, and apply deterministic perturbation.
fn weld_and_order_points<T: Scalar>(points: Vec<Point3<T>>, h: T) -> Vec<Point3<T>> {
    use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

    if points.is_empty() {
        return Vec::new();
    }

    let point_capacity = points.len();
    let weld_tol = <T as Scalar>::from_f64(1e-4) * h;
    let weld_tol_sq = weld_tol * weld_tol;
    let cell_s = weld_tol * T::from_int(2);
    let c_s_f64 = eunomia::NumericElement::to_f64(cell_s);
    let mut grid: HashMap<[isize; 3], Vec<usize>> = HashMap::with_capacity(points.len());
    let mut unique_points = Vec::with_capacity(points.len());

    for p in points {
        let cx = floor_to_isize(eunomia::NumericElement::to_f64(p.x) / c_s_f64);
        let cy = floor_to_isize(eunomia::NumericElement::to_f64(p.y) / c_s_f64);
        let cz = floor_to_isize(eunomia::NumericElement::to_f64(p.z) / c_s_f64);

        let mut duplicate = false;
        'outer: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = grid.get(&[cx + dx, cy + dy, cz + dz]) {
                        for &idx in indices {
                            if p.distance_squared(unique_points[idx]) < weld_tol_sq {
                                duplicate = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        if !duplicate {
            grid.entry([cx, cy, cz])
                .or_insert_with(|| Vec::with_capacity(4))
                .push(unique_points.len());
            unique_points.push(p);
        }
    }

    let mut rng = StdRng::seed_from_u64(deterministic_point_seed(&unique_points, h));
    let macro_h = eunomia::NumericElement::to_f64(T::from_int(5) * h);
    let mut blocks: HashMap<[isize; 3], Vec<Point3<T>>> =
        HashMap::with_capacity((unique_points.len() / 32).max(16));
    for p in unique_points {
        let cx = floor_to_isize(eunomia::NumericElement::to_f64(p.x) / macro_h);
        let cy = floor_to_isize(eunomia::NumericElement::to_f64(p.y) / macro_h);
        let cz = floor_to_isize(eunomia::NumericElement::to_f64(p.z) / macro_h);
        blocks
            .entry([cx, cy, cz])
            .or_insert_with(|| Vec::with_capacity(32))
            .push(p);
    }

    let mut block_list: Vec<_> = blocks.into_iter().collect();
    block_list.sort_unstable_by_key(|(key, _)| *key);
    block_list.shuffle(&mut rng);

    let mut ordered_points = Vec::with_capacity(point_capacity.max(16));
    for (_, mut block) in block_list {
        block.shuffle(&mut rng);
        ordered_points.extend(block);
    }

    let jitter_magnitude = <T as Scalar>::from_f64(1e-7) * h;
    for p in &mut ordered_points {
        p.x += <T as Scalar>::from_f64(rng.gen_range(-1.0..1.0)) * jitter_magnitude;
        p.y += <T as Scalar>::from_f64(rng.gen_range(-1.0..1.0)) * jitter_magnitude;
        p.z += <T as Scalar>::from_f64(rng.gen_range(-1.0..1.0)) * jitter_magnitude;
    }

    ordered_points
}

/// Carve away tetrahedra whose dense interior sampling escapes the SDF domain.
fn carve_tetrahedra<T: Scalar, S: Sdf3D<T>>(
    sdf: &S,
    points: &[Point3<T>],
    tetrahedra: &[[usize; 4]],
    h: T,
) -> Vec<[usize; 4]> {
    let point_four = T::from_int(4);
    let third = T::from_int(3);
    let half = <T as Scalar>::from_f64(0.5);
    let p_25 = <T as Scalar>::from_f64(0.25);
    let p_75 = <T as Scalar>::from_f64(0.75);
    let tol = <T as Scalar>::from_f64(0.25) * h;
    let mut keep = Vec::with_capacity(tetrahedra.len());

    for tet in tetrahedra {
        let p0 = points[tet[0]];
        let p1 = points[tet[1]];
        let p2 = points[tet[2]];
        let p3 = points[tet[3]];
        let checks = [
            Point3::from((p0.coords + p1.coords + p2.coords + p3.coords) / point_four),
            Point3::from((p0.coords + p1.coords + p2.coords) / third),
            Point3::from((p0.coords + p1.coords + p3.coords) / third),
            Point3::from((p0.coords + p2.coords + p3.coords) / third),
            Point3::from((p1.coords + p2.coords + p3.coords) / third),
            p0 + (p1 - p0) * half,
            p0 + (p2 - p0) * half,
            p0 + (p3 - p0) * half,
            p1 + (p2 - p1) * half,
            p1 + (p3 - p1) * half,
            p2 + (p3 - p2) * half,
            p0 + (p1 - p0) * p_25,
            p0 + (p1 - p0) * p_75,
            p0 + (p2 - p0) * p_25,
            p0 + (p2 - p0) * p_75,
            p0 + (p3 - p0) * p_25,
            p0 + (p3 - p0) * p_75,
            p1 + (p2 - p1) * p_25,
            p1 + (p2 - p1) * p_75,
            p1 + (p3 - p1) * p_25,
            p1 + (p3 - p1) * p_75,
            p2 + (p3 - p2) * p_25,
            p2 + (p3 - p2) * p_75,
        ];

        if checks.iter().all(|pt| sdf.eval(pt) <= tol) {
            keep.push(*tet);
        }
    }

    keep
}

/// Convert carved Delaunay tetrahedra into the crate's indexed volumetric mesh.
fn assemble_tetrahedral_mesh<T: Scalar>(
    points: Vec<Point3<T>>,
    keep: Vec<[usize; 4]>,
) -> IndexedMesh<T> {
    let mut used = vec![false; points.len()];
    for tet in &keep {
        for &idx in tet {
            used[idx] = true;
        }
    }

    let used_vertex_count = used.iter().filter(|&&u| u).count();
    let face_capacity = keep.len().saturating_mul(4);
    let mut mesh = IndexedMesh::with_capacity(used_vertex_count, face_capacity, keep.len());
    let mut idx_to_vid = vec![VertexId::default(); points.len()];
    for (i, p) in points.into_iter().enumerate() {
        if used[i] {
            idx_to_vid[i] = mesh.add_vertex_unique(p, leto::geometry::Vector3::zeros());
        }
    }

    let mut face_cache: HashMap<[usize; 3], FaceId> = HashMap::with_capacity(face_capacity);
    for tet in keep {
        let mut face_fids = [FaceId::default(); 4];
        let face_verts = [
            [tet[0], tet[1], tet[2]],
            [tet[0], tet[1], tet[3]],
            [tet[1], tet[2], tet[3]],
            [tet[2], tet[0], tet[3]],
        ];

        for (f_idx, mut fv) in face_verts.into_iter().enumerate() {
            fv.sort_unstable();
            let key = [fv[0], fv[1], fv[2]];
            let fid = *face_cache.entry(key).or_insert_with(|| {
                mesh.add_face(idx_to_vid[fv[0]], idx_to_vid[fv[1]], idx_to_vid[fv[2]])
            });
            face_fids[f_idx] = fid;
        }

        let mut cell = crate::domain::topology::Cell::tetrahedron(
            face_fids[0].as_usize(),
            face_fids[1].as_usize(),
            face_fids[2].as_usize(),
            face_fids[3].as_usize(),
        );
        cell.vertex_ids = vec![
            idx_to_vid[tet[0]].as_usize(),
            idx_to_vid[tet[1]].as_usize(),
            idx_to_vid[tet[2]].as_usize(),
            idx_to_vid[tet[3]].as_usize(),
        ];
        mesh.add_cell(cell);
    }

    mesh.rebuild_edges();
    mesh
}

/// Reorient each boundary face so its normal points away from the owning tetrahedron centroid.
fn orient_boundary_faces<T: Scalar>(mesh: &mut IndexedMesh<T>, b_faces: &[FaceId]) {
    let mut face_to_cell: HashMap<FaceId, usize> = HashMap::with_capacity(b_faces.len());
    for (cell_idx, cell) in mesh.cells.iter().enumerate() {
        for &fv_idx in &cell.faces {
            face_to_cell.insert(FaceId::from_usize(fv_idx), cell_idx);
        }
    }

    let third = T::from_int(3);
    for &fid in b_faces {
        if let Some(&cell_idx) = face_to_cell.get(&fid) {
            let cell = &mesh.cells[cell_idx];
            let face_data = *mesh.faces.get(fid);
            let a = mesh.vertices.position(face_data.vertices[0]);
            let b = mesh.vertices.position(face_data.vertices[1]);
            let c = mesh.vertices.position(face_data.vertices[2]);
            let face_centroid = (a.coords + b.coords + c.coords) / third;
            let unorm = (b.coords - a.coords).cross(c.coords - a.coords);

            let mut cell_sum = Vector3::zeros();
            for &vid in &cell.vertex_ids {
                cell_sum += mesh.vertices.position(VertexId::from_usize(vid)).coords;
            }
            let cell_centroid = cell_sum / T::from_count(cell.vertex_ids.len());
            if (face_centroid - cell_centroid).dot(unorm) < <T as eunomia::NumericElement>::ZERO {
                mesh.faces.get_mut(fid).flip();
            }
        }
    }
}

/// Relax the boundary rings and reproject them to the SDF surface for smoother output triangles.
fn relax_boundary_vertices<T: Scalar, S: Sdf3D<T>>(
    mesh: &mut IndexedMesh<T>,
    sdf: &S,
    b_faces: &[FaceId],
    h: T,
) {
    let mut b_vertices = HashSet::with_capacity(b_faces.len() / 2);
    let mut b_adj: HashMap<VertexId, Vec<VertexId>> = HashMap::with_capacity(b_faces.len() / 2);
    for fid in b_faces {
        let face = mesh.faces.get(*fid);
        let v = face.vertices;
        for &vid in &v {
            b_vertices.insert(vid);
        }
        b_adj
            .entry(v[0])
            .or_insert_with(|| Vec::with_capacity(6))
            .extend([v[1], v[2]]);
        b_adj
            .entry(v[1])
            .or_insert_with(|| Vec::with_capacity(6))
            .extend([v[0], v[2]]);
        b_adj
            .entry(v[2])
            .or_insert_with(|| Vec::with_capacity(6))
            .extend([v[0], v[1]]);
    }

    for neighbors in b_adj.values_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut next_pos = Vec::with_capacity(b_vertices.len());
    for _ in 0..10 {
        next_pos.clear();
        for &vid in &b_vertices {
            let neighbors = &b_adj[&vid];
            let mut sum = Vector3::zeros();
            for &n_vid in neighbors {
                sum += mesh.vertices.position(n_vid).coords;
            }

            let weight = T::from_count(neighbors.len());
            let mut p = Point3::from(sum / weight);
            let mut dist = sdf.eval(&p);
            for _ in 0..5 {
                let grad = sdf.gradient(&p);
                if grad.norm_squared() > <T as Scalar>::from_f64(1e-12) {
                    p -= grad * dist;
                }
                dist = sdf.eval(&p);
                if eunomia::NumericElement::abs(dist) < <T as Scalar>::from_f64(1e-6) * h {
                    break;
                }
            }
            next_pos.push((vid, p));
        }

        for &(vid, p) in &next_pos {
            mesh.vertices.set_position(vid, p);
        }
    }
}
