//! Normal-orientation analysis for `IndexedMesh` surfaces.
//!
//! Provides [`NormalAnalysis`] and [`analyze_normals`] — routinely used by CSG
//! examples and validation tools to report face-winding consistency and
//! vertex-normal alignment across a mesh.
//!
//! ## Algorithm
//!
//! 1. Build a half-edge adjacency map `(v_i, v_j) → face_idx`.
//! 2. Identify the face with the most extreme vertex (maximum X coordinate).
//!    For this face the outward direction is unambiguous: the face normal must
//!    have a **positive X component** to point away from the solid.
//! 3. Assign that seed face an `Outward` orientation based on sign of `n.x`.
//! 4. **Manifold BFS flood**: propagate orientation to all adjacent faces via
//!    the shared half-edge graph.  Two faces sharing edge (A→B) and (B→A) have
//!    consistent winding (both outward or both inward); sharing (A→B) and (A→B)
//!    (same direction) indicates a winding flip between the two faces.
//! 5. Faces unreachable from the seed (disconnected patches) are re-seeded
//!    from an unvisited extremal face.
//! 6. Count outward / inward from BFS labels; compute vertex-normal alignment.
//!
//! ## Properties
//!
//! - **Correct for non-convex meshes**: CSG difference, tori, concave shapes
//!   all produce `inward_faces = 0` when winding is globally consistent.
//! - **O(F + E)** time and space.
//! - **No centroid**: eliminates the false-positive bias of the old heuristic.
//!
//! ## Interpretation
//!
//! | `inward_faces / total_faces` | Likely cause                              |
//! |------------------------------|-------------------------------------------|
//! | 0%                           | Globally consistent outward winding       |
//! | < 5%                         | Acceptable; isolated CDT seam artefacts   |
//! | > 10%                        | Winding problem; check Boolean op result  |
//! | ≈ 50%                        | Mixed winding; mesh likely non-manifold   |
//!
//! `face_vertex_alignment_mean` near 1.0 means stored vertex normals agree with
//! computed face normals (good for smooth-shaded rendering and CFD post-processing).

use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Real, Vector3r};
use crate::domain::geometry::normal::triangle_normal;
use crate::domain::mesh::IndexedMesh;
use eunomia::FloatElement;

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-mesh normal-orientation statistics.
///
/// Returned by [`analyze_normals`].
///
/// # Invariants
///
/// - `outward_faces + inward_faces + degenerate_faces == total triangles checked`
/// - `face_vertex_alignment_mean` ∈ [−1, 1]; 1.0 = perfect agreement
/// - `face_vertex_alignment_min`  ∈ [−1, 1]; < 0 indicates at least one
///   face whose stored vertex normals point opposite to the winding normal
#[derive(Debug, Clone, PartialEq)]
pub struct NormalAnalysis {
    /// Number of faces whose computed normal is consistent with the outward
    /// manifold orientation (determined by BFS flood from the extremal seed face).
    pub outward_faces: usize,
    /// Number of faces whose computed normal is inconsistent with the outward
    /// manifold orientation (flipped winding relative to neighbours).
    pub inward_faces: usize,
    /// Number of degenerate (zero-area) faces skipped during analysis.
    pub degenerate_faces: usize,
    /// Mean dot product of the computed face normal vs the averaged stored
    /// vertex normals across all non-degenerate faces.
    pub face_vertex_alignment_mean: Real,
    /// Minimum dot product across all non-degenerate faces.
    pub face_vertex_alignment_min: Real,
}

impl NormalAnalysis {
    /// Total number of faces inspected (degenerate faces included).
    #[inline]
    #[must_use]
    pub fn total_faces(&self) -> usize {
        self.outward_faces + self.inward_faces + self.degenerate_faces
    }

    /// Fraction of non-degenerate faces that are inward-facing (0.0 – 1.0).
    ///
    /// Returns `0.0` when the mesh is empty.
    #[inline]
    #[must_use]
    pub fn inward_fraction(&self) -> Real {
        let n = Real::from_count(self.outward_faces + self.inward_faces);
        if n > 0.0 {
            Real::from_count(self.inward_faces) / n
        } else {
            0.0
        }
    }

    /// Returns true when the inward-face fraction is below a threshold.
    ///
    /// The threshold uses an [`aequitas`] `Dimensionless` quantity to
    /// prevent confusion with fraction vs percentage.
    #[inline]
    #[must_use]
    pub fn is_consistent_within(
        &self,
        max_inward_fraction: aequitas::systems::si::quantities::Dimensionless<Real>,
    ) -> bool {
        self.inward_fraction() <= max_inward_fraction.into_base()
    }

    /// Returns `true` when every non-degenerate face is outward-facing.
    #[inline]
    #[must_use]
    pub fn all_outward(&self) -> bool {
        self.inward_faces == 0
    }
}

// ── Public function ───────────────────────────────────────────────────────────

/// Analyse the normal orientation of every face in `mesh`.
///
/// Uses a **manifold BFS flood** seeded from the face with the most extreme
/// vertex to determine globally consistent outward orientation.  This is
/// correct for any closed orientable 2-manifold, including non-convex CSG
/// difference solids, tori, and re-entrant geometries.
///
/// # Arguments
///
/// * `mesh` — The surface mesh to analyse.  Takes a shared reference.
///
/// # Returns
///
/// A [`NormalAnalysis`] struct with per-category counts and alignment
/// statistics.
///
/// # Examples
///
/// ```rust
/// use gaia::{UvSphere, primitives::PrimitiveMesh, analyze_normals};
///
/// let sphere = UvSphere { radius: 1.0, segments: 32, stacks: 16, ..Default::default() }
///     .build().unwrap();
/// let report = analyze_normals(&sphere);
/// assert_eq!(report.inward_faces, 0, "sphere should be all-outward");
/// ```
#[must_use]
pub fn analyze_normals(mesh: &IndexedMesh) -> NormalAnalysis {
    let face_list = mesh.faces.as_slice();
    let n_faces = face_list.len();

    if n_faces == 0 {
        return NormalAnalysis {
            outward_faces: 0,
            inward_faces: 0,
            degenerate_faces: 0,
            face_vertex_alignment_mean: 0.0,
            face_vertex_alignment_min: 0.0,
        };
    }

    // Per-face computed normals (None = degenerate).
    let mut face_normals: Vec<Option<Vector3r>> = Vec::with_capacity(n_faces);
    for face in face_list {
        let a = mesh.vertices.position(face.vertices[0]);
        let b = mesh.vertices.position(face.vertices[1]);
        let c = mesh.vertices.position(face.vertices[2]);
        face_normals.push(triangle_normal(a, b, c));
    }

    // ── Step 2: build half-edge adjacency (directed edge → face index) ───────
    //
    // half_edge[(v_i, v_j)] = face_idx of the face that has directed edge i→j.
    // For a manifold mesh every directed edge appears in exactly one face.
    //
    // Uses `hashbrown::HashMap` for consistent performance with the rest of
    // the mesh pipeline (lower overhead than std HashMap).
    let mut half_edge: hashbrown::HashMap<(VertexId, VertexId), usize> =
        hashbrown::HashMap::with_capacity(n_faces * 3);
    for (fi, face) in face_list.iter().enumerate() {
        let v = face.vertices;
        for k in 0..3 {
            let j = (k + 1) % 3;
            half_edge.insert((v[k], v[j]), fi);
        }
    }

    // ── Step 3: BFS flood orientation from extremal seeds ───────────────────
    let orientation = flood_orientation_bfs(face_list, &face_normals, &half_edge, mesh);

    // ── Step 4: count outward / inward / degenerate ──────────────────────────
    let mut outward = 0usize;
    let mut inward = 0usize;
    let mut degen = 0usize;

    for fi in 0..n_faces {
        if face_normals[fi].is_none() {
            degen += 1;
        } else {
            match orientation[fi] {
                Some(true) => outward += 1,
                Some(false) => inward += 1,
                None => degen += 1, // unreachable non-manifold fragment
            }
        }
    }

    // ── Signed-volume verification ──────────────────────────────────────────
    //
    // The max-vertex-X seed heuristic assumes the extreme face's outward
    // normal has non-negative X.  This fails for concave CSG results
    // (e.g., N-ary Intersection/Difference producing pocket-like shapes).
    //
    // The divergence-theorem signed volume is the ground truth:
    //   - signed_vol > 0 → winding is outward  → BFS should label majority as outward
    //   - signed_vol < 0 → winding is inward   → BFS should label majority as inward
    //
    // When BFS disagrees with the signed-volume sign, the seed heuristic
    // was wrong.  Swap outward ↔ inward to match reality.
    let signed_vol: Real = crate::domain::geometry::measure::total_signed_volume(
        mesh.faces.iter_enumerated().map(|(_, face)| {
            (
                mesh.vertices.position(face.vertices[0]),
                mesh.vertices.position(face.vertices[1]),
                mesh.vertices.position(face.vertices[2]),
            )
        }),
    );
    let bfs_says_outward = outward >= inward;
    let vol_says_outward = signed_vol >= 0.0;
    if bfs_says_outward != vol_says_outward {
        std::mem::swap(&mut outward, &mut inward);
    }

    // ── Step 5: face ↔ vertex-normal alignment statistics ───────────────────
    let mut asum: Real = 0.0;
    let mut acnt = 0usize;
    let mut amin: Real = 1.0;

    for (fi, face) in face_list.iter().enumerate() {
        let Some(face_n) = face_normals[fi] else {
            continue;
        };
        let avg_n = (*mesh.vertices.normal(face.vertices[0])
            + *mesh.vertices.normal(face.vertices[1])
            + *mesh.vertices.normal(face.vertices[2]))
            / 3.0;
        let l = avg_n.norm();
        if l > 1e-12 {
            let al = face_n.dot(avg_n / l);
            asum += al;
            acnt += 1;
            amin = amin.min(al);
        }
    }

    NormalAnalysis {
        outward_faces: outward,
        inward_faces: inward,
        degenerate_faces: degen,
        face_vertex_alignment_mean: if acnt > 0 {
            asum / Real::from_count(acnt)
        } else {
            0.0
        },
        face_vertex_alignment_min: if acnt > 0 { amin } else { 0.0 },
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

use crate::infrastructure::storage::face_store::FaceData;

/// BFS orientation flood from extremal seeds.
///
/// Returns `orientation[fi]`: `Some(true)` = outward-consistent, `Some(false)` =
/// inward-consistent, `None` = degenerate or unreachable non-manifold fragment.
///
/// Handles disconnected components via a cursor-based outer loop — O(F) total work.
fn flood_orientation_bfs(
    face_list: &[FaceData],
    face_normals: &[Option<Vector3r>],
    half_edge: &hashbrown::HashMap<(VertexId, VertexId), usize>,
    mesh: &IndexedMesh,
) -> Vec<Option<bool>> {
    let n_faces = face_list.len();
    let mut orientation: Vec<Option<bool>> = vec![None; n_faces];
    let mut seed_cursor = 0usize;
    let mut queue = std::collections::VecDeque::with_capacity(n_faces);

    loop {
        while seed_cursor < n_faces
            && (orientation[seed_cursor].is_some() || face_normals[seed_cursor].is_none())
        {
            seed_cursor += 1;
        }
        if seed_cursor >= n_faces {
            break;
        }
        let mut best_x = f64::NEG_INFINITY;
        let mut seed_fi = seed_cursor;
        for fi in seed_cursor..n_faces {
            if orientation[fi].is_some() || face_normals[fi].is_none() {
                continue;
            }
            for &vid in &face_list[fi].vertices {
                let px = mesh.vertices.position(vid).x;
                if px > best_x {
                    best_x = px;
                    seed_fi = fi;
                }
            }
        }
        // Invariant: face_normals[seed_fi].is_some() — cursor exits with is_some();
        // scan only updates seed_fi for is_some() faces.
        let Some(seed_normal) = face_normals[seed_fi] else {
            continue;
        };
        orientation[seed_fi] = Some(seed_normal.x >= 0.0);
        queue.clear();
        queue.push_back(seed_fi);
        while let Some(fi) = queue.pop_front() {
            // Invariant: orientation is set before each push_back.
            let Some(is_outward) = orientation[fi] else {
                continue;
            };
            let v = face_list[fi].vertices;
            for k in 0..3 {
                let j = (k + 1) % 3;
                let va = v[k];
                let vb = v[j];
                if let Some(&nfi) = half_edge.get(&(vb, va)) {
                    if orientation[nfi].is_none() && face_normals[nfi].is_some() {
                        orientation[nfi] = Some(is_outward);
                        queue.push_back(nfi);
                    }
                } else if let Some(&nfi) = half_edge.get(&(va, vb))
                    && orientation[nfi].is_none()
                    && face_normals[nfi].is_some()
                {
                    orientation[nfi] = Some(!is_outward);
                    queue.push_back(nfi);
                }
            }
        }
    }
    orientation
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::Point3r;
    use crate::domain::geometry::primitives::{Cube, PrimitiveMesh, UvSphere};

    #[test]
    fn sphere_all_outward() {
        let mesh = UvSphere {
            radius: 1.0,
            segments: 32,
            stacks: 16,
            ..Default::default()
        }
        .build()
        .unwrap();
        let r = analyze_normals(&mesh);
        assert_eq!(r.inward_faces, 0, "UV sphere should have zero inward faces");
        assert_eq!(
            r.degenerate_faces, 0,
            "UV sphere should have no degenerate faces"
        );
        assert!(
            r.face_vertex_alignment_mean > 0.9,
            "face-vertex alignment mean should be > 0.9, got {}",
            r.face_vertex_alignment_mean
        );
    }

    #[test]
    fn cube_all_outward() {
        let mesh = Cube {
            origin: Point3r::origin(),
            width: 2.0,
            height: 2.0,
            depth: 2.0,
        }
        .build()
        .unwrap();
        let r = analyze_normals(&mesh);
        assert_eq!(r.inward_faces, 0, "cube should have zero inward faces");
    }

    #[test]
    fn empty_mesh_returns_zeros() {
        let mesh = IndexedMesh::new();
        let r = analyze_normals(&mesh);
        assert_eq!(r.outward_faces, 0);
        assert_eq!(r.inward_faces, 0);
        assert_eq!(r.degenerate_faces, 0);
        assert_eq!(r.face_vertex_alignment_mean.to_bits(), 0.0_f64.to_bits());
        assert_eq!(r.face_vertex_alignment_min.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn inward_fraction_zero_on_clean_mesh() {
        let mesh = UvSphere {
            radius: 1.0,
            segments: 16,
            stacks: 8,
            ..Default::default()
        }
        .build()
        .unwrap();
        let r = analyze_normals(&mesh);
        assert_eq!(r.inward_fraction().to_bits(), 0.0_f64.to_bits());
        assert!(r.all_outward());
    }

    #[test]
    fn is_consistent_within_accepts_clean_mesh() {
        use aequitas::systems::si::quantities::Dimensionless;

        let mesh = Cube {
            origin: Point3r::origin(),
            width: 2.0,
            height: 2.0,
            depth: 2.0,
        }
        .build()
        .unwrap();
        let r = analyze_normals(&mesh);
        assert!(r.is_consistent_within(Dimensionless::from_base(0.0)));
    }

    #[test]
    fn total_faces_matches_mesh() {
        let mesh = UvSphere {
            radius: 1.0,
            segments: 16,
            stacks: 8,
            ..Default::default()
        }
        .build()
        .unwrap();
        let r = analyze_normals(&mesh);
        assert_eq!(r.total_faces(), mesh.face_count());
    }

    // ── Adversarial BFS analysis tests ────────────────────────────────────

    /// Signed-volume correction: BFS labels all faces "outward" for a fully
    /// inward-wound mesh (consistent BFS), but the negative signed volume
    /// signals the seed was wrong. Swapping counts corrects the analysis.
    #[test]
    fn analyze_normals_all_inward_tet() {
        use crate::domain::mesh::IndexedMesh;

        let mut mesh = IndexedMesh::with_cell_size(0.01);
        let v0 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
        let v1 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
        let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 1.0));
        let v3 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
        // CW winding (inward)
        mesh.add_face(v0, v2, v1);
        mesh.add_face(v0, v1, v3);
        mesh.add_face(v0, v3, v2);
        mesh.add_face(v1, v2, v3);

        let r = analyze_normals(&mesh);
        assert_eq!(r.total_faces(), 4);
        // All faces have the same (inward) winding, so BFS labels them
        // consistently.  The signed-volume swap means all 4 are reported
        // as "inward" after correction.
        assert_eq!(
            r.inward_faces, 4,
            "all-inward tet should report 4 inward faces, got {}",
            r.inward_faces
        );
        assert_eq!(r.outward_faces, 0);
    }

    /// BFS re-seeds for each disconnected component, so the total face count
    /// must equal the sum across all components.
    #[test]
    fn analyze_normals_two_disjoint_cubes() {
        // Two separate cubes — both outward-wound.
        let cube1 = Cube {
            origin: Point3r::new(0.0, 0.0, 0.0),
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        }
        .build()
        .unwrap();
        let cube2 = Cube {
            origin: Point3r::new(10.0, 0.0, 0.0),
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        }
        .build()
        .unwrap();

        // Merge into one mesh.
        let mut combined = IndexedMesh::with_cell_size(1e-4);
        for fi in 0..cube1.face_count() {
            let fid = crate::domain::core::index::FaceId::from_usize(fi);
            let face = cube1.faces.get(fid);
            let a = combined.add_vertex_pos(*cube1.vertices.position(face.vertices[0]));
            let b = combined.add_vertex_pos(*cube1.vertices.position(face.vertices[1]));
            let c = combined.add_vertex_pos(*cube1.vertices.position(face.vertices[2]));
            combined.add_face(a, b, c);
        }
        for fi in 0..cube2.face_count() {
            let fid = crate::domain::core::index::FaceId::from_usize(fi);
            let face = cube2.faces.get(fid);
            let a = combined.add_vertex_pos(*cube2.vertices.position(face.vertices[0]));
            let b = combined.add_vertex_pos(*cube2.vertices.position(face.vertices[1]));
            let c = combined.add_vertex_pos(*cube2.vertices.position(face.vertices[2]));
            combined.add_face(a, b, c);
        }

        let r = analyze_normals(&combined);
        assert_eq!(
            r.total_faces(),
            cube1.face_count() + cube2.face_count(),
            "total faces must cover both components"
        );
        assert_eq!(
            r.inward_faces, 0,
            "two correctly-wound cubes must have zero inward faces"
        );
        assert!(r.all_outward());
    }
}
