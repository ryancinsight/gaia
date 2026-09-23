//! Repair pass: collapse degenerate (zero-area) faces.

use super::uf_find;
use crate::domain::core::constants::{BOOLEAN_COINCIDENT_LEN_SQ, BOOLEAN_DEGENERACY_SIN2_TOL};
use crate::domain::core::index::VertexId;
use crate::domain::geometry::normal::triangle_normal;
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::{FaceData, FaceStore};

/// Squared world length below which a degenerate face's shortest edge counts as
/// a coincident-vertex pair.
///
/// Delegates to [`BOOLEAN_COINCIDENT_LEN_SQ`] (SSOT).
const COINCIDENT_TOLERANCE_SQUARED: f64 = BOOLEAN_COINCIDENT_LEN_SQ;

/// `sin²θ` at which a face counts as degenerate, and gets repaired.
///
/// Dimensionless, so the same face is treated the same way at any mesh scale:
/// the test is `cross² < TOL × |ab|² × |ac|²`, the shape `DEGENERATE_NORMAL_REL_SQ`
/// uses. Delegates to [`BOOLEAN_DEGENERACY_SIN2_TOL`] (SSOT).
const DEGENERACY_SIN2_TOL: f64 = BOOLEAN_DEGENERACY_SIN2_TOL;

enum FaceGeometry {
    Degenerate { edge_lengths_squared: [f64; 3] },
    Nondegenerate,
}

#[derive(Clone, Copy)]
enum CollapseKind {
    Coincident,
    Sliver,
}

#[derive(Clone, Copy)]
struct CollapseCandidate {
    face_index: usize,
    keep: VertexId,
    remove: VertexId,
    kind: CollapseKind,
}

struct CollapseWorkspace {
    edge_use: hashbrown::HashMap<(VertexId, VertexId), usize>,
    skipped_faces: hashbrown::HashSet<usize>,
    face_keys: hashbrown::HashSet<[VertexId; 3]>,
    retained_faces: Vec<FaceData>,
    total_collapsed: usize,
}

impl CollapseWorkspace {
    fn with_capacity(face_count: usize) -> Self {
        Self {
            edge_use: hashbrown::HashMap::new(),
            skipped_faces: hashbrown::HashSet::with_capacity(face_count / 16),
            face_keys: hashbrown::HashSet::with_capacity(face_count),
            retained_faces: Vec::with_capacity(face_count),
            total_collapsed: 0,
        }
    }

    fn replace_faces(&mut self, mesh: &mut IndexedMesh) {
        let mut faces = FaceStore::with_capacity(self.retained_faces.len());
        for face in self.retained_faces.drain(..) {
            faces.push(face);
        }
        mesh.faces = faces;
    }
}

/// Collapse degenerate (zero-area) faces by merging the redundant vertex.
///
/// Near-coincident vertex pairs are merged in one union-find pass. Collinear
/// slivers are considered one at a time because each collapse changes the
/// edge-use counts used by the next candidate.
pub(super) fn collapse_degenerate_faces(mesh: &mut IndexedMesh) {
    let mut workspace = CollapseWorkspace::with_capacity(mesh.faces.len());
    merge_near_coincident_vertices(mesh, &mut workspace);
    build_edge_use(mesh, &mut workspace);

    while let Some(candidate) = find_collapse_candidate(mesh, &workspace) {
        if let CollapseKind::Sliver = candidate.kind
            && !sliver_collapse_is_safe(mesh, candidate, &mut workspace)
        {
            workspace.skipped_faces.insert(candidate.face_index);
            continue;
        }

        apply_collapse(mesh, candidate, &mut workspace);
    }

    if workspace.total_collapsed > 0 {
        tracing::debug!(
            "CSG postprocess: collapsed {} degenerate face(s)",
            workspace.total_collapsed
        );
    }
}

fn face_geometry(mesh: &IndexedMesh, face: &FaceData) -> FaceGeometry {
    let pa = mesh.vertices.position(face.vertices[0]);
    let pb = mesh.vertices.position(face.vertices[1]);
    let pc = mesh.vertices.position(face.vertices[2]);
    let ab = pb - pa;
    let ac = pc - pa;
    let cross_squared = ab.cross(ac).norm_squared();
    let edge_lengths_squared = [
        ab.norm_squared(),
        (pc - pb).norm_squared(),
        ac.norm_squared(),
    ];
    let max_edge_squared = edge_lengths_squared[0]
        .max(edge_lengths_squared[1])
        .max(edge_lengths_squared[2]);
    // `|ab × ac| = |ab||ac|·sinθ`, so dividing by the two edge lengths squared
    // leaves `sin²θ` — scale-free, unlike the `cross² / max_edge²` ratio this
    // replaced, which carried a `length²` and so classified the same face
    // differently at different mesh scales.
    let ab_sq = edge_lengths_squared[0];
    let ac_sq = edge_lengths_squared[2];
    let nondegenerate = max_edge_squared > 0.0
        && ab_sq > 0.0
        && ac_sq > 0.0
        && cross_squared >= DEGENERACY_SIN2_TOL * ab_sq * ac_sq;

    if nondegenerate {
        FaceGeometry::Nondegenerate
    } else {
        FaceGeometry::Degenerate {
            edge_lengths_squared,
        }
    }
}

fn merge_near_coincident_vertices(mesh: &mut IndexedMesh, workspace: &mut CollapseWorkspace) {
    let vertex_count = mesh.vertices.len();
    if vertex_count == 0 {
        return;
    }

    let mut parent: Vec<u32> = (0..vertex_count as u32).collect();
    for face in mesh.faces.iter() {
        let FaceGeometry::Degenerate {
            edge_lengths_squared,
        } = face_geometry(mesh, face)
        else {
            continue;
        };
        let shortest = edge_lengths_squared[0]
            .min(edge_lengths_squared[1])
            .min(edge_lengths_squared[2]);
        if shortest > COINCIDENT_TOLERANCE_SQUARED {
            continue;
        }

        let [d01, d12, d20] = edge_lengths_squared;
        let (keep, remove) = if d01 <= d12 && d01 <= d20 {
            (face.vertices[0], face.vertices[1])
        } else if d12 <= d20 {
            (face.vertices[1], face.vertices[2])
        } else {
            (face.vertices[2], face.vertices[0])
        };
        let keep_root = uf_find(&mut parent, keep.0);
        let remove_root = uf_find(&mut parent, remove.0);
        if keep_root != remove_root {
            let (lower, upper) = if keep_root < remove_root {
                (keep_root, remove_root)
            } else {
                (remove_root, keep_root)
            };
            parent[upper as usize] = lower;
        }
    }

    let roots: Vec<u32> = (0..vertex_count)
        .map(|index| uf_find(&mut parent, index as u32))
        .collect();
    if !roots
        .iter()
        .enumerate()
        .any(|(index, &root)| root != index as u32)
    {
        return;
    }

    workspace.face_keys.clear();
    workspace.retained_faces.clear();
    let mut removed = 0;
    for face in mesh.faces.iter() {
        let mut remapped = *face;
        for vertex in &mut remapped.vertices {
            *vertex = VertexId(roots[vertex.0 as usize]);
        }
        if has_repeated_vertices(remapped.vertices)
            || !workspace
                .face_keys
                .insert(canonical_face_key(remapped.vertices))
        {
            removed += 1;
            continue;
        }
        workspace.retained_faces.push(remapped);
    }

    workspace.total_collapsed += removed;
    workspace.replace_faces(mesh);
}

#[inline]
fn edge_key(a: VertexId, b: VertexId) -> (VertexId, VertexId) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[inline]
fn add_face_edges(
    map: &mut hashbrown::HashMap<(VertexId, VertexId), usize>,
    vertices: [VertexId; 3],
) {
    if has_repeated_vertices(vertices) {
        return;
    }
    for (a, b) in [
        (vertices[0], vertices[1]),
        (vertices[1], vertices[2]),
        (vertices[2], vertices[0]),
    ] {
        *map.entry(edge_key(a, b)).or_insert(0) += 1;
    }
}

#[inline]
fn remove_face_edges(
    map: &mut hashbrown::HashMap<(VertexId, VertexId), usize>,
    vertices: [VertexId; 3],
) {
    for (a, b) in [
        (vertices[0], vertices[1]),
        (vertices[1], vertices[2]),
        (vertices[2], vertices[0]),
    ] {
        let key = edge_key(a, b);
        match map.get_mut(&key) {
            Some(count) if *count > 1 => *count -= 1,
            _ => {
                map.remove(&key);
            }
        }
    }
}

fn build_edge_use(mesh: &IndexedMesh, workspace: &mut CollapseWorkspace) {
    workspace.edge_use = hashbrown::HashMap::with_capacity(mesh.faces.len() * 3 / 2);
    for face in mesh.faces.iter() {
        add_face_edges(&mut workspace.edge_use, face.vertices);
    }
}

fn find_collapse_candidate(
    mesh: &IndexedMesh,
    workspace: &CollapseWorkspace,
) -> Option<CollapseCandidate> {
    for (face_index, face) in mesh.faces.iter().enumerate() {
        if workspace.skipped_faces.contains(&face_index) {
            continue;
        }
        let FaceGeometry::Degenerate {
            edge_lengths_squared,
        } = face_geometry(mesh, face)
        else {
            continue;
        };
        let shortest = edge_lengths_squared[0]
            .min(edge_lengths_squared[1])
            .min(edge_lengths_squared[2]);

        if shortest <= COINCIDENT_TOLERANCE_SQUARED {
            let [d01, d12, d20] = edge_lengths_squared;
            let (keep, remove) = if d01 <= d12 && d01 <= d20 {
                (face.vertices[0], face.vertices[1])
            } else if d12 <= d20 {
                (face.vertices[1], face.vertices[2])
            } else {
                (face.vertices[2], face.vertices[0])
            };
            return Some(CollapseCandidate {
                face_index,
                keep,
                remove,
                kind: CollapseKind::Coincident,
            });
        }

        let [d01, d12, d20] = edge_lengths_squared;
        let (middle_index, endpoint_a_index, endpoint_b_index) = if d01 >= d12 && d01 >= d20 {
            (2, 0, 1)
        } else if d12 >= d20 {
            (0, 1, 2)
        } else {
            (1, 2, 0)
        };
        let middle = face.vertices[middle_index];
        let endpoint_a = face.vertices[endpoint_a_index];
        let endpoint_b = face.vertices[endpoint_b_index];
        let middle_position = mesh.vertices.position(middle);
        let position_a = mesh.vertices.position(endpoint_a);
        let position_b = mesh.vertices.position(endpoint_b);
        let (keep, remove) = if (position_a - middle_position).norm_squared()
            <= (position_b - middle_position).norm_squared()
        {
            (endpoint_a, middle)
        } else {
            (endpoint_b, middle)
        };

        if workspace
            .edge_use
            .get(&edge_key(keep, remove))
            .copied()
            .unwrap_or(0)
            > 2
        {
            continue;
        }
        return Some(CollapseCandidate {
            face_index,
            keep,
            remove,
            kind: CollapseKind::Sliver,
        });
    }
    None
}

fn sliver_collapse_is_safe(
    mesh: &IndexedMesh,
    candidate: CollapseCandidate,
    workspace: &mut CollapseWorkspace,
) -> bool {
    workspace.face_keys.clear();
    for face in mesh.faces.iter() {
        let vertices = remap_vertex(face.vertices, candidate.remove, candidate.keep);
        if has_repeated_vertices(vertices) {
            continue;
        }
        if !workspace.face_keys.insert(canonical_face_key(vertices)) {
            return false;
        }
    }

    for face in mesh.faces.iter() {
        if !face.vertices.contains(&candidate.remove) {
            continue;
        }
        let before = face.vertices.map(|vertex| *mesh.vertices.position(vertex));
        let previous_normal = triangle_normal(&before[0], &before[1], &before[2]);
        let vertices = remap_vertex(face.vertices, candidate.remove, candidate.keep);
        if has_repeated_vertices(vertices) {
            continue;
        }
        let after = vertices.map(|vertex| *mesh.vertices.position(vertex));
        let next_normal = triangle_normal(&after[0], &after[1], &after[2]);
        if let (Some(previous), Some(next)) = (previous_normal, next_normal)
            && previous.dot(next) < 0.0
        {
            return false;
        }
    }

    true
}

fn apply_collapse(
    mesh: &mut IndexedMesh,
    candidate: CollapseCandidate,
    workspace: &mut CollapseWorkspace,
) {
    for face in mesh.faces.iter_mut() {
        if !face.vertices.contains(&candidate.remove) {
            continue;
        }
        remove_face_edges(&mut workspace.edge_use, face.vertices);
        face.vertices = remap_vertex(face.vertices, candidate.remove, candidate.keep);
        add_face_edges(&mut workspace.edge_use, face.vertices);
    }

    workspace.face_keys.clear();
    workspace.retained_faces.clear();
    let mut removed = 0;
    for face in mesh.faces.iter() {
        if has_repeated_vertices(face.vertices) {
            removed += 1;
            continue;
        }
        if !workspace
            .face_keys
            .insert(canonical_face_key(face.vertices))
        {
            remove_face_edges(&mut workspace.edge_use, face.vertices);
            removed += 1;
            continue;
        }
        workspace.retained_faces.push(*face);
    }

    workspace.total_collapsed += removed;
    workspace.skipped_faces.clear();
    workspace.replace_faces(mesh);
}

fn remap_vertex(vertices: [VertexId; 3], remove: VertexId, keep: VertexId) -> [VertexId; 3] {
    vertices.map(|vertex| if vertex == remove { keep } else { vertex })
}

fn has_repeated_vertices(vertices: [VertexId; 3]) -> bool {
    vertices[0] == vertices[1] || vertices[1] == vertices[2] || vertices[2] == vertices[0]
}

fn canonical_face_key(mut vertices: [VertexId; 3]) -> [VertexId; 3] {
    if vertices[0] > vertices[1] {
        vertices.swap(0, 1);
    }
    if vertices[1] > vertices[2] {
        vertices.swap(1, 2);
    }
    if vertices[0] > vertices[1] {
        vertices.swap(0, 1);
    }
    vertices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::{Point3r, Vector3r};

    /// A face with `sin²θ = sin2` at vertex `a`, scaled by `scale`:
    /// `ab = (L, 0, 0)` and `ac = (L, L·t, 0)` with `t = sin2/√(1 − sin2)`.
    fn face_with_sin2(scale: f64, sin2: f64) -> (IndexedMesh, FaceData) {
        let t = (sin2 / (1.0 - sin2)).sqrt();
        let mut mesh: IndexedMesh = IndexedMesh::with_cell_size(scale * 1e-6);
        let n = Vector3r::zeros();
        let a = mesh.add_vertex(Point3r::new(0.0, 0.0, 0.0), n);
        let b = mesh.add_vertex(Point3r::new(scale, 0.0, 0.0), n);
        let c = mesh.add_vertex(Point3r::new(scale, scale * t, 0.0), n);
        let face = FaceData::untagged(a, b, c);
        (mesh, face)
    }

    fn is_degenerate(mesh: &IndexedMesh, face: &FaceData) -> bool {
        matches!(face_geometry(mesh, face), FaceGeometry::Degenerate { .. })
    }

    /// The degeneracy decision must not depend on the mesh's scale: a face with a
    /// given `sin²θ` is repaired or kept identically at 10 µm and at 1 km.
    ///
    /// Under the previous `cross² / max_edge²` threshold this failed — that ratio
    /// carries a `length²`, so the same face drifted across the boundary with
    /// scale.
    #[test]
    fn degeneracy_decision_is_scale_invariant() {
        // Deliberately away from the threshold: at exactly `sin²θ = tol` the
        // comparison is a rounding coin flip and pins nothing.
        for &sin2 in &[1e-14_f64, 1e-10] {
            let mut decisions = Vec::new();
            for scale in [1e-5_f64, 1e-3, 1.0, 1e3, 1e5] {
                let (mesh, face) = face_with_sin2(scale, sin2);
                decisions.push(is_degenerate(&mesh, &face));
            }
            assert!(
                decisions.windows(2).all(|w| w[0] == w[1]),
                "sin²θ = {sin2:e} must be classified the same at every scale: {decisions:?}"
            );
            assert_eq!(
                decisions[0],
                sin2 < DEGENERACY_SIN2_TOL,
                "the decision must follow the sin²θ bound"
            );
        }
    }
}
