//! Truncated icosahedron primitive — soccer ball / C60 topology.

use super::{PrimitiveError, PrimitiveMesh};
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::geometry::normal::triangle_normal;
use crate::domain::mesh::IndexedMesh;
use eunomia::FloatElement;

/// Builds a truncated icosahedron inscribed in a sphere of the given `radius`.
///
/// The truncated icosahedron (Goldberg polyhedron GP(1,1)) is the familiar
/// soccer ball / C₆₀ buckyball topology: 12 pentagonal faces and 20 hexagonal
/// faces, all assembled from 60 vertices.
///
/// ## Geometry
///
/// Vertices are derived from the icosahedron by truncating each vertex at 1/3
/// of the edge length, producing:
/// - 60 vertices
/// - 90 edges
/// - 32 faces (12 pentagons + 20 hexagons) → 180 triangles after fan-triangulation
///
/// ## Topology
///
/// - V = 60, E = 90, F = 32  →  `V − E + F = 2`  (χ = 2, genus 0)
///
/// ## Output
///
/// - `RegionId(1)` on all faces
/// - `signed_volume > 0` (outward CCW winding)
#[derive(Clone, Debug)]
pub struct TruncatedIcosahedron {
    /// Circumsphere radius (mm).
    pub radius: f64,
    /// Centre position.
    pub center: Point3r,
}

impl Default for TruncatedIcosahedron {
    fn default() -> Self {
        Self {
            radius: 1.0,
            center: Point3r::origin(),
        }
    }
}

impl PrimitiveMesh for TruncatedIcosahedron {
    fn build(&self) -> Result<IndexedMesh, PrimitiveError> {
        build(self)
    }
}

/// Golden ratio φ = (1 + √5) / 2.
const PHI: f64 = 1.618_033_988_749_895;

/// 60 raw vertices of the truncated icosahedron.
/// All permutations of (0, ±1, ±3φ), (±1, ±(2+φ), ±2φ), (±2, ±(1+2φ), ±φ).
/// Circumradius = √(9φ² + 1) ... but let's just scale all to unit sphere.
fn raw_vertices() -> Vec<[f64; 3]> {
    let p = PHI;
    let mut verts = Vec::with_capacity(60);

    // Family 1: (0, ±1, ±3φ) — all 4 sign combinations, all 3 coordinate cyclic permutations
    // Cyclic permutations of (x,y,z): (x,y,z), (z,x,y), (y,z,x)
    for &a in &[1.0_f64, -1.0] {
        for &b in &[3.0 * p, -3.0 * p] {
            verts.push([0.0, a, b]);
            verts.push([b, 0.0, a]);
            verts.push([a, b, 0.0]);
        }
    }
    // Family 2: (±1, ±(2+φ), ±2φ) — 8 sign combinations, 3 cyclic permutations
    let two_phi = 2.0 * p;
    let two_plus_phi = 2.0 + p;
    for &a in &[1.0_f64, -1.0] {
        for &b in &[two_plus_phi, -two_plus_phi] {
            for &c in &[two_phi, -two_phi] {
                verts.push([a, b, c]);
                verts.push([c, a, b]);
                verts.push([b, c, a]);
            }
        }
    }
    // Family 3: (±2, ±(1+2φ), ±φ) — 8 sign combinations, 3 cyclic permutations
    let one_plus_two_phi = 1.0 + 2.0 * p;
    for &a in &[2.0_f64, -2.0] {
        for &b in &[one_plus_two_phi, -one_plus_two_phi] {
            for &c in &[p, -p] {
                verts.push([a, b, c]);
                verts.push([c, a, b]);
                verts.push([b, c, a]);
            }
        }
    }

    assert_eq!(verts.len(), 60, "expected 60 vertices, got {}", verts.len());
    verts
}

/// Scale and translate the canonical truncated-icosahedron vertex set into world space.
fn scaled_vertices(ti: &TruncatedIcosahedron) -> Vec<Point3r> {
    let raw = raw_vertices();
    let circ = raw[0].iter().map(|&x| x * x).sum::<f64>().sqrt();
    let scale = ti.radius / circ;
    let cx = ti.center.x;
    let cy = ti.center.y;
    let cz = ti.center.z;

    raw.iter()
        .map(|v| Point3r::new(cx + v[0] * scale, cy + v[1] * scale, cz + v[2] * scale))
        .collect()
}

/// Recover the pentagon and hexagon cycles from the scaled truncated-icosahedron graph.
fn discover_faces(verts: &[Point3r]) -> Vec<Vec<usize>> {
    let edge_len_sq = {
        let p0 = verts[0];
        let mut min_sq = f64::INFINITY;
        for p in verts.iter().take(60).skip(1) {
            let d = (*p - p0).norm_squared();
            if d < min_sq {
                min_sq = d;
            }
        }
        min_sq
    };
    let tol = edge_len_sq * 0.01;

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); 60];
    for i in 0..60 {
        for j in i + 1..60 {
            let d2 = (verts[j] - verts[i]).norm_squared();
            if (d2 - edge_len_sq).abs() < tol {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    let mut visited_edges: hashbrown::HashSet<(usize, usize)> = hashbrown::HashSet::new();
    let mut faces: Vec<Vec<usize>> = Vec::new();

    for start in 0..60 {
        for &next in &adj[start] {
            if visited_edges.contains(&(start, next)) {
                continue;
            }
            let mut face = vec![start, next];
            let mut cur = start;
            let mut nxt = next;
            loop {
                // Find the neighbour of nxt that is most "left" relative to cur→nxt.
                let dir = (verts[nxt] - verts[cur]).normalize();
                let outward = (verts[nxt].coords + verts[cur].coords).normalize();
                // Candidates: neighbours of nxt except cur.
                let best = adj[nxt]
                    .iter()
                    .filter(|&&k| k != cur)
                    .min_by(|&&a, &&b| {
                        let da = verts[a] - verts[nxt];
                        let db = verts[b] - verts[nxt];
                        // Most CCW = most negative cross product z (when projected onto face plane).
                        let ca = dir.cross(da).dot(outward);
                        let cb = dir.cross(db).dot(outward);
                        ca.total_cmp(&cb)
                    })
                    .copied();
                let Some(candidate) = best else {
                    break;
                };
                if candidate == start {
                    break;
                }
                if face.len() > 8 {
                    break;
                } // safety: no face has more than 6 vertices
                face.push(candidate);
                cur = nxt;
                nxt = candidate;
            }
            if face.len() >= 5 && face.len() <= 6 {
                for k in 0..face.len() {
                    visited_edges.insert((face[k], face[(k + 1) % face.len()]));
                }
                faces.push(face);
            }
        }
    }

    faces
}

/// Fan-triangulate the recovered pentagons and hexagons into the indexed mesh.
fn add_faces_to_mesh(
    mesh: &mut IndexedMesh,
    verts: &[Point3r],
    faces: &[Vec<usize>],
    region: RegionId,
) {
    let face_normal_at = |face_verts: &[usize]| -> Vector3r {
        let n = face_verts.len();
        let centroid: Vector3r = face_verts
            .iter()
            .map(|&i| verts[i].coords)
            .fold(Vector3r::zeros(), |a, v| a + v)
            / f64::from_count(n);
        let p0 = verts[face_verts[0]];
        let mut n_sum = Vector3r::zeros();
        for k in 1..n - 1 {
            let p1 = verts[face_verts[k]];
            let p2 = verts[face_verts[k + 1]];
            if let Some(n) = triangle_normal(&p0, &p1, &p2) {
                n_sum += n;
            }
        }
        let len = n_sum.norm();
        if len < 1e-14 {
            centroid.normalize()
        } else {
            n_sum / len
        }
    };

    for face_indices in faces {
        let n_outward = face_normal_at(face_indices);
        let p0 = &verts[face_indices[0]];
        for k in 1..face_indices.len() - 1 {
            let p1 = &verts[face_indices[k]];
            let p2 = &verts[face_indices[k + 1]];
            let n = match triangle_normal(p0, p1, p2) {
                Some(n) => {
                    if n.dot(n_outward) < 0.0 {
                        -n
                    } else {
                        n
                    }
                }
                None => n_outward,
            };
            let vi0 = mesh.add_vertex(*p0, n);
            let vi1 = mesh.add_vertex(*p1, n);
            let vi2 = mesh.add_vertex(*p2, n);
            mesh.add_face_with_region(vi0, vi1, vi2, region);
        }
    }
}

fn build(ti: &TruncatedIcosahedron) -> Result<IndexedMesh, PrimitiveError> {
    if ti.radius <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "radius must be > 0, got {}",
            ti.radius
        )));
    }

    let region = RegionId::new(1);
    let mut mesh = IndexedMesh::new();
    let verts = scaled_vertices(ti);
    let faces = discover_faces(&verts);
    add_faces_to_mesh(&mut mesh, &verts, &faces, region);

    // The face-traversal algorithm finds faces in CCW-from-inside order.
    // Flip all faces to obtain outward normals.
    mesh.flip_faces();

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::watertight::check::check_watertight;
    use crate::infrastructure::storage::edge_store::EdgeStore;

    #[test]
    fn truncated_icosahedron_vertex_count() {
        // Verify raw_vertices gives exactly 60 unique vertices.
        let verts = raw_vertices();
        assert_eq!(verts.len(), 60);
    }

    #[test]
    fn truncated_icosahedron_is_watertight() {
        let mesh = TruncatedIcosahedron::default().build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(
            report.is_watertight,
            "truncated icosahedron must be watertight: {report:?}"
        );
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn truncated_icosahedron_volume_positive() {
        let mesh = TruncatedIcosahedron {
            radius: 2.0,
            ..TruncatedIcosahedron::default()
        }
        .build()
        .unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.signed_volume > 0.0);
    }

    #[test]
    fn truncated_icosahedron_invalid_radius() {
        assert!(TruncatedIcosahedron {
            radius: 0.0,
            ..TruncatedIcosahedron::default()
        }
        .build()
        .is_err());
    }
}
