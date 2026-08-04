//! Watertight checking.
//!
//! Validates a closed triangle mesh using four independent criteria:
//!
//! 1. **Manifold + closed**: every edge is shared by exactly 2 faces.
//! 2. **Euler characteristic**: `V - E + F = 2` for a genus-0 closed sphere.
//!    A torus gives `V - E + F = 0`, etc. Mismatches reveal topological defects.
//! 3. **Orientation consistency**: all face pairs sharing an edge have opposite
//!    directed-edge orientations (no two adjacent faces wind the same way).
//! 4. **Positive signed volume**: the divergence-theorem volume integral must
//!    be finite and positive for an outward-oriented mesh.
//!
//! ## Euler's Theorem
//!
//! For a convex polyhedron (or any genus-0 closed surface):
//!
//! $$V - E + F = 2 \cdot (1 - g)$$
//!
//! where $g$ is the genus (number of handles). For a sphere or cube $g = 0$
//! so the characteristic is 2. A torus has $g = 1$ so the characteristic is 0.
//!
//! For a triangle mesh: $E = 3F/2$ (each face contributes 3 half-edges, each
//! edge is shared by exactly 2 faces in a manifold), so the relation reduces
//! to $V - E + F = 2$ for a closed manifold of genus 0.

use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::measure;
use crate::domain::topology::manifold;
use crate::domain::topology::orientation;
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::infrastructure::storage::face_store::FaceStore;
use crate::infrastructure::storage::vertex_pool::VertexPool;

/// Comprehensive watertight status report.
#[derive(Clone, Debug)]
pub struct WatertightReport {
    /// Is the mesh a closed 2-manifold (no boundary edges)?
    pub is_closed: bool,
    /// Number of boundary edges.
    pub boundary_edge_count: usize,
    /// Number of non-manifold edges.
    pub non_manifold_edge_count: usize,
    /// Is orientation consistent?
    pub orientation_consistent: bool,
    /// Signed volume (should be positive for outward-oriented mesh).
    pub signed_volume: f64,
    /// Is the mesh closed, consistently oriented, and outward-oriented?
    pub is_watertight: bool,
    /// Euler characteristic $\chi = V - E + F$.
    ///
    /// - `2` for a closed sphere-topology surface (genus 0)
    /// - `0` for a torus (genus 1)
    /// - Negative values indicate complex topology or mesh defects
    ///
    /// `None` when vertices/edges/faces counts are not available.
    pub euler_characteristic: Option<i64>,
    /// Reference Euler characteristic for a closed genus-0 manifold.
    ///
    /// This is diagnostic metadata, not a watertightness requirement: valid
    /// meshes may have handles (for example, a torus has characteristic 0).
    pub euler_expected: i64,
}

/// Check if a mesh is watertight.
#[must_use]
pub fn check_watertight<T: Scalar>(
    vertex_pool: &VertexPool<T>,
    face_store: &FaceStore,
    edge_store: &EdgeStore,
) -> WatertightReport {
    let manifold_report = manifold::check_manifold(edge_store);
    let orientation_ok = orientation::check_orientation(face_store, edge_store).is_ok();

    // Compute signed volume
    let signed_vol = measure::total_signed_volume(face_store.iter_enumerated().map(|(_, face)| {
        (
            vertex_pool.position(face.vertices[0]),
            vertex_pool.position(face.vertices[1]),
            vertex_pool.position(face.vertices[2]),
        )
    }));
    let signed_vol_f64 = eunomia::NumericElement::to_f64(signed_vol);

    // Euler characteristic: V - E + F = 2 for a closed genus-0 manifold.
    // Delegated to the canonical SSOT implementation (counts only referenced
    // vertices, uses edge_store.len() O(1) for E).
    let euler = euler_chi_from_stores(face_store, edge_store);

    let is_closed = manifold_report.is_closed_manifold;

    WatertightReport {
        is_closed,
        boundary_edge_count: manifold_report.boundary_edges,
        non_manifold_edge_count: manifold_report.non_manifold_edges,
        orientation_consistent: orientation_ok,
        signed_volume: signed_vol_f64,
        is_watertight: is_closed
            && orientation_ok
            && signed_vol_f64.is_finite()
            && signed_vol_f64 > 0.0,
        euler_characteristic: Some(euler),
        euler_expected: 2,
    }
}

/// Assert the mesh is watertight, returning an error if not.
pub fn assert_watertight<T: Scalar>(
    vertex_pool: &VertexPool<T>,
    face_store: &FaceStore,
    edge_store: &EdgeStore,
) -> MeshResult<WatertightReport> {
    let report = check_watertight(vertex_pool, face_store, edge_store);
    if !report.is_closed {
        return Err(MeshError::NotWatertight {
            count: report.boundary_edge_count,
        });
    }
    if !report.orientation_consistent {
        return Err(
            orientation::check_orientation(face_store, edge_store).expect_err(
                "invariant: orientation report marked an inconsistent mesh as consistent",
            ),
        );
    }
    if !report.signed_volume.is_finite() || report.signed_volume <= 0.0 {
        return Err(MeshError::Other(format!(
            "mesh is not outward-oriented: signed volume {}",
            report.signed_volume
        )));
    }
    Ok(report)
}

/// Compute the Euler characteristic `V - E + F` from a pre-built `EdgeStore`.
///
/// # Complexity
/// - **Time**: O(F) for the vertex-reference scan; O(1) for E and F lookups.
/// - **Memory**: one `HashSet<VertexId>` of size ≤ 3F.
///
/// This is the canonical SSOT implementation. Prefer this over reinventing
/// edge/vertex counting inline — the two-HashSet pattern used before this
/// function was introduced allocated O(F) extra and counted edges in O(F)
/// instead of using `edge_store.len()` which is already O(1).
///
/// # Note
/// `vertex_pool.len()` is **not** used because it includes dead/welded entries
/// from CSG input meshes that inflate V incorrectly. Only referenced vertices
/// (those appearing in at least one face) are counted.
#[inline]
#[must_use]
pub fn euler_chi_from_stores(face_store: &FaceStore, edge_store: &EdgeStore) -> i64 {
    let mut referenced: hashbrown::HashSet<crate::domain::core::index::VertexId> =
        hashbrown::HashSet::with_capacity(face_store.len() * 3 / 2);
    for face in face_store.iter() {
        referenced.insert(face.vertices[0]);
        referenced.insert(face.vertices[1]);
        referenced.insert(face.vertices[2]);
    }
    let v = referenced.len() as i64;
    let e = edge_store.len() as i64;
    let f = face_store.len() as i64;
    v - e + f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::geometry::primitives::{Cube, PrimitiveMesh, Torus};

    #[test]
    fn globally_inverted_closed_mesh_is_not_outward_watertight() {
        let mut mesh = Cube::unit().build().expect("unit cube");
        mesh.flip_faces();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);

        assert!(report.is_closed);
        assert!(report.orientation_consistent);
        assert!(report.signed_volume < 0.0);
        assert!(!report.is_watertight);
        let error = assert_watertight(&mesh.vertices, &mesh.faces, &edges)
            .expect_err("inverted mesh must fail outward-orientation validation");
        assert!(
            matches!(error, MeshError::Other(message) if message.contains("not outward-oriented") && message.contains('-'))
        );
    }

    #[test]
    fn assertion_accepts_valid_non_spherical_topology() {
        let mesh = Torus::default().build().expect("torus");
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = assert_watertight(&mesh.vertices, &mesh.faces, &edges)
            .expect("closed oriented torus is watertight");

        assert_eq!(report.euler_characteristic, Some(0));
        assert!(report.is_watertight);
    }
}
