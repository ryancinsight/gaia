//! Serpentine tube primitive — planar, multi-pass channel with U-turn bends.
//!
//! A serpentine channel consists of alternating straight legs and 180°
//! circular-arc U-turn bends, all confined to the **XZ plane**.
//! It is the canonical millifluidic mixing channel primitive.
//!
//! ## Coordinate layout (side view, XZ plane)
//!
//! ```text
//!      bend_radius
//!    ◄─────►
//!   ╭──────╮   ← U-turn (180° arc), apex above z = straight_length
//!   │      │
//!   │      │  straight leg (z axis)
//!   │      │
//!   ╰──────╯   ← U-turn, apex below z = 0
//!   │      │
//! ══╧══════╧══ inlet/outlet (Z = 0)
//! x=0      x=2·bend_radius
//! ```
//!
//! ## Centreline definition
//!
//! `pitch = 2 · bend_radius` (centre-to-centre X between adjacent legs).
//!
//! **Straight leg `k`** (k = 0 … `n_passes−1)`:
//! - x = `k · pitch`
//! - even k: travels +Z (z: 0 → `straight_length`)
//! - odd k: travels −Z (z: `straight_length` → 0)
//!
//! **U-turn bend `k`** (after leg k, ψ ∈ [0, π]):
//!
//! *Top bend* (after even leg, CW rotation about +Y):
//! ```text
//! C(ψ) = (x_col + R − R·cos ψ,   0,   L + R·sin ψ)
//! T(ψ) = (sin ψ,  0,  cos ψ)
//! ```
//! *Bottom bend* (after odd leg, CCW rotation about +Y):
//! ```text
//! C(ψ) = (x_col + R − R·cos ψ,   0,   −R·sin ψ)
//! T(ψ) = (sin ψ,  0,  −cos ψ)
//! ```
//!
//! ## Frame convention (no constriction guaranteed)
//!
//! The serpentine is planar in XZ, so **B = +Y** is constant everywhere.
//! Given any unit tangent T = (tx, 0, tz), the unique normal satisfying
//! T × N = +Y is:
//!
//! ```text
//! N = (T.z,  0,  −T.x)
//! ```
//!
//! This formula gives continuous N at every straight↔bend junction:
//!
//! | Section                   | T         | N          |
//! |---------------------------|-----------|------------|
//! | `going_up` straight         | (0,0,+1)  | (+1,0,0)   |
//! | top bend ψ=0              | (0,0,+1)  | (+1,0,0) ✓ |
//! | top bend ψ=π              | (0,0,−1)  | (−1,0,0)   |
//! | `going_down` straight       | (0,0,−1)  | (−1,0,0) ✓ |
//! | bottom bend ψ=0           | (0,0,−1)  | (−1,0,0) ✓ |
//! | bottom bend ψ=π           | (0,0,+1)  | (+1,0,0)   |
//! | `going_up` straight (next)  | (0,0,+1)  | (+1,0,0) ✓ |
//!
//! ## Region IDs
//!
//! | `RegionId` | Surface                               |
//! |----------|---------------------------------------|
//! | 1        | Tube wall (all straight + bend walls) |
//! | 2        | Inlet cap (end of leg 0, Z = 0)       |
//! | 3        | Outlet cap (end of last leg)          |
//!
//! ## Volume (Pappus theorem)
//!
//! ```text
//! V ≈ π r² · L_total
//! L_total = n_passes · straight_length + (n_passes − 1) · π · bend_radius
//! ```
//!
//! ## Uses
//!
//! Millifluidic mixing channels, Dean-vortex enhancers, lab-on-chip reactors,
//! heat-exchanger microchannels.

use std::f64::consts::{PI, TAU};

use super::{PrimitiveError, PrimitiveMesh};
use crate::domain::core::index::{RegionId, VertexId};
use crate::domain::core::scalar::{Point3r, Scalar, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds a planar serpentine tube (multi-pass channel with U-turn bends).
///
/// The channel lies in the **XZ plane**.  Leg 0 starts at the origin and
/// travels in +Z.  Each subsequent leg is offset by `pitch = 2·bend_radius`
/// in +X and reverses direction in Z.
#[derive(Clone, Debug)]
pub struct SerpentineTube {
    /// Tube cross-section radius (mm).  Must be `< bend_radius`.
    pub tube_radius: f64,
    /// Centreline bend radius of each U-turn (mm).
    /// The leg-to-leg centre pitch equals `2 · bend_radius`.
    pub bend_radius: f64,
    /// Length of each straight leg (mm).
    pub straight_length: f64,
    /// Number of straight legs (≥ 2).  `n_passes − 1` U-turn bends connect them.
    pub n_passes: usize,
    /// Angular divisions around the tube cross-section (≥ 3).
    pub tube_segments: usize,
    /// Axial divisions per straight leg (≥ 1).
    pub straight_segments: usize,
    /// Arc divisions per 180° U-turn bend (≥ 3).
    pub bend_segments: usize,
}

impl Default for SerpentineTube {
    fn default() -> Self {
        Self {
            tube_radius: 0.3,
            bend_radius: 1.5,
            straight_length: 8.0,
            n_passes: 4,
            tube_segments: 16,
            straight_segments: 12,
            bend_segments: 24,
        }
    }
}

impl PrimitiveMesh for SerpentineTube {
    fn build(&self) -> Result<IndexedMesh, PrimitiveError> {
        build(self)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal implementation
// ─────────────────────────────────────────────────────────────────────────────

/// A single centreline sampling station.
///
/// Cross-section frame uses **B = +Y** globally (XZ-planar curve).
/// Normal `N = frame_normal(tangent)` so that `T × N = +Y` everywhere.
struct Station {
    centre: Point3r,
    /// Unit tangent in the direction of travel (always in the XZ plane).
    tangent: Vector3r,
}

/// Constant-B=+Y frame normal from tangent.
///
/// For T = (tx, 0, tz): N = (tz, 0, −tx)  →  T × N = (0,1,0) = +Y ✓
#[inline]
fn frame_normal(t: Vector3r) -> Vector3r {
    Vector3r::new(t.z, 0.0, -t.x)
}

/// Sample the straight legs and U-turn bends into one ordered centreline station list.
fn build_serpentine_stations(st: &SerpentineTube) -> Vec<Station> {
    let mut stations = Vec::new();
    let pitch = 2.0 * st.bend_radius;

    for pass in 0..st.n_passes {
        let x_col = f64::from_usize(pass) * pitch;
        let going_up = pass % 2 == 0;
        let tangent = if going_up {
            Vector3r::new(0.0, 0.0, 1.0)
        } else {
            Vector3r::new(0.0, 0.0, -1.0)
        };
        let z_start = if going_up { 0.0 } else { st.straight_length };
        let z_end = if going_up { st.straight_length } else { 0.0 };

        for k in usize::from(pass != 0)..=st.straight_segments {
            let t = f64::from_usize(k) / f64::from_usize(st.straight_segments);
            stations.push(Station {
                centre: Point3r::new(x_col, 0.0, z_start + t * (z_end - z_start)),
                tangent,
            });
        }

        if pass + 1 < st.n_passes {
            for k in 1..=st.bend_segments {
                let psi = PI * f64::from_usize(k) / f64::from_usize(st.bend_segments);
                let (cp, sp) = (psi.cos(), psi.sin());
                let (centre, tangent) = if going_up {
                    (
                        Point3r::new(
                            x_col + st.bend_radius - st.bend_radius * cp,
                            0.0,
                            st.straight_length + st.bend_radius * sp,
                        ),
                        Vector3r::new(sp, 0.0, cp),
                    )
                } else {
                    (
                        Point3r::new(
                            x_col + st.bend_radius - st.bend_radius * cp,
                            0.0,
                            -st.bend_radius * sp,
                        ),
                        Vector3r::new(sp, 0.0, -cp),
                    )
                };
                stations.push(Station { centre, tangent });
            }
        }
    }

    stations
}

/// Add one circular end cap at the current start or end station.
fn add_tube_cap(
    mesh: &mut IndexedMesh,
    station: &Station,
    ring: &[VertexId],
    region: RegionId,
    invert_normal: bool,
) {
    let cap_normal = if invert_normal {
        -station.tangent
    } else {
        station.tangent
    };
    let center = mesh.add_vertex(station.centre, cap_normal);
    for ib in 0..ring.len() {
        let ib1 = (ib + 1) % ring.len();
        if invert_normal {
            mesh.add_face_with_region(center, ring[ib1], ring[ib], region);
        } else {
            mesh.add_face_with_region(center, ring[ib], ring[ib1], region);
        }
    }
}

fn build(st: &SerpentineTube) -> Result<IndexedMesh, PrimitiveError> {
    // ── Parameter validation ──────────────────────────────────────────────────
    if st.tube_radius <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "tube_radius must be > 0, got {}",
            st.tube_radius
        )));
    }
    if st.bend_radius <= st.tube_radius {
        return Err(PrimitiveError::InvalidParam(format!(
            "bend_radius ({}) must be > tube_radius ({})",
            st.bend_radius, st.tube_radius
        )));
    }
    if st.straight_length <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "straight_length must be > 0, got {}",
            st.straight_length
        )));
    }
    if st.n_passes < 2 {
        return Err(PrimitiveError::InvalidParam(format!(
            "n_passes must be >= 2, got {}",
            st.n_passes
        )));
    }
    if st.tube_segments < 3 {
        return Err(PrimitiveError::TooFewSegments(st.tube_segments));
    }
    if st.straight_segments < 1 {
        return Err(PrimitiveError::InvalidParam(
            "straight_segments must be >= 1".into(),
        ));
    }
    if st.bend_segments < 3 {
        return Err(PrimitiveError::InvalidParam(
            "bend_segments must be >= 3".into(),
        ));
    }

    let wall_region = RegionId::new(1);
    let inlet_region = RegionId::new(2);
    let outlet_region = RegionId::new(3);

    let mut mesh = IndexedMesh::new();
    let r = st.tube_radius;
    let ns = st.tube_segments;
    let all_stations = build_serpentine_stations(st);

    // ── Build vertex rings ────────────────────────────────────────────────────
    // B = (0,1,0) fixed.  N = frame_normal(T).
    // Cross-section surface normal: n_out(beta) = cos(beta)·N + sin(beta)·B
    let b_fixed = Vector3r::y();
    let n_stations = all_stations.len();
    let mut rings: Vec<Vec<VertexId>> = Vec::with_capacity(n_stations);

    for station in &all_stations {
        let n_frame = frame_normal(station.tangent);
        let row: Vec<VertexId> = (0..ns)
            .map(|ib| {
                let beta = f64::from_usize(ib) / f64::from_usize(ns) * TAU;
                let n_out = n_frame * beta.cos() + b_fixed * beta.sin();
                let pos = station.centre + n_out * r;
                mesh.add_vertex(pos, n_out)
            })
            .collect();
        rings.push(row);
    }

    // ── Lateral tube wall ─────────────────────────────────────────────────────
    for ia in 0..n_stations - 1 {
        for ib in 0..ns {
            let ib1 = (ib + 1) % ns;
            let v00 = rings[ia][ib];
            let v10 = rings[ia + 1][ib];
            let v11 = rings[ia + 1][ib1];
            let v01 = rings[ia][ib1];
            mesh.add_face_with_region(v00, v01, v11, wall_region);
            mesh.add_face_with_region(v00, v11, v10, wall_region);
        }
    }

    // ── Inlet cap (station 0, outward normal = −T₀) ───────────────────────────
    add_tube_cap(&mut mesh, &all_stations[0], &rings[0], inlet_region, true);

    // ── Outlet cap (last station, outward normal = +T_last) ───────────────────
    add_tube_cap(
        &mut mesh,
        &all_stations[n_stations - 1],
        &rings[n_stations - 1],
        outlet_region,
        false,
    );

    Ok(mesh)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::watertight::check::check_watertight;
    use crate::infrastructure::storage::edge_store::EdgeStore;
    use std::f64::consts::PI;

    fn watertight_report(
        st: &SerpentineTube,
    ) -> crate::application::watertight::check::WatertightReport {
        let mesh = st.build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        check_watertight(&mesh.vertices, &mesh.faces, &edges)
    }

    #[test]
    fn serpentine_default_is_watertight() {
        let report = watertight_report(&SerpentineTube::default());
        assert!(report.is_watertight, "serpentine must be watertight");
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn serpentine_volume_positive_and_pappus() {
        // Pappus: V ≈ pi * r^2 * (n_passes * sl + (n_passes-1) * pi * big_r)
        let st = SerpentineTube {
            tube_radius: 0.3,
            bend_radius: 1.5,
            straight_length: 8.0,
            n_passes: 4,
            tube_segments: 32,
            straight_segments: 24,
            bend_segments: 48,
        };
        let report = watertight_report(&st);
        assert!(report.signed_volume > 0.0, "volume must be positive");

        let r = st.tube_radius;
        let n_passes = st.n_passes as f64;
        let l_total = n_passes * st.straight_length + (n_passes - 1.0) * PI * st.bend_radius;
        let expected = PI * r * r * l_total;
        let error = (report.signed_volume - expected).abs() / expected;
        assert!(
            error < 0.02,
            "Pappus volume error {:.4}% should be < 2% at high resolution",
            error * 100.0
        );
    }

    #[test]
    fn serpentine_two_passes_is_watertight() {
        let report = watertight_report(&SerpentineTube {
            n_passes: 2,
            tube_segments: 12,
            straight_segments: 6,
            bend_segments: 12,
            ..SerpentineTube::default()
        });
        assert!(report.is_watertight);
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn serpentine_six_passes_is_watertight() {
        let report = watertight_report(&SerpentineTube {
            n_passes: 6,
            ..SerpentineTube::default()
        });
        assert!(report.is_watertight);
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn serpentine_rejects_invalid_params() {
        assert!(SerpentineTube {
            tube_radius: 0.0,
            ..SerpentineTube::default()
        }
        .build()
        .is_err());
        assert!(SerpentineTube {
            bend_radius: 0.2,
            tube_radius: 0.3,
            ..SerpentineTube::default()
        }
        .build()
        .is_err());
        assert!(SerpentineTube {
            straight_length: 0.0,
            ..SerpentineTube::default()
        }
        .build()
        .is_err());
        assert!(SerpentineTube {
            n_passes: 1,
            ..SerpentineTube::default()
        }
        .build()
        .is_err());
        assert!(SerpentineTube {
            tube_segments: 2,
            ..SerpentineTube::default()
        }
        .build()
        .is_err());
        assert!(SerpentineTube {
            bend_segments: 2,
            ..SerpentineTube::default()
        }
        .build()
        .is_err());
    }
}
