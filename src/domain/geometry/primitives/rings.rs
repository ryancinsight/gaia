//! Cross-section ring tables for the swept primitives.
//!
//! A primitive that sweeps a cross-section places one ring of vertex ids per
//! station, and every ring of a sweep holds the same number of vertices. The
//! table keeps the rings in one contiguous buffer with a fixed stride, so a
//! sweep costs one allocation instead of one per station, and the wall
//! builders read adjacent rings from adjacent memory.
//!
//! The face builders that consume a ring live here as well. The tube
//! primitives each carried their own copy of the same wall-quad and end-cap
//! loops; one owner keeps their winding conventions in step.

use crate::domain::core::index::{RegionId, VertexId};
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// The direction a cap's fan winds around its ring.
///
/// The two variants exist because a cap's fan runs one way around the tube's
/// inlet and the other around its outlet, and the choice flips the cap's
/// normal; a bare flag at the call site hid which of the two a call wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapWinding {
    /// The fan visits the ring's vertices in the order the ring lists them.
    Forward,
    /// The fan visits the ring's vertices in the reverse of that order.
    Reversed,
}

/// A table of equal-length vertex rings in one contiguous buffer.
///
/// Ring `r` holds `stride` vertices at `vertices[r * stride..][..stride]`, so
/// the table is a single allocation and adjacent rings are adjacent in
/// memory.
#[derive(Debug, Clone)]
pub(crate) struct VertexRings {
    /// The ring vertices, ring-major with a fixed stride.
    vertices: Vec<VertexId>,
    /// The number of rings.
    rows: usize,
    /// The number of vertices in each ring.
    stride: usize,
}

impl VertexRings {
    /// A table of `rows` rings, each `stride` vertices long.
    pub(crate) fn with_rows(rows: usize, stride: usize) -> Self {
        let cells = rows
            .checked_mul(stride)
            .expect("invariant: a ring table's cell count fits a usize");
        Self {
            vertices: vec![VertexId::default(); cells],
            rows,
            stride,
        }
    }

    /// The number of rings in the table.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// The number of vertices in each ring.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// The vertices of ring `index`.
    pub(crate) fn ring(&self, index: usize) -> &[VertexId] {
        debug_assert!(index < self.rows, "ring index within the table");
        &self.vertices[index * self.stride..(index + 1) * self.stride]
    }

    /// Fill ring `ring` with the vertex `fill` returns for each of its slots.
    pub(crate) fn fill_ring(&mut self, ring: usize, mut fill: impl FnMut(usize) -> VertexId) {
        debug_assert!(ring < self.rows, "ring index within the table");
        let start = ring * self.stride;
        for slot in 0..self.stride {
            self.vertices[start + slot] = fill(slot);
        }
    }
}

/// Connect adjacent rings with outward-oriented wall quads.
///
/// Ring `r` and ring `r + 1` are joined by two triangles per slot, so the
/// wall of a sweep of `rows` rings carries `2 * (rows - 1) * stride` faces.
pub(crate) fn add_ring_wall_quads(mesh: &mut IndexedMesh, rings: &VertexRings, region: RegionId) {
    let ns = rings.stride();
    for ring in 0..rings.rows() - 1 {
        let lower = rings.ring(ring);
        let upper = rings.ring(ring + 1);
        for slot in 0..ns {
            let next = (slot + 1) % ns;
            let v00 = lower[slot];
            let v10 = upper[slot];
            let v11 = upper[next];
            let v01 = lower[next];
            mesh.add_face_with_region(v00, v01, v11, region);
            mesh.add_face_with_region(v00, v11, v10, region);
        }
    }
}

/// Close one end of a ring with a fan cap.
///
/// `center` is the cap centre and `normal` its outward normal. `winding`
/// selects the direction the fan runs around the ring, which is what fixes
/// the cap's face normal against the tube's direction of travel.
pub(crate) fn add_ring_end_cap(
    mesh: &mut IndexedMesh,
    ring: &[VertexId],
    center: Point3r,
    normal: Vector3r,
    region: RegionId,
    winding: CapWinding,
) {
    let vc = mesh.add_vertex(center, normal);
    for slot in 0..ring.len() {
        let next = (slot + 1) % ring.len();
        match winding {
            CapWinding::Forward => {
                mesh.add_face_with_region(vc, ring[slot], ring[next], region);
            }
            CapWinding::Reversed => {
                mesh.add_face_with_region(vc, ring[next], ring[slot], region);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_table_is_one_contiguous_buffer() {
        let mut rings = VertexRings::with_rows(3, 4);
        assert_eq!(rings.rows(), 3);
        assert_eq!(rings.stride(), 4);

        for ring in 0..3 {
            rings.fill_ring(ring, |slot| {
                VertexId::new(u32::try_from(ring * 4 + slot).expect("slot index fits a u32"))
            });
        }

        // The rings are adjacent runs of one buffer, not one allocation each.
        let stride_bytes = std::mem::size_of::<VertexId>() * 4;
        assert_eq!(
            rings.ring(1).as_ptr().addr() - rings.ring(0).as_ptr().addr(),
            stride_bytes
        );
        assert_eq!(
            rings.ring(2).as_ptr().addr() - rings.ring(1).as_ptr().addr(),
            stride_bytes
        );
        assert_eq!(rings.ring(0).len(), 4);

        // Each slot holds the id its ring was filled with.
        assert_eq!(rings.ring(0)[0], VertexId::new(0));
        assert_eq!(rings.ring(1)[3], VertexId::new(7));
        assert_eq!(rings.ring(2)[2], VertexId::new(10));
    }
}
