//! # Mesh Element Keys
//!
//! Two independent index systems coexist:
//!
//! ## New: `SlotMap` generational keys (`*Key` types)
//!
//! Used by [`crate::domain::mesh::HalfEdgeMesh`] and [`crate::domain::mesh::with_mesh`].
//! Each key is an 8-byte generational index; stale keys (after element
//! deletion) return `None` rather than silently aliasing a new element.
//!
//! ## Legacy: u32 newtype indices (`*Id` types)
//!
//! Used by [`crate::infrastructure::storage::VertexPool`],
//! [`crate::infrastructure::storage::FaceStore`], and
//! [`crate::infrastructure::storage::EdgeStore`] — all of which are contiguous
//! `Vec`-backed stores indexed by a plain `u32`.
//!
//! All `*Id` types are `#[repr(transparent)]` over `u32`, guaranteeing that
//! `bytemuck` transmutation and FFI use are sound.
//!
//! # Key Invariant
//!
//! `*Key` values are only valid within the `HalfEdgeMesh<'id>` that created
//! them.  Mixing keys from different meshes is a logic error (returns `None`,
//! never undefined behaviour).
//!
//! # Narrowing casts
//!
//! `from_usize` casts `usize → u32` with a saturating `as u32` — valid for
//! meshes with < 4Gi elements.  When element counts may exceed `u32::MAX` use
//! `TryFrom<usize>` which returns `Err` instead of wrapping.

use slotmap::new_key_type;
use std::fmt;

// ── SlotMap generational keys (HalfEdgeMesh) ─────────────────────────────────

new_key_type! {
    /// Key for a vertex in a [`crate::domain::mesh::HalfEdgeMesh`].
    ///
    /// # Invariant
    /// Valid only within the mesh that produced it.
    /// Stale keys (after vertex deletion) return `None` on lookup.
    pub struct VertexKey;
}

new_key_type! {
    /// Key for a directed half-edge in a [`crate::domain::mesh::HalfEdgeMesh`].
    ///
    /// # Invariant
    /// `twin(twin(he)) == he` — the twin relationship is an involution.
    /// See [`crate::domain::topology::halfedge`] for the full invariant set.
    pub struct HalfEdgeKey;
}

new_key_type! {
    /// Key for a face (polygon) in a [`crate::domain::mesh::HalfEdgeMesh`].
    pub struct FaceKey;
}

new_key_type! {
    /// Key for a named CFD boundary patch in a [`crate::domain::mesh::HalfEdgeMesh`].
    pub struct PatchKey;
}

new_key_type! {
    /// Key for a mesh region (channel segment, junction) in a
    /// [`crate::domain::mesh::HalfEdgeMesh`].
    pub struct RegionKey;
}

// ── Legacy u32 index types (IndexedMesh / VertexPool / FaceStore / EdgeStore) ─

/// Strongly-typed vertex index for [`crate::infrastructure::storage::VertexPool`].
///
/// A plain `u32` index into a contiguous vertex array.  Use [`VertexKey`] for
/// the new [`crate::domain::mesh::HalfEdgeMesh`]-based API.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct VertexId(pub u32);

impl VertexId {
    /// Create from a raw `u32` index.
    #[inline]
    #[must_use]
    pub fn new(raw: u32) -> Self {
        VertexId(raw)
    }
    /// Create from a `usize` index.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds `u32::MAX`.
    #[inline]
    #[must_use]
    pub fn from_usize(n: usize) -> Self {
        Self::try_from(n).expect("Index exceeds u32::MAX")
    }
    /// Return the raw `u32` index.
    #[inline]
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }
    /// Return as `usize`.
    #[inline]
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<usize> for VertexId {
    type Error = std::num::TryFromIntError;

    #[inline]
    fn try_from(n: usize) -> Result<Self, Self::Error> {
        u32::try_from(n).map(Self)
    }
}

impl fmt::Display for VertexId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strongly-typed face index for [`crate::infrastructure::storage::FaceStore`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct FaceId(pub u32);

impl FaceId {
    /// Create from a `usize` index.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds `u32::MAX`.
    #[inline]
    #[must_use]
    pub fn from_usize(n: usize) -> Self {
        Self::try_from(n).expect("Index exceeds u32::MAX")
    }
    /// Return as `usize`.
    #[inline]
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<usize> for FaceId {
    type Error = std::num::TryFromIntError;

    #[inline]
    fn try_from(n: usize) -> Result<Self, Self::Error> {
        u32::try_from(n).map(Self)
    }
}

impl fmt::Display for FaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strongly-typed edge index for [`crate::infrastructure::storage::EdgeStore`].
///
/// Indexes the flattened edge list in `EdgeStore`; distinct from
/// [`HalfEdgeKey`] which indexes the directed half-edges of `HalfEdgeMesh`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct EdgeId(pub u32);

impl EdgeId {
    /// Create from a `usize` index.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds `u32::MAX`.
    #[inline]
    #[must_use]
    pub fn from_usize(n: usize) -> Self {
        Self::try_from(n).expect("Index exceeds u32::MAX")
    }
    /// Return as `usize`.
    #[inline]
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<usize> for EdgeId {
    type Error = std::num::TryFromIntError;

    #[inline]
    fn try_from(n: usize) -> Result<Self, Self::Error> {
        u32::try_from(n).map(Self)
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strongly-typed region index for [`crate::infrastructure::storage::FaceStore`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct RegionId(pub u32);

impl RegionId {
    /// Sentinel value indicating "no region assigned".
    pub const INVALID: Self = Self(u32::MAX);
    /// Create from a raw `u32` index.
    #[inline]
    #[must_use]
    pub fn new(raw: u32) -> Self {
        RegionId(raw)
    }
    /// Create from a `usize` index.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds `u32::MAX`.
    #[inline]
    #[must_use]
    pub fn from_usize(n: usize) -> Self {
        Self::try_from(n).expect("Index exceeds u32::MAX")
    }
    /// Return as `usize`.
    #[inline]
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<usize> for RegionId {
    type Error = std::num::TryFromIntError;

    #[inline]
    fn try_from(n: usize) -> Result<Self, Self::Error> {
        u32::try_from(n).map(Self)
    }
}

impl fmt::Display for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::SlotMap;

    #[test]
    fn vertex_key_generational_safety() {
        let mut map: SlotMap<VertexKey, u32> = SlotMap::with_key();
        let k = map.insert(42);
        assert_eq!(map[k], 42);
        map.remove(k);
        // Stale key returns None — no panic, no silent aliasing
        assert!(map.get(k).is_none());
    }

    #[test]
    fn vertex_id_roundtrip() {
        let id = VertexId::new(7);
        assert_eq!(id.raw(), 7);
        assert_eq!(id.as_usize(), 7);
        assert_eq!(VertexId::from_usize(42).as_usize(), 42);
    }

    #[test]
    fn region_id_invalid_sentinel() {
        assert_ne!(RegionId::INVALID, RegionId::from_usize(0));
        assert_eq!(RegionId::INVALID.0, u32::MAX);
    }
}
