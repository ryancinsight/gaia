//! Hex-to-Tet mesh decomposition
//!
//! Robust decomposition of 8-node hexahedra into tetrahedra.
//!
//! This module is a **volume/FEM tool** --- intentional `Mesh<T>` usage for
//! hexahedral/tetrahedral cell topology.

mod convert;
mod decompose;
mod types;

#[cfg(test)]
mod tests;

/// Converter for decomposing hexahedral meshes into tetrahedral ones
pub struct HexToTetConverter;
