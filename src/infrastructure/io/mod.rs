//! Mesh I/O: STL, OBJ, PLY, 3MF, glTF/GLB, DXF, VTK, `OpenFOAM`, and CFDrs scheme import.
//!
//! All format modules are built unconditionally: formats with external
//! dependencies (STL, 3MF, VTK, scheme) declare those dependencies directly,
//! and formats with none (OBJ, PLY, glTF, DXF, OpenFOAM) need no gate.

/// Field-level primitives shared by the ASCII importers.
///
/// Crate-internal: this is an implementation detail of `obj` and `ply`, not
/// public API.
pub(crate) mod parse;

pub mod stl;

pub mod openfoam;

pub mod obj;

pub mod ply;

pub mod gltf_export;

pub mod dxf;

pub mod three_mf;

pub mod vtk;
