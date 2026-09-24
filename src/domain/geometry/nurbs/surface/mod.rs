//! NURBS and B-Spline Surfaces -- tensor-product parameterisation.
//!
//! `BSplineSurface` is the non-rational case (all weights == 1).
//! `NurbsSurface` is the rational case with per-control-point positive weights.
//! Both are parameterised over a rectangular domain `[u0,u1] x [v0,v1]` and
//! generic over the scalar seam `T` with an `f64` default.

mod bspline;
mod error;
mod grid;
mod rational;
mod validate;

#[cfg(test)]
mod tests;

pub use bspline::BSplineSurface;
pub use error::SurfaceError;
pub use grid::{ControlGrid, WeightGrid};
pub use rational::NurbsSurface;
