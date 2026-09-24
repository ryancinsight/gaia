//! # NURBS and B-Spline Curves
//!
//! Provides `BSplineCurve` (non-rational) and `NurbsCurve` (rational) for any
//! embedding dimension.  Dimension `D` is a const-generic type parameter
//! so all allocations are stack-based for D ≤ 4, and `T` is the
//! [`crate::domain::core::scalar::Scalar`] precision seam with an
//! `f64` default, so existing callers compile without annotation.
//!
//! ## Mathematical Foundation
//!
//! ### B-Spline Curve (non-rational)
//!
//! ```text
//! C(t) = Σᵢ N_{i,p}(t) · Pᵢ
//! ```
//!
//! where `N_{i,p}` are the B-spline basis functions computed by Cox–de Boor
//! recursion (see [`super::basis`]).
//!
//! ### NURBS Curve (rational)
//!
//! ```text
//!        Σᵢ N_{i,p}(t) · wᵢ · Pᵢ
//! C(t) = ─────────────────────────
//!           Σᵢ N_{i,p}(t) · wᵢ
//! ```
//!
//! ### Theorem — Partition of Unity
//!
//! For any parameter value `t` in the domain:
//! ```text
//! Σᵢ N_{i,p}(t) = 1
//! ```
//! This ensures that a B-spline curve is an affine combination of its control
//! points and that translations and rotations of the control polygon map
//! exactly to the curve.
//!
//! ### Theorem — Convex Hull Property
//!
//! Each point `C(t)` lies in the convex hull of the control points local to the
//! active knot span.  For p+1 overlapping spans this is a "local convex hull".
//! Critical for conservative AABB computation in BVH construction.
//!
//! ## Example
//!
//! ```rust,no_run
//! use gaia::domain::geometry::nurbs::knot::KnotVector;
//! use gaia::domain::geometry::nurbs::curve::NurbsCurve;
//! use leto::geometry::Vector as SVector;
//!
//! // Quadratic NURBS arc (quarter circle in XY plane)
//! let ctrl = vec![
//!     SVector::<f64, 3>::new(1.0, 0.0, 0.0),
//!     SVector::<f64, 3>::new(1.0, 1.0, 0.0),
//!     SVector::<f64, 3>::new(0.0, 1.0, 0.0),
//! ];
//! let weights = vec![1.0_f64, std::f64::consts::FRAC_1_SQRT_2, 1.0];
//! let knots = KnotVector::try_new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
//! let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
//!
//! let mid = curve.point(0.5);
//! let scale = mid[0].hypot(mid[1]);
//! assert!((scale - 1.0).abs() < 1e-10, "point should be on unit circle");
//! ```

mod bspline;
mod error;
mod rational;

#[cfg(test)]
mod tests;

pub use bspline::BSplineCurve;
pub use error::CurveError;
pub use rational::NurbsCurve;
