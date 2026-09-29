//! Adversarial CDT, PSLG, and Ruppert tests targeting known mesh library
//! failure modes not covered by the primary adversarial test suite.
//!
//! # Targeted Failure Modes
//!
//! - **Spiral point distributions**: stress Lawson walk locality — O(√n) walk
//!   distance instead of near-O(1) when Hilbert ordering is ineffective.
//! - **Long constraint crossing many DT edges**: CDT constraint recovery must
//!   flip O(k) edges where k can be O(n).  Known to crash Triangle 1.6 on
//!   certain inputs (Shewchuk, personal communication).
//! - **Star-shaped constraints**: many constraints radiating from a single
//!   vertex create a high-degree fan that stresses cavity re-triangulation.
//! - **Closely-spaced parallel constraints**: narrow channels between
//!   constraint edges produce extreme aspect-ratio triangles during CDT
//!   recovery.
//! - **Concentric polygon constraints**: nested constraint polygons with
//!   holes test hole-removal correctness.
//! - **Points on constraint edges**: Steiner points placed exactly on
//!   constraint segments test OnEdge location handling.
//! - **Ruppert area-only refinement**: validates the max_area constraint
//!   independent of angle quality.
//! - **CDT with many crossing DT edges**: a single long constraint that
//!   crosses many Delaunay edges exercises the flip-recovery algorithm.
//! - **PSLG with multiple adjacent small holes**: holes close together test
//!   hole-removal flood-fill isolation.

use crate::application::delaunay::dim2::constraint::enforce::Cdt;
use crate::application::delaunay::dim2::pslg::graph::Pslg;
use crate::application::delaunay::dim2::triangulation::adjacency::Adjacency;
use crate::application::delaunay::dim2::triangulation::bowyer_watson::DelaunayTriangulation;
use std::f64::consts::PI;

mod part1;
mod part2;
