//! Adversarial tests targeting known failure modes in mesh libraries.
//!
//! These tests exercise pathological geometries and edge cases that commonly
//! expose bugs in Delaunay triangulation, CDT constraint recovery, and
//! Ruppert refinement implementations.
//!
//! # Motivation
//!
//! Production mesh libraries (Triangle, CGAL, Gmsh, etc.) have documented
//! issues with:
//! - Co-circular (co-spherical in 3-D) point sets → degenerate incircle
//! - Collinear / near-collinear point sets → degenerate orient2d
//! - Near-coincident points → floating-point resolution collapse
//! - Very thin slivers → near-zero area, circumradius blowup
//! - Regular grids → systematic co-circularity
//! - Walk cycles → infinite loops in Lawson walk
//! - Constraint through existing DT edges → no-op vs. spurious flip
//! - Small input angles → Ruppert non-termination
//!
//! Each test is annotated with the failure mode it targets and a reference
//! to the relevant literature where applicable.

use crate::application::delaunay::dim2::constraint::enforce::Cdt;
use crate::application::delaunay::dim2::pslg::graph::Pslg;
use crate::application::delaunay::dim2::triangulation::bowyer_watson::DelaunayTriangulation;
use std::f64::consts::PI;

mod part1;
mod part2;
