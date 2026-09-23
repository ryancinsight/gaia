//! Public indexed-mesh API for CSG Boolean operations.

mod csg;
mod repair;

pub use csg::{csg_boolean, csg_boolean_nary};

#[cfg(test)]
#[path = "indexed_tests.rs"]
mod indexed_tests;
