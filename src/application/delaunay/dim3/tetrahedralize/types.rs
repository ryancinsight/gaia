//! 3D Delaunay cell types: faces and tetrahedra.

use leto::geometry::Point3;

use crate::domain::core::scalar::Scalar;

/// A mathematical representation of a triangulated face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Face {
    /// Ordered vertex indices guaranteeing deterministic hashing.
    pub v: [usize; 3],
}

impl Face {
    /// Construct a normalized face. The vertices are rigorously sorted
    /// to guarantee equivalent faces hash identically, regardless of winding.
    #[must_use]
    pub fn new(mut v0: usize, mut v1: usize, mut v2: usize) -> Self {
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        if v1 > v2 {
            std::mem::swap(&mut v1, &mut v2);
        }
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        Self { v: [v0, v1, v2] }
    }
}

/// A Delaunay tetrahedron formed by 4 vertices.
#[derive(Debug, Clone)]
pub struct Tetrahedron<T: Scalar> {
    /// Indices into the shared vertex buffer.
    pub v: [usize; 4],
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> Tetrahedron<T> {
    /// Construct a tetrahedron and enforce positive orientation at the exact
    /// native-precision predicate boundary.
    pub fn new(mut v: [usize; 4], points: &[Point3<T>]) -> Self {
        use crate::domain::geometry::predicates;

        // Shewchuk's exact `insphere` predicate analytically requires the 4 defining vertices
        // to be strictly positively oriented according to HIS convention (gp::orient3d > 0).
        // Since `gaia::predicates::orient_3d` negates `gp::orient3d` to maintain the right-hand rule,
        // a Shewchuk-positive orientation actually corresponds to `gaia::orient_3d.is_negative()`.
        // Therefore, we swap to fix the orientation if `gaia::orient_3d` evaluates as POSITIVE.
        if predicates::orient_3d_pts(&points[v[0]], &points[v[1]], &points[v[2]], &points[v[3]])
            .is_positive()
        {
            v.swap(2, 3);
        }

        Self {
            v,
            _marker: std::marker::PhantomData,
        }
    }

    /// Evaluate circumsphere inclusion using the exact native-precision
    /// predicates.
    #[inline(always)]
    pub fn contains_in_circumsphere(&self, p: &Point3<T>, points: &[Point3<T>]) -> bool {
        use crate::domain::geometry::predicates;

        let at = |i: usize| [points[i].x, points[i].y, points[i].z];
        let orientation = predicates::insphere(
            at(self.v[0]),
            at(self.v[1]),
            at(self.v[2]),
            at(self.v[3]),
            [p.x, p.y, p.z],
        );

        // Exact Delaunay property: Points strictly inside the circumsphere invalidate the tetrahedron.
        // Points *exactly* on the circumsphere (Degenerate) are safely excluded to prevent
        // infinitely cascading cavity erosion across regular convex surfaces like cylinders.
        orientation.is_positive()
    }

    /// Retrieve the 4 encompassing faces of the tetrahedron.
    #[must_use]
    pub fn faces(&self) -> [Face; 4] {
        [
            Face::new(self.v[0], self.v[1], self.v[2]),
            Face::new(self.v[0], self.v[1], self.v[3]),
            Face::new(self.v[0], self.v[2], self.v[3]),
            Face::new(self.v[1], self.v[2], self.v[3]),
        ]
    }

    /// Check if this tetrahedron shares any vertices with the bounding super-tetrahedron.
    #[must_use]
    pub fn shares_vertex_with_super(&self, super_start_idx: usize) -> bool {
        self.v
            .iter()
            .any(|&idx| idx >= super_start_idx && idx < super_start_idx + 4)
    }

    // calculate_circumsphere has been structurally eliminated.
    // Mathematical invariants are natively resolved by exact Shewchuk predicates.
}
