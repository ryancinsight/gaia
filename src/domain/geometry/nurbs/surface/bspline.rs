use super::super::basis::{eval_basis_and_deriv_to_slice, eval_basis_to_slice};
use super::super::knot::KnotVector;
use super::validate::validate_surface_dims;
use super::{ControlGrid, SurfaceError};
use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;
use leto::geometry::{Point3, UnitVector3, Vector3};

// BSplineSurface
// ---------------------------------------------------------------------------

/// A non-rational tensor-product B-spline surface.
///
/// Evaluation:
///   S(u, v) = `sum_i` `sum_j`  N_{i,p}(u) * N_{j,q}(v) * P_{ij}
///
/// where N_{i,p} and N_{j,q} are B-spline basis functions computed by
/// Cox-de Boor recursion.
#[derive(Clone, Debug)]
pub struct BSplineSurface<T = Real> {
    /// Control point grid, row-major with `v` rows and `u` columns.
    pub control_grid: ControlGrid<T>,
    /// Knot vector in the u direction.
    pub knots_u: KnotVector<T>,
    /// Knot vector in the v direction.
    pub knots_v: KnotVector<T>,
    /// Degree in the u direction.
    pub degree_u: usize,
    /// Degree in the v direction.
    pub degree_v: usize,
}

impl<T: Scalar> BSplineSurface<T> {
    /// Create a B-spline surface, validating knot / control-point consistency.
    pub fn new(
        control_grid: ControlGrid<T>,
        knots_u: KnotVector<T>,
        knots_v: KnotVector<T>,
        degree_u: usize,
        degree_v: usize,
    ) -> Result<Self, SurfaceError> {
        validate_surface_dims(&control_grid, &knots_u, &knots_v, degree_u, degree_v)?;
        Ok(Self {
            control_grid,
            knots_u,
            knots_v,
            degree_u,
            degree_v,
        })
    }

    /// Create with clamped uniform knot vectors constructed automatically.
    pub fn clamped(
        control_grid: ControlGrid<T>,
        degree_u: usize,
        degree_v: usize,
    ) -> Result<Self, SurfaceError> {
        if control_grid.n_rows() == 0 || control_grid.n_cols() == 0 {
            return Err(SurfaceError::EmptyControlGrid);
        }
        let n_u = control_grid.n_cols() - 1; // u = cols
        let n_v = control_grid.n_rows() - 1; // v = rows
        let ku = KnotVector::clamped_uniform(n_u, degree_u);
        let kv = KnotVector::clamped_uniform(n_v, degree_v);
        Self::new(control_grid, ku, kv, degree_u, degree_v)
    }

    /// Parameter domain `((u_min, u_max), (v_min, v_max))`.
    #[must_use]
    pub fn domain(&self) -> ((T, T), (T, T)) {
        (self.knots_u.domain(), self.knots_v.domain())
    }

    /// Evaluate the surface at `(u, v)`.
    #[must_use]
    pub fn point(&self, u: T, v: T) -> Point3<T> {
        let n_u = self.control_grid.n_cols() - 1; // u = cols
        let n_v = self.control_grid.n_rows() - 1; // v = rows
        let su = self.knots_u.find_span(u, n_u);
        let sv = self.knots_v.find_span(v, n_v);
        let zero = <T as NumericElement>::ZERO;
        let mut bu_buf = [zero; 9];
        let mut bv_buf = [zero; 9];
        let mut bu_vec;
        let mut bv_vec;
        let bu = if self.degree_u <= 8 {
            &mut bu_buf[..=self.degree_u]
        } else {
            bu_vec = vec![zero; self.degree_u + 1];
            &mut bu_vec[..]
        };
        let bv = if self.degree_v <= 8 {
            &mut bv_buf[..=self.degree_v]
        } else {
            bv_vec = vec![zero; self.degree_v + 1];
            &mut bv_vec[..]
        };
        eval_basis_to_slice(su, u, self.degree_u, &self.knots_u, bu);
        eval_basis_to_slice(sv, v, self.degree_v, &self.knots_v, bv);

        let pu = self.degree_u;
        let pv = self.degree_v;
        let mut res = Vector3::<T>::zeros();
        for (j, &nu) in bu.iter().enumerate() {
            for (k, &nv) in bv.iter().enumerate() {
                // get(row, col) = get(v_idx, u_idx)
                res += self.control_grid.get(sv - pv + k, su - pu + j).coords * (nu * nv);
            }
        }
        Point3::from(res)
    }

    /// Evaluate surface point and partial derivatives `(S, dS/du, dS/dv)`.
    #[must_use]
    pub fn point_and_derivs(&self, u: T, v: T) -> (Point3<T>, Vector3<T>, Vector3<T>) {
        let n_u = self.control_grid.n_cols() - 1; // u = cols
        let n_v = self.control_grid.n_rows() - 1; // v = rows
        let su = self.knots_u.find_span(u, n_u);
        let sv = self.knots_v.find_span(v, n_v);
        let zero = <T as NumericElement>::ZERO;
        let mut bu_buf = [zero; 9];
        let mut dbu_buf = [zero; 9];
        let mut bv_buf = [zero; 9];
        let mut dbv_buf = [zero; 9];
        let mut bu_vec;
        let mut dbu_vec;
        let mut bv_vec;
        let mut dbv_vec;
        let (bu, dbu) = if self.degree_u <= 8 {
            (
                &mut bu_buf[..=self.degree_u],
                &mut dbu_buf[..=self.degree_u],
            )
        } else {
            bu_vec = vec![zero; self.degree_u + 1];
            dbu_vec = vec![zero; self.degree_u + 1];
            (&mut bu_vec[..], &mut dbu_vec[..])
        };
        let (bv, dbv) = if self.degree_v <= 8 {
            (
                &mut bv_buf[..=self.degree_v],
                &mut dbv_buf[..=self.degree_v],
            )
        } else {
            bv_vec = vec![zero; self.degree_v + 1];
            dbv_vec = vec![zero; self.degree_v + 1];
            (&mut bv_vec[..], &mut dbv_vec[..])
        };
        eval_basis_and_deriv_to_slice(su, u, self.degree_u, &self.knots_u, bu, dbu);
        eval_basis_and_deriv_to_slice(sv, v, self.degree_v, &self.knots_v, bv, dbv);

        let pu = self.degree_u;
        let pv = self.degree_v;
        let mut s = Vector3::<T>::zeros();
        let mut ds_du = Vector3::<T>::zeros();
        let mut ds_dv = Vector3::<T>::zeros();

        for (j, (&nu, &dnu)) in bu.iter().zip(dbu.iter()).enumerate() {
            for (k, (&nv, &dnv)) in bv.iter().zip(dbv.iter()).enumerate() {
                let pt = self.control_grid.get(sv - pv + k, su - pu + j).coords;
                s += pt * (nu * nv);
                ds_du += pt * (dnu * nv);
                ds_dv += pt * (nu * dnv);
            }
        }
        (Point3::from(s), ds_du, ds_dv)
    }

    /// Unit surface normal at `(u, v)` = normalize(dS/du cross dS/dv).
    /// Returns `None` if the surface is degenerate at `(u, v)`.
    #[must_use]
    pub fn normal(&self, u: T, v: T) -> Option<UnitVector3<T>> {
        let (_, du, dv) = self.point_and_derivs(u, v);
        UnitVector3::try_new(du.cross(dv), <T as Scalar>::from_f64(1e-15))
    }
}

// ---------------------------------------------------------------------------
