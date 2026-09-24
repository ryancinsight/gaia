use super::super::knot::KnotVector;
use super::{ControlGrid, SurfaceError};
use crate::domain::core::scalar::Scalar;

pub(super) fn validate_surface_dims<T: Scalar>(
    control_grid: &ControlGrid<T>,
    knots_u: &KnotVector<T>,
    knots_v: &KnotVector<T>,
    degree_u: usize,
    degree_v: usize,
) -> Result<(), SurfaceError> {
    if control_grid.n_rows() == 0 || control_grid.n_cols() == 0 {
        return Err(SurfaceError::EmptyControlGrid);
    }
    if degree_u == 0 {
        return Err(SurfaceError::ZeroDegree { direction: 'u' });
    }
    if degree_v == 0 {
        return Err(SurfaceError::ZeroDegree { direction: 'v' });
    }

    let exp_u = control_grid.n_cols() + degree_u + 1;
    if knots_u.len() != exp_u {
        return Err(SurfaceError::KnotCountMismatch {
            direction: 'u',
            got: knots_u.len(),
            expected: exp_u,
        });
    }
    let exp_v = control_grid.n_rows() + degree_v + 1;
    if knots_v.len() != exp_v {
        return Err(SurfaceError::KnotCountMismatch {
            direction: 'v',
            got: knots_v.len(),
            expected: exp_v,
        });
    }
    Ok(())
}
