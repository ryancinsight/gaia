use geometry_predicates as gp;

fn main() {
    println!("fixture\traw_determinant\traw_bits");
    let h = 2.0_f64.powi(600);
    for (name, value) in [
        (
            "orient2d.finite_overflow",
            gp::orient2d([0.0, 0.0], [1.0, 0.0], [h, h]),
        ),
        (
            "orient2d.cyclic",
            gp::orient2d([1.0, 0.0], [h, h], [0.0, 0.0]),
        ),
        ("orient2d.underflow", {
            let s = 2.0_f64.powi(-600);
            gp::orient2d([0.0, 0.0], [s, 0.0], [0.0, s])
        }),
        ("orient3d.overflow", {
            let s = 2.0_f64.powi(400);
            gp::orient3d([0.0; 3], [s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s])
        }),
        ("orient3d.underflow", {
            let s = 2.0_f64.powi(-400);
            gp::orient3d([0.0; 3], [s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s])
        }),
        ("incircle.overflow", {
            let s = 2.0_f64.powi(300);
            gp::incircle([s, 0.0], [0.0, s], [-s, 0.0], [0.0; 2])
        }),
        ("incircle.underflow", {
            let s = 2.0_f64.powi(-300);
            gp::incircle([s, 0.0], [0.0, s], [-s, 0.0], [0.0; 2])
        }),
        ("insphere.overflow", {
            let s = 2.0_f64.powi(240);
            gp::insphere(
                [s, 0.0, 0.0],
                [0.0, s, 0.0],
                [0.0, 0.0, s],
                [-s, 0.0, 0.0],
                [0.0; 3],
            )
        }),
        ("insphere.underflow", {
            let s = 2.0_f64.powi(-240);
            gp::insphere(
                [s, 0.0, 0.0],
                [0.0, s, 0.0],
                [0.0, 0.0, s],
                [-s, 0.0, 0.0],
                [0.0; 3],
            )
        }),
    ] {
        println!("{name}\t{value:e}\t{:016x}", value.to_bits());
    }
    println!(
        "invalid.orientation_from_nan\t{:?}",
        gaia::domain::geometry::predicates::Orientation::from_det(f64::NAN)
    );
    println!(
        "invalid.topology_sign_from_nan\t{:?}",
        gaia::domain::topology::predicates::Sign::from_exact_f64(f64::NAN)
    );
}
