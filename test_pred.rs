fn main() {
    let to_r = |p: [f64; 3]| p;
    let d = 100.0;
    let v0 = to_r([0.0, d, 0.0]);
    let v1 = to_r([-d, -d, -d]);
    let v2 = to_r([d, -d, -d]);
    let v3 = to_r([0.0, -d, d]);

    // Test orientation. Shewchuk orient3d > 0 means D is BELOW A,B,C.
    // Right hand rule (standard physics volume) is POSITIVE if D is ABOVE A,B,C.
    // So Shewchuk > 0  => Standard < 0.
    let gp_orient = geometry_predicates::orient3d(v0, v1, v2, v3);
    println!("gp::orient3d(v0, v1, v2, v3): {}", gp_orient);

    let e = to_r([0.0, 0.0, 0.0]);
    let gp_insphere_orig = geometry_predicates::insphere(v0, v1, v2, v3, e);
    println!("gp::insphere(v0, v1, v2, v3, 0.0): {}", gp_insphere_orig);

    let gp_orient_swap = geometry_predicates::orient3d(v0, v1, v3, v2);
    println!("gp::orient3d(v0, v1, v3, v2): {}", gp_orient_swap);

    let gp_insphere_swap = geometry_predicates::insphere(v0, v1, v3, v2, e);
    println!("gp::insphere(v0, v1, v3, v2, 0.0): {}", gp_insphere_swap);
}
