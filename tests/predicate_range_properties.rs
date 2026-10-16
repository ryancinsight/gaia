//! Full stored-bit permutation and exact-degeneracy properties for GAIA-026.

use gaia::domain::geometry::predicates::{self as geometry, Orientation};
use gaia::domain::topology::predicates::{self as topology, Sign};
use leto::geometry::{Point2, Point3};
use proptest::prelude::*;

fn finite(bits: u64) -> f64 {
    // Preserve every finite pattern. Map only NaN/infinity to the adjacent
    // finite exponent; no numerical rescaling or rounding occurs.
    let finite_bits = if bits & (0x7ff_u64 << 52) == 0x7ff_u64 << 52 {
        bits ^ (1_u64 << 52)
    } else {
        bits
    };
    f64::from_bits(finite_bits)
}

fn opposite(sign: Orientation) -> Orientation {
    match sign {
        Orientation::Positive => Orientation::Negative,
        Orientation::Negative => Orientation::Positive,
        Orientation::Degenerate => Orientation::Degenerate,
    }
}

fn topology_sign(sign: Sign) -> Orientation {
    match sign {
        Sign::Positive => Orientation::Positive,
        Sign::Negative => Orientation::Negative,
        Sign::Zero => Orientation::Degenerate,
    }
}

proptest! {
    #[test]
    fn orientation_2d_permutation_and_public_aliases(bits in any::<[u64; 6]>()) {
        let values=bits.map(finite);
        let [first,second,third]=[[values[0],values[1]],[values[2],values[3]],[values[4],values[5]]];
        let expected=geometry::orient_2d_arr(first,second,third);
        prop_assert_eq!(geometry::orient_2d_arr(second,third,first),expected);
        prop_assert_eq!(geometry::orient_2d_arr(second,first,third),opposite(expected));
        prop_assert_eq!(geometry::orient_2d(&Point2::from(first),&Point2::from(second),&Point2::from(third)),expected);
        let point=|v:[f64;2]|Point3::new(v[0],v[1],0.0);
        prop_assert_eq!(topology_sign(topology::orient2d(&point(first),&point(second),&point(third))),expected);
        prop_assert_eq!(geometry::orient_2d_arr(first,first,third),Orientation::Degenerate);
    }

    #[test]
    fn orientation_3d_permutation_and_conventions(bits in any::<[u64; 12]>()) {
        let values=bits.map(finite);
        let [first,second,third,fourth]=std::array::from_fn(|row|std::array::from_fn(|axis|values[row*3+axis]));
        let expected=geometry::orient_3d(first,second,third,fourth);
        prop_assert_eq!(geometry::orient_3d(second,third,first,fourth),expected);
        prop_assert_eq!(geometry::orient_3d(second,first,third,fourth),opposite(expected));
        prop_assert_eq!(geometry::orient_3d_pts(&Point3::from(first),&Point3::from(second),&Point3::from(third),&Point3::from(fourth)),expected);
        prop_assert_eq!(topology_sign(topology::orient3d(&Point3::from(first),&Point3::from(second),&Point3::from(third),&Point3::from(fourth))),opposite(expected));
        prop_assert_eq!(geometry::orient_3d(first,first,third,fourth),Orientation::Degenerate);
    }

    #[test]
    fn circle_winding_and_exact_degeneracy(bits in any::<[u64; 8]>()) {
        let values=bits.map(finite);
        let [first,second,third,fourth]:[Point2<f64>;4]=std::array::from_fn(|row|Point2::new(values[row*2],values[row*2+1]));
        let expected=geometry::incircle(&first,&second,&third,&fourth);
        prop_assert_eq!(geometry::incircle(&second,&third,&first,&fourth),expected);
        prop_assert_eq!(geometry::incircle(&second,&first,&third,&fourth),opposite(expected));
        let point=|v:Point2<f64>|Point3::new(v.x,v.y,0.0);
        prop_assert_eq!(topology_sign(topology::incircle2d(&point(first),&point(second),&point(third),&point(fourth))),expected);
        prop_assert_eq!(geometry::incircle(&first,&first,&third,&fourth),Orientation::Degenerate);
    }

    #[test]
    fn sphere_winding_and_exact_degeneracy(bits in any::<[u64; 15]>()) {
        let values=bits.map(finite);
        let [first,second,third,fourth,query]=std::array::from_fn(|row|std::array::from_fn(|axis|values[row*3+axis]));
        let expected=geometry::insphere(first,second,third,fourth,query);
        prop_assert_eq!(geometry::insphere(second,third,first,fourth,query),expected);
        prop_assert_eq!(geometry::insphere(second,first,third,fourth,query),opposite(expected));
        prop_assert_eq!(topology_sign(topology::insphere3d(&Point3::from(first),&Point3::from(second),&Point3::from(third),&Point3::from(fourth),&Point3::from(query))),expected);
        prop_assert_eq!(geometry::insphere(first,first,third,fourth,query),Orientation::Degenerate);
    }
}

#[test]
fn tetrahedron_extreme_range_keeps_orientation_and_circumsphere_decisions() {
    use gaia::application::delaunay::dim3::tetrahedralize::Tetrahedron;

    for exponent in [-400, 400] {
        let scale = 2.0_f64.powi(exponent);
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(scale, 0.0, 0.0),
            Point3::new(0.0, scale, 0.0),
            Point3::new(0.0, 0.0, scale),
        ];
        let tetrahedron = Tetrahedron::new([0, 1, 2, 3], &points);
        // The constructor enforces Shewchuk-positive winding, opposite to
        // geometry's right-hand convention. These exact indices expose that
        // actual caller decision even when native volume would overflow/underflow.
        assert_eq!(tetrahedron.v, [0, 1, 3, 2]);
        let inside = Point3::new(scale / 4.0, scale / 4.0, scale / 4.0);
        let outside = Point3::new(scale * 2.0, scale * 2.0, scale * 2.0);
        assert!(tetrahedron.contains_in_circumsphere(&inside, &points));
        assert!(!tetrahedron.contains_in_circumsphere(&points[0], &points));
        assert!(!tetrahedron.contains_in_circumsphere(&outside, &points));
    }
}
