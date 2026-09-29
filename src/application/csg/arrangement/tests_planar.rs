//! Tests for the parent module, extracted from the module body.

use super::*;
use hashbrown::HashSet;
use proptest::prelude::*;

#[test]
fn indexed_segment_collection_finds_all_collinear_points() {
    let pts = vec![
        [0.0, 0.0],
        [10.0, 0.0],
        [2.5, 0.0],
        [5.0, 0.0],
        [7.5, 0.0],
        [5.0, 1e-7],
    ];
    let index = PlanarPointGridIndex::new(&pts, 0.1);
    let mut candidates = Vec::new();
    let mut got = Vec::new();
    collect_points_on_segment_interior_indexed(
        &pts,
        &index,
        pts[0],
        pts[1],
        (0, 1),
        1e-8,
        1e-12,
        &mut candidates,
        &mut got,
    );
    let slots: HashSet<usize> = got.into_iter().map(|(_, slot)| slot).collect();
    assert!(slots.contains(&0));
    assert!(slots.contains(&1));
    assert!(slots.contains(&2));
    assert!(slots.contains(&3));
    assert!(slots.contains(&4));
}

proptest! {
    #[test]
    fn segment_corridor_candidates_cover_exact_accepts(
        pts in prop::collection::vec((-120_i16..120_i16, -120_i16..120_i16), 6..100),
    ) {
        let unique_pts: Vec<[Real; 2]> = pts
            .into_iter()
            .map(|(x, y)| [Real::from(x) * 0.05, Real::from(y) * 0.05])
            .collect();

        let p1 = unique_pts[0];
        let p2 = unique_pts[1];
        let endpoint_slots = (0usize, 1usize);
        let t_eps = 1e-8;
        let dist_sq_tol = 1e-10;

        let exact = collect_points_on_segment_interior(
            &unique_pts,
            p1,
            p2,
            endpoint_slots,
            t_eps,
            dist_sq_tol,
        );
        let exact_slots: HashSet<usize> = exact.into_iter().map(|(_, slot)| slot).collect();

        let index = PlanarPointGridIndex::new(&unique_pts, dist_sq_tol.sqrt());
        let mut candidates = Vec::new();
        index.collect_segment_corridor_candidates(p1, p2, dist_sq_tol.sqrt(), &mut candidates);
        let candidate_slots: HashSet<usize> = candidates.into_iter().collect();

        for slot in exact_slots {
            prop_assert!(
                candidate_slots.contains(&slot),
                "candidate retrieval missed exact-accepted slot {slot}"
            );
        }
    }

    #[test]
    fn indexed_segment_collection_matches_bruteforce(
        pts in prop::collection::vec((-100_i16..100_i16, -100_i16..100_i16), 6..80),
    ) {
        let unique_pts: Vec<[Real; 2]> = pts
            .into_iter()
            .map(|(x, y)| [Real::from(x) * 0.05, Real::from(y) * 0.05])
            .collect();

        let p1 = unique_pts[0];
        let p2 = unique_pts[1];
        let endpoint_slots = (0usize, 1usize);
        let t_eps = 1e-8;
        let dist_sq_tol = 1e-10;

        let brute = collect_points_on_segment_interior(
            &unique_pts,
            p1,
            p2,
            endpoint_slots,
            t_eps,
            dist_sq_tol,
        );

        let index = PlanarPointGridIndex::new(&unique_pts, dist_sq_tol.sqrt());
        let mut candidates = Vec::new();
        let mut fast = Vec::new();
        collect_points_on_segment_interior_indexed(
            &unique_pts,
            &index,
            p1,
            p2,
            endpoint_slots,
            t_eps,
            dist_sq_tol,
            &mut candidates,
            &mut fast,
        );

        let brute_slots: HashSet<usize> = brute.into_iter().map(|(_, slot)| slot).collect();
        let fast_slots: HashSet<usize> = fast.into_iter().map(|(_, slot)| slot).collect();
        prop_assert_eq!(fast_slots, brute_slots);
    }
}
