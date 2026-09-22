use super::*;
use crate::domain::core::scalar::{Point3r, Vector3r};
use proptest::prelude::*;

fn build_cross_mesh_merge_map_reference(
    pure_a: &[VertexId],
    pure_b: &[VertexId],
    pool: &VertexPool,
    tol_sq: Real,
) -> HashMap<VertexId, VertexId> {
    if pure_a.is_empty() || pure_b.is_empty() {
        return HashMap::new();
    }

    let tol = tol_sq.sqrt();
    if tol <= 0.0 {
        return HashMap::new();
    }
    let inv_cell = 1.0 / tol;
    let grid_on_a = pure_a.len() <= pure_b.len();
    let (grid_source, probe_source): (&[VertexId], &[VertexId]) = if grid_on_a {
        (pure_a, pure_b)
    } else {
        (pure_b, pure_a)
    };

    let mut grid: HashMap<(i64, i64, i64), Vec<VertexId>> =
        HashMap::with_capacity(grid_source.len());
    for &vid in grid_source {
        let p = pool.position(vid);
        let ix = (p.x * inv_cell).floor() as i64;
        let iy = (p.y * inv_cell).floor() as i64;
        let iz = (p.z * inv_cell).floor() as i64;
        grid.entry((ix, iy, iz))
            .or_insert_with(|| Vec::with_capacity(2))
            .push(vid);
    }

    let all_merge_vids: Vec<VertexId> = pure_a.iter().chain(pure_b.iter()).copied().collect();
    let mut index_of: HashMap<VertexId, usize> = HashMap::with_capacity(all_merge_vids.len());
    for (i, &vid) in all_merge_vids.iter().enumerate() {
        index_of.insert(vid, i);
    }

    let mut parent: Vec<usize> = (0..all_merge_vids.len()).collect();

    fn find_root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            let p = parent[x];
            let gp = parent[p];
            parent[x] = gp;
            x = gp;
        }
        x
    }

    for &probe in probe_source {
        let probe_p = pool.position(probe);
        let ix = (probe_p.x * inv_cell).floor() as i64;
        let iy = (probe_p.y * inv_cell).floor() as i64;
        let iz = (probe_p.z * inv_cell).floor() as i64;
        for dx in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dz in -1_i64..=1 {
                    if let Some(cands) = grid.get(&(ix + dx, iy + dy, iz + dz)) {
                        for &cand in cands {
                            let (va, vb, pa, pb) = if grid_on_a {
                                (cand, probe, pool.position(cand), probe_p)
                            } else {
                                (probe, cand, probe_p, pool.position(cand))
                            };
                            if (pb - pa).norm_squared() < tol_sq {
                                let ia = index_of[&va];
                                let ib = index_of[&vb];
                                let ra = find_root(&mut parent, ia);
                                let rb = find_root(&mut parent, ib);
                                if ra != rb {
                                    parent[rb] = ra;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut merge_map: HashMap<VertexId, VertexId> = HashMap::with_capacity(all_merge_vids.len());
    for &vid in &all_merge_vids {
        let i = index_of[&vid];
        let root_i = find_root(&mut parent, i);
        let root = all_merge_vids[root_i];
        if root != vid {
            merge_map.insert(vid, root);
        }
    }
    merge_map
}

proptest! {
    #[test]
    fn dense_cache_merge_map_matches_reference(
        a_pts in prop::collection::vec(prop::array::uniform3(-40_i16..40_i16), 0..24),
        b_pts in prop::collection::vec(prop::array::uniform3(-40_i16..40_i16), 0..24),
    ) {
        let mut pool = VertexPool::default_millifluidic();
        let n = Vector3r::new(0.0, 0.0, 1.0);

        let pure_a: Vec<VertexId> = a_pts
            .into_iter()
            .map(|p| {
                pool.insert_unique(
                    Point3r::new(
                        f64::from(p[0]) * 1e-2,
                        f64::from(p[1]) * 1e-2,
                        f64::from(p[2]) * 1e-2,
                    ),
                    n,
                )
            })
            .collect();
        let pure_b: Vec<VertexId> = b_pts
            .into_iter()
            .map(|p| {
                pool.insert_unique(
                    Point3r::new(
                        f64::from(p[0]) * 1e-2,
                        f64::from(p[1]) * 1e-2,
                        f64::from(p[2]) * 1e-2,
                    ),
                    n,
                )
            })
            .collect();

        let tol_sq = 4.0e-4_f64;
        let got = build_cross_mesh_merge_map(&pure_a, &pure_b, &pool, tol_sq);
        let reference = build_cross_mesh_merge_map_reference(&pure_a, &pure_b, &pool, tol_sq);
        prop_assert_eq!(got, reference);
    }
}

#[test]
fn merge_map_prefers_a_side_root() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let a0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    // Placed outside the weld tolerance (1e-4) so the pool keeps it as a
    // distinct vertex, but inside the 2e-4 tolerance this test passes to
    // `build_cross_mesh_merge_map`. Note that 2e-4 is the *N-way* pass's
    // tolerance, not this module's production 2e-6 -- the helper takes the
    // tolerance as a parameter, and this test exercises it at a spacing the
    // pool will actually preserve.
    let b0 = pool.insert_or_weld(Point3r::new(1.5e-4, 0.0, 0.0), n);

    let map = build_cross_mesh_merge_map(&[a0], &[b0], &pool, 4.0e-8);
    assert_eq!(
        map.get(&b0),
        Some(&a0),
        "B-side near-duplicate should map to A-side root"
    );
    assert!(
        !map.contains_key(&a0),
        "A-side canonical root should not be remapped"
    );
}

#[test]
fn adversarial_bridge_keeps_a_root_in_component() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);

    // b0 is close to both a0 and a1, forcing transitive unions.
    let a0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let a1 = pool.insert_or_weld(Point3r::new(3.0e-4, 0.0, 0.0), n);
    let b0 = pool.insert_or_weld(Point3r::new(1.5e-4, 0.0, 0.0), n);

    let map = build_cross_mesh_merge_map(&[a0, a1], &[b0], &pool, 4.0e-8);

    let root_of_b = map.get(&b0).copied().unwrap_or(b0);
    assert!(
        root_of_b == a0 || root_of_b == a1,
        "component representative must remain on A-side"
    );
}
