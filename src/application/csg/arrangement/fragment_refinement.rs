//! Fragment refinement for arrangement CSG.
//!
//! Performs cross-mesh-only near-duplicate unification after independent CDTs.

use hashbrown::{HashMap, HashSet};

use super::super::corefine::{corefine_face, CorefinerScratch, SeamVertexMap};
use super::super::intersect::SnapSegment;
use super::classify::FragRecord;
use crate::domain::core::constants::COREFINE_WELD_TOL_SQ;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Real;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

/// Corefine each non-skipped face into fragment records using the canonical
/// arrangement CDT path.
pub(crate) fn append_corefined_fragments<T, F>(
    fragments: &mut Vec<T>,
    faces: &[FaceData],
    skipped_faces: &HashSet<usize>,
    snap_segments: &[Vec<SnapSegment>],
    pool: &mut VertexPool,
    seam_map: &SeamVertexMap,
    mut build_fragment: F,
) where
    F: FnMut(FaceData, usize) -> T,
{
    debug_assert_eq!(
        faces.len(),
        snap_segments.len(),
        "every source face must have a snap-segment slot"
    );

    let mut scratch = CorefinerScratch::new();

    for (face_index, face) in faces.iter().enumerate() {
        if skipped_faces.contains(&face_index) {
            continue;
        }

        let face_segments = &snap_segments[face_index];
        if face_segments.is_empty() {
            fragments.push(build_fragment(*face, face_index));
            continue;
        }

        for sub_face in corefine_face(face, face_segments, pool, seam_map, &mut scratch) {
            fragments.push(build_fragment(sub_face, face_index));
        }
    }
}

/// Build cross-mesh consolidation map from near-duplicate A/B vertex pairs.
///
/// # Algorithm
///
/// 1. Hash the smaller partition (`A` or `B`) into a uniform 3-D grid.
/// 2. For each probe vertex in the larger partition, search 27 neighboring
///    cells and collect near-duplicate A/B pairs within `tol_sq`.
/// 3. Union these pairs in a dense indexed DSU; unions are directed `B -> A`
///    to preserve canonical A-side roots. Positions are cached into contiguous
///    arrays to avoid repeated `VertexPool` lookups in the hot inner loop.
/// 4. Emit `vid -> root_vid` substitutions for non-root vertices.
///
/// # Theorem — Indexed-DSU Equivalence
///
/// Replacing a key-addressed DSU (`HashMap<VertexId, VertexId>`) with an
/// index-addressed DSU over a bijection `VertexId <-> index` preserves the
/// partition relation exactly because `find`/`union` are performed over the
/// same pair sequence under an isomorphic state encoding. ∎
///
/// # Theorem — Position-Cache Equivalence
///
/// Replacing repeated `pool.position(vid)` reads with a one-time cached copy
/// of each participating vertex position preserves all union decisions because
/// each cached point is value-identical to its source pool position. Thus every
/// distance predicate `||p_a - p_b||² < tol_sq` evaluates identically. ∎
///
/// # Theorem — A-Root Invariant
///
/// Every accepted union is applied as `parent[root_b] = root_a`, where `a` is
/// drawn from partition `A` and `b` from partition `B`. Therefore every mixed
/// connected component has at least one A vertex and, by induction over unions,
/// its representative remains an A-side vertex. ∎
fn build_cross_mesh_merge_map(
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

    let a_positions: Vec<_> = pure_a.iter().map(|&vid| *pool.position(vid)).collect();
    let b_positions: Vec<_> = pure_b.iter().map(|&vid| *pool.position(vid)).collect();

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> =
        HashMap::with_capacity(usize::min(a_positions.len(), b_positions.len()));

    if grid_on_a {
        for (i, p) in a_positions.iter().enumerate() {
            let ix = (p.x * inv_cell).floor() as i64;
            let iy = (p.y * inv_cell).floor() as i64;
            let iz = (p.z * inv_cell).floor() as i64;
            grid.entry((ix, iy, iz))
                .or_insert_with(|| Vec::with_capacity(2))
                .push(i);
        }
    } else {
        for (i, p) in b_positions.iter().enumerate() {
            let ix = (p.x * inv_cell).floor() as i64;
            let iy = (p.y * inv_cell).floor() as i64;
            let iz = (p.z * inv_cell).floor() as i64;
            grid.entry((ix, iy, iz))
                .or_insert_with(|| Vec::with_capacity(2))
                .push(i);
        }
    }

    let na = pure_a.len();
    let nb = pure_b.len();
    let mut parent: Vec<usize> = (0..(na + nb)).collect();

    fn find_root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            let p = parent[x];
            let gp = parent[p];
            parent[x] = gp;
            x = gp;
        }
        x
    }

    if grid_on_a {
        // Grid on A, probe B.
        for (b_idx, probe_p) in b_positions.iter().enumerate() {
            let ix = (probe_p.x * inv_cell).floor() as i64;
            let iy = (probe_p.y * inv_cell).floor() as i64;
            let iz = (probe_p.z * inv_cell).floor() as i64;
            for dx in -1_i64..=1 {
                for dy in -1_i64..=1 {
                    for dz in -1_i64..=1 {
                        if let Some(cands) = grid.get(&(ix + dx, iy + dy, iz + dz)) {
                            for &a_idx in cands {
                                let pa = a_positions[a_idx];
                                let ddx = probe_p.x - pa.x;
                                let ddy = probe_p.y - pa.y;
                                let ddz = probe_p.z - pa.z;
                                if ddx * ddx + ddy * ddy + ddz * ddz < tol_sq {
                                    let ia = a_idx;
                                    let ib = na + b_idx;
                                    let ra = find_root(&mut parent, ia);
                                    let rb = find_root(&mut parent, ib);
                                    if ra != rb {
                                        // A-root invariant: attach B-root under A-root.
                                        parent[rb] = ra;
                                        debug_assert!(
                                            ra < na,
                                            "A-root invariant violated: ra={ra} must be \
                                             in A-partition (na={na})"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Grid on B, probe A.
        for (a_idx, probe_p) in a_positions.iter().enumerate() {
            let ix = (probe_p.x * inv_cell).floor() as i64;
            let iy = (probe_p.y * inv_cell).floor() as i64;
            let iz = (probe_p.z * inv_cell).floor() as i64;
            for dx in -1_i64..=1 {
                for dy in -1_i64..=1 {
                    for dz in -1_i64..=1 {
                        if let Some(cands) = grid.get(&(ix + dx, iy + dy, iz + dz)) {
                            for &b_idx in cands {
                                let pb = b_positions[b_idx];
                                let ddx = pb.x - probe_p.x;
                                let ddy = pb.y - probe_p.y;
                                let ddz = pb.z - probe_p.z;
                                if ddx * ddx + ddy * ddy + ddz * ddz < tol_sq {
                                    let ia = a_idx;
                                    let ib = na + b_idx;
                                    let ra = find_root(&mut parent, ia);
                                    let rb = find_root(&mut parent, ib);
                                    if ra != rb {
                                        // A-root invariant: attach B-root under A-root.
                                        parent[rb] = ra;
                                        debug_assert!(
                                            ra < na,
                                            "A-root invariant violated: ra={ra} must be \
                                             in A-partition (na={na})"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut merge_map: HashMap<VertexId, VertexId> =
        HashMap::with_capacity((na + nb).saturating_div(2));
    for i in 0..(na + nb) {
        let root_i = find_root(&mut parent, i);
        if root_i != i {
            let vid = if i < na { pure_a[i] } else { pure_b[i - na] };
            let root = if root_i < na {
                pure_a[root_i]
            } else {
                pure_b[root_i - na]
            };
            merge_map.insert(vid, root);
        }
    }
    merge_map
}

/// Consolidate near-duplicate corefined vertices across meshes A and B.
///
/// Uses a spatial hash and union-find over vertices that are exclusive to A or
/// exclusive to B. Vertices shared by both sources are preserved as seam anchors.
///
/// # Tolerance
///
/// Twice the corefine Steiner snap tolerance in linear terms, because those snap
/// rounds are what produce the near-duplicates this pass merges. Two sibling
/// passes merge cross-mesh near-duplicates at other values -- the same-named
/// pass in `multi_mesh_resolution` (whose duplicates come from the `VertexPool`
/// weld) and `patch::patch_small_boundary_holes` (whose duplicates come from
/// shallow-angle Steiner drift). All three follow one rule: twice the tolerance
/// of the pass that produced the near-duplicates. The numbers therefore differ
/// because their upstreams do. Unifying them by value alone would break whichever
/// pass moved.
pub(crate) fn consolidate_cross_mesh_vertices(frags: &mut Vec<FragRecord>, pool: &VertexPool) {
    // Twice the corefine Steiner snap tolerance in linear terms, which is four
    // times it in squared terms -- the same relation `corefine` itself applies
    // to `WELD_TOL_SQ` at its comparison sites. Derived from the SSOT constant
    // rather than restated, so a change to the corefine tolerance cannot leave
    // this pass silently behind.
    const CONSOLIDATE_TOL_SQ: Real = 4.0 * COREFINE_WELD_TOL_SQ;

    let mut vids_a: HashSet<VertexId> = HashSet::with_capacity(frags.len());
    let mut vids_b: HashSet<VertexId> = HashSet::with_capacity(frags.len());
    for fr in &*frags {
        for &v in &fr.face.vertices {
            if fr.from_a {
                vids_a.insert(v);
            } else {
                vids_b.insert(v);
            }
        }
    }

    // Already shared seam IDs must never move.
    let mut seam_vids: HashSet<VertexId> = HashSet::with_capacity(vids_a.len().min(vids_b.len()));
    seam_vids.extend(vids_a.intersection(&vids_b).copied());
    let mut pure_a: Vec<VertexId> =
        Vec::with_capacity(vids_a.len().saturating_sub(seam_vids.len()));
    pure_a.extend(vids_a.difference(&seam_vids).copied());
    let mut pure_b: Vec<VertexId> =
        Vec::with_capacity(vids_b.len().saturating_sub(seam_vids.len()));
    pure_b.extend(vids_b.difference(&seam_vids).copied());

    // The merge map is union-find based; preserve the lowest vertex ID as
    // representative by making the operand traversal independent of hash
    // iteration order.
    pure_a.sort_unstable();
    pure_b.sort_unstable();

    if pure_a.is_empty() || pure_b.is_empty() {
        return;
    }
    let merge_map = build_cross_mesh_merge_map(&pure_a, &pure_b, pool, CONSOLIDATE_TOL_SQ);
    if merge_map.is_empty() {
        return;
    }

    for fr in frags.iter_mut() {
        for v in &mut fr.face.vertices {
            if let Some(&root) = merge_map.get(v) {
                *v = root;
            }
        }
    }
    frags.retain(|fr| {
        let v = fr.face.vertices;
        v[0] != v[1] && v[1] != v[2] && v[2] != v[0]
    });

    let mut seen: HashSet<[VertexId; 3]> = HashSet::with_capacity(frags.len());
    frags.retain(|fr| {
        let mut key = fr.face.vertices;
        key.sort();
        seen.insert(key)
    });
}

#[cfg(test)]
mod tests;
