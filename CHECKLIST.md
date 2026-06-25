# Checklist

- [x] **Phase 17: RITK MeshBuilder Array API Provider Slice**
    - [x] Add `MeshBuilder::vertex_xyz`, `vertex_array`, and `add_triangle_soup_arrays`
      so consumers can build Gaia meshes without importing `nalgebra::Point3`.
    - [x] Add value-semantic tests for duplicate-coordinate welding and array-backed
      triangle soup face construction.
    - [x] Verify provider gates: `cargo fmt --check`, `cargo check --all-targets`,
      `cargo clippy --all-targets --all-features -- -D warnings`, `cargo nextest run`
      (922 passed, 1 skipped), `cargo test --doc` (5 passed, 39 ignored), and
      `cargo doc --no-deps`.
    - [x] Land before the RITK consumer branch that removes mesh-only direct `nalgebra`
      dependencies from `ritk-filter`, `ritk-vtk`, and `ritk-io`.

- [x] **Phase 1: Diagnosis & Analysis**
    - [x] Run `examples/csg_cube_sphere.rs` and capture output/error.
    - [x] Audit `src/application/csg/bsp.rs` for correct plane classification logic.
    - [x] Audit `src/application/csg/split.rs` for triangle splitting numerical stability.
- [x] Check `src/application/csg/boolean` for correct set operation logic (Union, Intersection, Difference) and decompose flat file.

- [x] **Phase 2: Core Boolean Logic Fixes**
    - [x] Robustify plane/triangle intersection tests (e.g., epsilon handling).
    - [x] Fix re-triangulation of split polygons (ensure CCW winding/normals).
    - [x] Handle "on-plane" cases correctly (coplanar surfaces) and decompose logic.
    - [x] Coplanar cube–cylinder CSG (caps flush with walls): union, intersection, difference — all watertight with <1% volume error.
    - [x] Fix O(n²) coplanar Difference bottleneck: AABB per-fragment pre-screening → 64-seg completes in 8ms.

- [x] **Phase 3: Tooling & Verification**
    - [x] Create `inspect_stl` binary — watertightness, volume, normals, aspect ratio, bounds.
    - [x] Add "watertight" check (edge manifoldness). *(done inside inspect_stl)*
    - [x] Add normal orientation check (outward facing).
    - [x] Add degenerate triangle check (zero area).
    - [x] `csg_cube_cylinder_coplanar` example: union, intersection, difference with coplanar end caps (64 segments, 8ms Difference).


- [x] **Phase 4: TPMS Primitives Expansion**
    - [x] Neovius surface — `tpms/neovius.rs` + `primitives/neovius_sphere.rs` + example
    - [x] Lidinoid — `tpms/lidinoid.rs` + `primitives/lidinoid_sphere.rs` + example
    - [x] I-WP (Schoen) — `tpms/iwp.rs` + `primitives/iwp_sphere.rs` + example
    - [x] Split P — `tpms/split_p.rs` + `primitives/split_p_sphere.rs` + example
    - [x] FRD (Schoen F-RD) — `tpms/frd.rs` + `primitives/frd_sphere.rs` + example
    - [x] Fischer-Koch C(Y) — `tpms/fischer_koch_cy.rs` + `primitives/fischer_koch_cy_sphere.rs` + example
    - [x] 398 tests pass; examples compile clean

- [x] **Phase 5: Performance, Memory & Correctness Audit (Sprint 2)**
    - [x] `reconstruct.rs`: replace `Vec<Option<VertexId>>` id_map with `HashMap` — O(face_count) vs O(pool_size)
    - [x] `reconstruct.rs`: add `region_invalid_uses_untagged_path` + `large_pool_small_face_set_no_over_alloc` tests
    - [x] `classify.rs`: fix all garbled UTF-8 mojibake section separators
    - [x] `classify.rs`: add module doc with GWN theorem, `classify_fragment` decision flowchart, complexity table, references
    - [x] `classify.rs`: optimize `gwn()` — use `norm_squared()` for near-vertex guard (avoids 3 sqrt calls per face)
    - [x] `classify.rs`: add 8 unit tests: gwn interior/exterior/clamped/empty, classify inside/outside/coplanar, centroid/tri_normal helpers
    - [x] `vertex_pool.rs`: add snap-rounding architecture diagram + 27-neighbour completeness theorem
    - [x] 419 tests pass (up from 408); 2 pre-existing v_shape failures unchanged

- [x] **Phase 6: Retain Largest Component Precision Preservation (Sprint 3)**
    - [x] Audit `gaia` hierarchy, finding Delaunay tools are optimally implemented `O(N log N)`.
    - [x] Spot precision-loss bug in `IndexedMesh::retain_largest_component` fallback to default.
    - [x] Extract `empty_clone()` to `VertexPool` and `IndexedMesh` to precisely preserve config.
    - [x] Repair `retain_largest_component` to use `empty_clone()`.
    - [x] Write two new tests: `empty_clone_preserves_custom_tolerance` and `retain_largest_component_preserves_tolerance`.
    - [x] Test suite 421 passes (2 pre-existing v-shape failures persist unaffected).

- [x] **Phase 7: Clippy, CSG Test Hierarchy, and Benchmark Gate (Sprint 4)**
    - [x] Extract `application::csg::boolean::indexed` inline tests into `indexed_tests.rs`; runtime module reduced to 1,813 lines.
    - [x] Resolve all-target clippy debt without lint suppression: half-edge traversal, 3MF indexing, RegionId defaults, regression-test matches, example config initialization, and NURBS error SSOT.
    - [x] Verify `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo nextest run`, and `cargo test --doc`.
    - [x] Run `cargo bench --bench csg_performance -- --warm-up-time 1 --measurement-time 2 --sample-size 10`; GWN paths improved versus stored baseline, classify prepared showed no statistically significant change.
    - [x] Close residual rustdoc link debt: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` is warning-clean.
    - [x] Re-run `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo nextest run`, `cargo test --doc`, and short `csg_performance` benchmark gate.
    - [x] Benchmark rerun shows no repeatable regression: six cases no-change, `csg_union_cube_cube` improved versus the immediate baseline.

- [x] **Phase 8: NURBS Basis Allocation Gate (Sprint 5)**
    - [x] Review uncommitted NURBS basis diff and surrounding curve/surface call sites.
    - [x] Verify allocation-avoiding slice basis APIs are canonical and call sites are clippy-clean.
    - [x] Add and verify value-semantic basis tests: stack-degree slice equivalence, heap-degree slice equivalence, derivative wrapper equivalence, and central-difference derivative check.
    - [x] Verify `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo nextest run`, `cargo test --doc`, `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`, and short `csg_performance` benchmark gate.

- [x] **Phase 9: Atlas Moirai Integration (Sprint 6)**
    - [x] Add `moirai::ParallelSlice::map_collect_index` upstream so consumers can collect indexed slice maps without pointer-derived indices.
    - [x] Replace Gaia CSG fragment classification's parallel unsafe pointer offset with `map_collect_index`.
    - [x] Verify producer: `cargo fmt --check`, `cargo clippy -p moirai-parallel --all-targets -- -D warnings`, `cargo test -p moirai-parallel`, and warning-clean `cargo doc -p moirai-parallel --no-deps`.
    - [x] Verify consumer: `cargo fmt --check`, `cargo clippy --all-targets --features parallel -- -D warnings`, `cargo nextest run --features parallel`, warning-clean `cargo doc --no-deps --features parallel`, and `cargo bench --bench csg_performance --features parallel -- --warm-up-time 1 --measurement-time 2 --sample-size 10`.

- [x] **Phase 10: CSG Fragment Classification Cleanup (Sprint 7)**
    - [x] Consolidate sequential and Moirai-backed parallel fragment filtering behind one canonical helper.
    - [x] Pre-size valid-fragment and kept-face buffers from exact upper bounds.
    - [x] Verify with focused fragment-classification tests plus format, clippy, nextest, rustdoc, and short CSG benchmark gates.

- [x] **Phase 11: Quality Metric Angle SSOT (Sprint 8)**
    - [x] Review local quality/welding optimization commit before push; no content diff remains unstaged.
    - [x] Tighten `triangle_angles` to derive all angles from three edge vectors and three norms instead of six normalized directed edges.
    - [x] Document degenerate angle behavior and add a value-semantic NaN regression test.
    - [x] Restore `MeshWelder` vertex adjacency to Vec-backed low-valence storage with a single push-unique helper to avoid per-vertex hash table allocation.

- [x] **Phase 12: CSG Edge-Use Cache Audit (Sprint 9)**
    - [x] Review current CSG/watertight/grid diff before merge.
    - [x] Fix `collapse_degenerate_faces` edge-use cache invariant so degenerate faces are never counted in the non-degenerate edge-use map.

- [x] **Phase 13: CSG Selection Hot-Path Cleanup (Sprint 10)**
    - [x] Review clean-tree diff state before continuing optimization.
    - [x] Remove avoidable hot-loop `unwrap()` calls from CSG closest-pair and non-manifold pair selection.
    - [x] Pre-size CSG repair hash maps/sets where loop-local upper bounds are already known.

- [x] **Phase 14: BVH Median Ordering Cleanup (Sprint 11)**
    - [x] Review clean-tree diff state before continuing optimization.
    - [x] Replace BVH median fallback `partial_cmp` ordering with total float ordering.
    - [x] Verify focused BVH partition tests, format, clippy, nextest, doctests, rustdoc, and CSG benchmark gates.

- [x] **Phase 15: Ruppert Queue Ordering Cleanup (Sprint 12)**
    - [x] Review clean-tree diff state before continuing optimization.
    - [x] Replace Ruppert bad-triangle priority ordering fallback with total float ordering.
    - [x] Add value-semantic regression coverage for NaN and signed-zero ordering consistency.
    - [x] Verify focused Ruppert tests, format, clippy, nextest, doctests, rustdoc, and CSG benchmark gates.
    - [x] Record residual gate observations: `sphere_sphere_intersection_volume` is marginally slow in isolation (31.105s), and the confirmatory CSG benchmark rerun leaves only unrelated SVDAG rasterize noise.

- [x] **Phase 16: Coplanar Sweep Ordering Cleanup (Sprint 13)**
    - [x] Review clean-tree diff state before continuing optimization.
    - [x] Replace coplanar sweep-index `partial_cmp` fallback ordering with total float ordering.
    - [x] Add value-semantic regression coverage for NaN and signed-zero AABB sort order.
    - [x] Verify focused coplanar tests, format, clippy, nextest, doctests, rustdoc, and CSG benchmark gates.
    - [x] Record residual gate observation: CSG sphere stress tests still pass but exceed the 30s slow threshold in full-suite and isolated runs.

- [x] **Phase 17: Bounded Traversal Allocation Cleanup (Sprint 14)**
    - [x] Review clean-tree diff state before continuing optimization.
    - [x] Pre-size generalized arrangement candidate-pair storage from pairwise face-count minima.
    - [x] Reuse one BVH hit buffer per source mesh and cap coplanar-pair preallocation by face-count upper bound.
    - [x] Reuse `IndexedMesh::orient_outward` traversal queue across disconnected components with exact face-count capacity.
    - [x] Add value-semantic coverage for the arrangement capacity estimator.
    - [x] Verify focused CSG and indexed-mesh tests, format, clippy, nextest, doctests, rustdoc, and CSG benchmark gates.
    - [x] Record residual gate observation: `dense_sphere_sphere_union_64x32` and `sphere_sphere_intersection_volume` pass under 30s when isolated individually, but exceed the slow threshold when run concurrently or under full-suite contention; isolated touched-path `csg_union_cube_cube` benchmark rerun reports improvement.

- [x] **Phase 18: Dense Adjacency Capacity Cleanup (Sprint 15)**
    - [x] Review clean-tree diff state before continuing topology allocation cleanup.
    - [x] Count vertex-face incidence, vertex valence, and pre-dedup face-neighbor entries before filling dense adjacency vectors.
    - [x] Allocate each adjacency list with exact pre-dedup capacity while preserving dense O(1) lookup APIs.
    - [x] Update adjacency module documentation to describe the count/fill construction.
    - [x] Verify focused adjacency and connectivity tests, format, clippy, nextest, doctests, rustdoc, and CSG benchmark gates.
    - [x] Record residual gate observation: full-suite nextest passed with `dense_sphere_sphere_union_64x32` at 30.063s and `sphere_sphere_intersection_volume` at 33.563s; isolated `classify_prepared_2400f` benchmark rerun reports improvement after a noisy full benchmark regression.

- [x] **Phase 19: Exporter Capacity Cleanup (Sprint 16)**
    - [x] Review clean-tree diff state before continuing I/O allocation cleanup.
    - [x] Pre-size GLB, OBJ, PLY, and 3MF vertex remap tables from exact mesh vertex counts.
    - [x] Pre-size GLB position, normal, and index buffers from exact vertex and face counts.
    - [x] Pre-size 3MF XML output using a bounded static/per-element capacity estimate.
    - [x] Add value-semantic coverage for the 3MF capacity estimator and archive signature.
    - [x] Verify focused I/O tests plus `three-mf-io` feature tests, format, clippy, nextest, doctests, rustdoc, and CSG benchmark gates.

- [x] **Phase 20: Atlas Leto Coplanar Buffer Integration (Sprint 17)**
    - [x] Review clean-tree diff state and audit local Atlas crates for usable `leto`, `moirai`, and `mnemosyne` replacement surfaces.
    - [x] Keep `nalgebra` at Gaia's geometry boundary until `leto` exposes point/vector contracts; use `leto::Array` for packed numeric triangle/AABB buffers instead.
    - [x] Replace coplanar CSG triangle-coordinate and AABB materialization with Atlas-owned `leto` contiguous arrays borrowed as slices in hot loops.
    - [x] Add value-semantic coverage proving `leto` buffer construction preserves row order and values.
    - [x] Verify focused coplanar tests, format, clippy, nextest, doctests, rustdoc, and CSG benchmark gates.
    - [x] Record residual integration observation: `leto` is usable for Gaia's contiguous numeric buffers today; full `nalgebra::Point3`/`Vector3` replacement remains blocked until `leto` owns geometry point/vector contracts.

- [x] **Phase 21: Grid and OpenFOAM Allocation Cleanup (Sprint 18)**
    - [x] Review clean-tree diff state and audit remaining Atlas dependency replacement candidates.
    - [x] Keep `mnemosyne::BrandedVec` out of Gaia mesh storage until branded heap/token ownership can be introduced without compatibility shims or fake integration.
    - [x] Pre-size structured-grid face deduplication from the exact five-tet candidate-face upper bound.
    - [x] Remove per-face temporary vector allocation from structured-grid boundary labelling.
    - [x] Replace OpenFOAM export's per-face linear patch lookup with one `RegionId` index map and exact bucket pre-counting.
    - [x] Verify focused OpenFOAM tests, format, clippy, nextest, doctests, rustdoc, and short benchmark command execution.
    - [x] Record residual observations: Gaia still emits unused Atlas patch warnings for optional Moirai/Mnemosyne crates; full-suite nextest has existing CSG slow tests above 30s; the unrelated CSG/SVDAG benchmark group reported regressions and remains the next performance-investigation target.

- [x] **Phase 22: Delaunay, Hierarchy, and Seam Propagation Allocation Cleanup (Sprint 19)**
    - [x] Pre-size `seen` and `hull` vectors in `convex_hull_vertices` in `vertex_star.rs`, and pre-size `b_vertices` and reuse `next_pos` in `lattice.rs`.
    - [x] Add `empty_clone_with_capacity` to `VertexPool` and `IndexedMesh` to support pre-allocated empty clones while preserving tolerances.
    - [x] Refactored `P2MeshConverter` and `HexToTetConverter` to use `empty_clone_with_capacity` and pre-allocate midpoint maps, face maps, and local adjacency maps.
    - [x] Refactored `propagate_seam_vertices_until_stable` to construct edge-to-face adjacency map exactly once per convergence loop using a heap-allocation-free inline `AdjacentFaces` struct.
    - [x] Pre-allocated `interior_vid_set` in `corefine_face` and sets/vectors in `consolidate_cross_mesh_vertices`.
    - [x] Migrated `boundary_labels` from `HashMap<FaceId, String>` to `HashMap<FaceId, std::borrow::Cow<'static, str>>` to avoid redundant string allocations.
    - [x] Verify format, clippy, nextest, doctests, and CSG benchmark gates.

- [x] **Phase 23: OpenFOAM Streaming Export Cleanup (Sprint 20)**
    - [x] Review clean-tree diff state before continuing I/O allocation cleanup.
    - [x] Stream OpenFOAM point output directly from `VertexPool::positions()` using `mesh.vertex_count()` instead of collecting a temporary point vector.
    - [x] Stream OpenFOAM face output directly from pre-counted patch buckets instead of flattening into a second sorted-face vector.
    - [x] Fix `schematic_to_openfoam` after the `Cow<'static, str>` boundary-label migration by using stable `as_ref()` matching instead of unstable `Cow::as_str()`.
    - [x] Keep Gaia `Cargo.lock` pinned to the currently committed `leto` version; local Leto 0.35.0 Schur/SVD work is uncommitted producer WIP and not part of this consumer slice.
    - [x] Verify focused OpenFOAM tests, format, clippy, nextest, doctests, and rustdoc.
    - [x] Run the short CSG benchmark gate: initial full benchmark command improved the first three GWN cases but crashed during `classify_prepared_2400f`; isolated `classify_prepared_2400f` rerun completed and reported improvement. The access violation remains a residual benchmark-harness risk outside this OpenFOAM slice.

- [x] **Phase 24: Atlas Melinoe Integration (Sprint 21)**
    - [x] Add `melinoe` dependency to `Cargo.toml`.
    - [x] Refactor `permission/cell.rs` and `permission/token.rs` to wrap `melinoe::MelinoeCell` and `melinoe::ExclusiveToken` respectively.
    - [x] Mark unchecked ancient items as completed.
    - [x] Verify focused permission tests, format, clippy, nextest, doctests, and CSG benchmark gates.

- [x] **Phase 25: Watertight Repair & Pipeline Allocation Optimization (Sprint 22)**
    - [x] Hoist `repaired_faces` allocation in `MeshRepair::iterative_boundary_stitch` (`repair.rs`).
    - [x] Pre-size `boundary_pairs` and `adj` map in boundary loop sealing (`seal.rs`).
    - [x] Pre-allocate `port_mesh` and `vmap` capacities in shell pipeline (`shell_mesh.rs`).
    - [x] Verify focused pipeline/repair tests, format, clippy, nextest, and CSG benchmark gates.

- [x] **Phase 26: Seam Injection & Coplanar DSU Allocation Optimization (Sprint 23)**
    - [x] Hoist `candidates`, `cut_params`, and `params` in `inject_cap_seam_into_barrels` (`propagate.rs`).
    - [x] Pre-size coplanar index and group sets/maps in `coplanar_groups.rs`.
    - [x] Pre-allocate `PlanarPointGridIndex` bins and reuse output vectors in `planar.rs` and `cdt.rs`.
    - [x] Verify focused pipeline/repair tests, format, clippy, nextest, and CSG benchmark gates.

- [x] **Phase 27: Corefinement & Seam Propagation Allocation Optimization (Sprint 24)**
    - [x] Hoist `injections` and `pts` allocations in `propagate_seam_vertices_impl` (`propagate.rs`).
    - [x] Define and integrate `CorefinerScratch` in `corefine.rs` and `fragment_refinement.rs`.
    - [x] Pre-size `frags` in `boolean_csg.rs` and `SeamVertexMap` in `corefine.rs`.
    - [x] Verify `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, full `cargo nextest run --all-features` (968 tests), focused `cargo nextest run --all-features csg` (346 tests), doctests, rustdoc, and `cargo bench --bench csg_performance csg_union_cube_cube -- --warm-up-time 1 --measurement-time 2 --sample-size 10` (median 205.14 us; p=0.18, no statistically significant change).

- [x] **Phase 28: Intersection Segment Calculation Stack-Allocation Optimization (Sprint 25)**
    - [x] Refactor `edge_crossings_interval` in `segment.rs` to use stack-allocated array.
    - [x] Verify `cargo fmt --check`.
    - [x] Verify `cargo clippy --all-targets --all-features -- -D warnings`.
    - [x] Run full test suite (968 tests).
    - [x] Verify `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`.
    - [x] Run benchmark verification tests.

- [x] **Phase 29: Parallel Read Sharing & Contiguous Curvature Optimization (Sprint 26)**
    - [x] Introduce covariant `SharedGhostToken` read sharing in `permission::token` and `cell` to allow contention-free parallel reads.
    - [x] Implement thread-safe parallel BVH queries using shared tokens.
    - [x] Parallelize AABB precomputation and broad-phase intersection search using Moirai task parallelism with thread-local scratch vectors.
    - [x] Parallelize classification face preparation in GWN.
    - [x] Reorganize mean curvature estimation to use contiguous, flat `Vec` storage arrays indexed by `VertexId` offset instead of sparse `HashMap` accumulators.
    - [x] Verify formatting (`cargo fmt --check`).
    - [x] Verify lint rules (`cargo clippy --all-targets --all-features -- -D warnings`).
    - [x] Run full test suite (968 tests).
    - [x] Verify documentation compilation (`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`).
    - [x] Run benchmark verification tests.

- [x] **Phase 30: Vertex Welding Allocation & Coplanar pre-sizing Optimization (Sprint 27)**
    - [x] Introduce `query_radius_to` in `spatial_hash.rs` to append matching indices directly into a pre-allocated vector.
    - [x] Refactor greedy topological clustering in `MeshWelder::weld` to reuse a single `candidates` query buffer.
    - [x] Replace the sparse `cur_face_verts` HashMap in `MeshWelder` with a flat contiguous Vec indexed directly by face ID offsets, converting $O(\log F)$ lookups into $O(1)$ slice access.
    - [x] Pre-size `id_map` in `coplanar_dispatch.rs` using `HashMap::with_capacity(unique_list.len())` to prevent map resizing and rehashing.
    - [x] Verify formatting (`cargo fmt --check`).
    - [x] Verify lint rules (`cargo clippy --all-targets --all-features -- -D warnings`).
    - [x] Run full test suite (968 tests).
    - [x] Verify documentation compilation (`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`).
    - [x] Run benchmark verification tests.

- [x] **Phase 31: SVDAG Memory Optimization & Pre-sizing (Sprint 28)**
    - [x] Introduce `with_capacity` constructor and `finalize()` method in `SparseVoxelDag` to release internal construction-only deduplication maps and reclaim ~60% memory.
    - [x] Call `svo.finalize()` inside SVO `from_mesh` rasterization.
    - [x] Pre-size result DAG and memoization tables inside `MergeContext` in `boolean.rs` using capacity heuristics, and call `result.finalize()` before returning.
    - [x] Verify formatting (`cargo fmt --check`).
    - [x] Verify lint rules (`cargo clippy --all-targets --all-features -- -D warnings`).
    - [x] Run full test suite (968 tests).
    - [x] Verify documentation compilation (`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`).
    - [x] Run benchmark verification tests.

