# Checklist

- [x] **Phase 1: Diagnosis & Analysis**
    - [x] Run `examples/csg_cube_sphere.rs` and capture output/error.
    - [x] Audit `src/application/csg/bsp.rs` for correct plane classification logic.
    - [x] Audit `src/application/csg/split.rs` for triangle splitting numerical stability.
- [x] Check `src/application/csg/boolean` for correct set operation logic (Union, Intersection, Difference) and decompose flat file.

- [x] **Phase 2: Core Boolean Logic Fixes**
    - [ ] Robustify plane/triangle intersection tests (e.g., epsilon handling).
    - [ ] Fix re-triangulation of split polygons (ensure CCW winding/normals).
    - [x] Handle "on-plane" cases correctly (coplanar surfaces) and decompose logic.
    - [x] Coplanar cube–cylinder CSG (caps flush with walls): union, intersection, difference — all watertight with <1% volume error.
    - [x] Fix O(n²) coplanar Difference bottleneck: AABB per-fragment pre-screening → 64-seg completes in 8ms.

- [ ] **Phase 3: Tooling & Verification**
    - [x] Create `inspect_stl` binary — watertightness, volume, normals, aspect ratio, bounds.
    - [ ] Add "watertight" check (edge manifoldness). *(done inside inspect_stl)*
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
