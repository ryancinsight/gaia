# Backlog

Shared state and ownership board for `gaia-mesh` (import path `gaia`).
`CHECKLIST.md` holds owner-local execution steps; this file holds priority,
status, and acceptance. One fact has one owner.

Schema per item: **outcome**, **scope / non-goals**, **acceptance oracle**,
**dependencies**, **risk / change class**, **status**, **owner**.

Seeded 2026-08-20 by the Atlas gap audit (`atlas-gap-audit`) at
`4980732`. Every item cites the evidence that opened it.

Triage order: correctness → security → architecture required for correctness →
missing verification → documentation drift → PM cleanup.

---

## GAIA-001 — Derive or replace the GWN classification band

- **Outcome**: `GWN_INSIDE_THRESHOLD` (0.65) and `GWN_OUTSIDE_THRESHOLD` (0.35)
  either carry a written derivation bounding the misclassification probability
  for the supported operand class, or are replaced by a derived, scale-aware
  criterion. `classify_fragment`'s tiebreaker band stops being an unexplained
  literal at the one place the README admits exactness ends.
- **Scope**: `src/domain/core/constants.rs:95-102`,
  `src/application/csg/arrangement/classify.rs`,
  `src/application/csg/arrangement/gwn.rs`. Non-goals: changing the exact
  predicate boundary; re-tuning the constants to make a failing case pass
  (that is the prohibited empirical hack).
- **Acceptance oracle**: the constant's Rustdoc carries a derivation in the
  form already used by `GWN_SOLID_ANGLE_CLIP`
  (`src/domain/core/constants.rs:86-93`), plus an adversarial test whose inputs
  sit inside the band and whose expected classification comes from an
  independent oracle (exact orient/insphere sign or analytic membership), not
  from the current implementation's output.
- **Dependencies**: none.
- **Risk / change class**: [correctness] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `README.md` "Where exactness ends"; `src/domain/core/constants.rs:96`
and `:102` document what the constants *do* and never why those values;
the sibling constant at `:86-93` shows the standard this repo already holds.

---

## GAIA-002 — Native-precision 3-D predicate boundary

- **Outcome**: the 3-D Bowyer-Watson kernel stops funnelling every coordinate
  through `f64`, or the `f32` instantiation is removed from the public
  contract so no caller can request a precision the kernel does not honour.
- **Scope**: `src/application/delaunay/dim3/tetrahedralize.rs:33-40`
  (`point_to_f64_arr`), `src/domain/geometry/predicates.rs:100-104` (`fn r`),
  `src/application/csg/predicates3d.rs`. Non-goals: reimplementing Shewchuk;
  the `geometry-predicates` dependency stays.
- **Acceptance oracle**: either (a) an `IndexedMesh<f32>` tetrahedralization
  test asserting the Delaunay empty-circumsphere property holds under the
  native predicate path with a derived `f32` error bound, or (b) a compile-time
  bound that makes the 3-D kernel `f64`-only, with the README precision
  contract updated to match in the same change.
- **Dependencies**: none.
- **Risk / change class**: [arch] [minor] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `src/application/delaunay/dim3/tetrahedralize.rs:20-24` states the
gap in its own module docs; `README.md` § 2 "Precision contract" repeats it;
`docs/mesh_library_gap_audit.md` tracks it. It is disclosed, not closed.

---

## GAIA-003 — Collapse the `Real` alias onto the `Scalar` seam

- **Outcome**: the `T: Scalar` seam is the crate's actual precision contract
  rather than one of two parallel ones. `Real` survives only as a caller-facing
  default type parameter, not as the type 848 internal sites are written
  against.
- **Scope**: `src/domain/core/scalar.rs:114-120` and its consumers. Non-goals:
  removing `Point3r`/`Vector3r` as public defaults; a big-bang rewrite — this
  burns down per module family, each increment green.
- **Acceptance oracle**: measured drop in concrete-`Real` sites (baseline 848
  from `rg -c '\bReal\b' src`), and a matching drop in the
  `cast_precision_loss` / `cast_possible_truncation` / `cast_sign_loss`
  ratchet counts in `Cargo.toml` (baseline 759 / 297 / 99). The ratchet
  counts only decrease.
- **Dependencies**: GAIA-002 fixes the same seam at the predicate boundary and
  should land first for the 3-D kernel.
- **Risk / change class**: [arch] [patch] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `src/domain/core/scalar.rs:114` `pub type Real = f64;`; 848 `Real`
sites vs 130 `T: Scalar` sites; `IndexedMesh` appears 431× without a type
argument vs 36× as `IndexedMesh<T>`. The Cargo.toml ratchet block names the
same root cause: "the real fix is a `Scalar`-parameterized conversion
boundary, not blanket `as` casts".

---

## GAIA-004 — Near-degenerate predicate regression suite

- **Outcome**: the exactness claim in `predicates.rs` is verified at gaia's own
  boundary, not assumed from the dependency.
- **Scope**: `src/domain/geometry/predicates.rs` tests,
  `src/domain/topology/predicates.rs`. Non-goals: testing
  `geometry-predicates` itself; the target is gaia's wrappers, its sign
  convention negation, and its `Orientation::from_det` mapping.
- **Acceptance oracle**: a differential test that evaluates each wrapper
  against a naive `f64` determinant on inputs constructed to make the naive
  form return the wrong sign (collinear-to-within-ULP triples, cospherical
  point sets, coordinates spanning many binades), asserting the wrapper's sign
  and that the naive form disagrees — the disagreement is what proves the
  check is live. Plus a `should_panic`-free property test that
  `orient_3d(a,b,c,d) == -orient_3d(a,b,d,c)` under permutation.
- **Dependencies**: none.
- **Risk / change class**: [verification] [patch] — S.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `src/domain/geometry/predicates.rs:255-330` — every existing
predicate test uses unit-scale, well-separated or exactly-zero inputs
(`(0,0)`,`(1,0)`,`(0,1)`). None would fail if the wrappers called naive `f64`
arithmetic, so none tests the property the module header claims.

---

## GAIA-005 — Retire the 39 ignored doctests

- **Outcome**: the public API's documented examples compile and run, so
  `cargo test --doc` is a real contract gate instead of covering 8 of 47
  examples.
- **Scope**: the 36 ```` ```rust,ignore ```` and 3 ```` ```ignore ```` fences
  across `src/`. Non-goals: ```` ```text ```` diagram blocks (97 of them),
  which are prose, not examples.
- **Acceptance oracle**: `rg -c '```rust,ignore|```ignore' src` reaches 0, or
  each surviving `ignore` is converted to `no_run` / `compile_fail` with a
  stated reason; `cargo test --doc --all-features` reports a runnable count
  matching the public item count in the touched modules.
- **Dependencies**: none. Burns down per module.
- **Risk / change class**: [verification] [docs] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: 36 `rust,ignore` + 3 `ignore` fences vs 4 `rust` + 1 `no_run`;
`CHECKLIST.md` Phase 52 records "All eight runnable doctests pass; 39
additional doctests remain intentionally ignored". Examples that never compile
rot silently — `src/lib.rs:12-19` and `src/domain/core/scalar.rs:46-52` are
both `rust,ignore`.

---

## GAIA-006 — Six example files are not build targets

- **Outcome**: every `.rs` file under `examples/` is either a declared target
  that CI compiles, or deleted.
- **Scope**: `examples/csg/{union,difference,intersection,compound}.rs`,
  `examples/primitives/{serpentine_tube,y_junction}.rs`, and the
  `[[example]]` table in `Cargo.toml`. Non-goals: adding new examples;
  `examples/well_plate_schematic.png` is a separate output-hygiene item
  (GAIA-009).
- **Acceptance oracle**: `find examples -name '*.rs' | wc -l` equals the
  `[[example]]` count in `Cargo.toml` (currently 69 vs 63), and the
  `module_reachability` gate is extended to cover `examples/` so the class
  cannot silently reappear.
- **Dependencies**: none.
- **Risk / change class**: [verification] [patch] — S.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `Cargo.toml` sets `autoexamples = false` and declares 63
`[[example]]` paths; 69 `.rs` files exist under `examples/`. The six above
appear in no target, so `cargo clippy --all-targets` never sees them.
`tests/module_reachability.rs:8` scans `src/` only.

---

## GAIA-007 — Miri gate for the GhostCell `Send`/`Sync` impls

- **Outcome**: the crate's only two `unsafe` items are covered by the
  verification the deeper-gate rule requires for reachable unsafe.
- **Scope**: `src/infrastructure/permission/cell.rs:34` and `:40`;
  a `miri` job in `.github/workflows/ci.yml` under a nightly verification
  toolchain, with the pinned build toolchain unchanged.
- **Acceptance oracle**: `cargo +nightly miri nextest run` (or `miri test`
  over the permission and half-edge modules) is green in CI and covers a test
  that exercises branded aliasing across threads, not only construction.
- **Dependencies**: none.
- **Risk / change class**: [verification] [patch] — S.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `rg 'unsafe ' src` returns exactly `cell.rs:34` and `cell.rs:40`
(`unsafe impl Send`/`Sync` for `GhostCell`); `.github/workflows/ci.yml` runs
fmt, clippy, nextest, doctests and doc — no miri step.

---

## GAIA-008 — Supply-chain and semver gates

- **Outcome**: CI enforces the checks the stack's engineering gates require for
  a published crate: advisory/licence/ban scanning, unused-dependency
  detection, and public-surface semver classification.
- **Scope**: `.github/workflows/ci.yml`, plus a `deny.toml`. Non-goals:
  changing the dependency set; the gates report first, remediation is its own
  item.
- **Acceptance oracle**: `cargo deny check`, `cargo machete`, and
  `cargo semver-checks check-release` run in CI; the semver job is required on
  any PR touching `pub` surface and gates `rust-release.yml`.
- **Dependencies**: none.
- **Risk / change class**: [verification] [security] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `.github/workflows/ci.yml` job `gate` has exactly five steps (fmt,
clippy, nextest, doctests, doc). `CHECKLIST.md` records semver comparisons run
by hand ("196 checks passed, 57 skipped") — a manual sequence performed more
than twice is a mechanization defect.

---

## GAIA-009 — Committed run outputs and undeclared features

- **Outcome**: `outputs/` stops being a tracked 5.8 MB directory of generated
  meshes, OpenFOAM cases and STL dumps; only reviewed golden figures survive,
  under a named golden-fixture path. Separately, `stl-io` and `millifluidic`
  either gate something or are deleted, and the README install example stops
  advertising an inert feature.
- **Scope**: `outputs/` (106 tracked files), `.gitignore`, the `[features]`
  table in `Cargo.toml`, `README.md` § Installation. Non-goals: deleting the
  four reviewed `outputs/book-review/*.png` renders, which the book's figure
  manifests cite as review evidence.
- **Acceptance oracle**: `git ls-files outputs | wc -l` covers only the
  manifest-cited golden renders; the generating examples are documented as the
  regeneration path; `rg 'feature = "stl-io"|feature = "millifluidic"' src`
  is non-empty or the features are gone from `Cargo.toml`, with the README's
  `features = ["stl-io", "vtk-io"]` example corrected in the same change.
- **Dependencies**: none.
- **Risk / change class**: [pm-hygiene] [docs] [patch] — S.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `git ls-files outputs` returns 106 files across
`schematic_to_openfoam/`, `millifluidic/`, `millifluidic_chip_stl/`,
`schematic_to_3d/`, `frustum_e2e_test/` (5.8 MB); `Cargo.toml` already
excludes the directory from the published package, which is the packaging
half of the fix and not the tracking half. `rg 'feature = "stl-io"' src` and
`rg 'feature = "millifluidic"' src` both return zero hits, while
`src/infrastructure/io/mod.rs:1-7` compiles `stl` unconditionally.

---

## GAIA-010 — Retroactive ADRs for the decided architecture

- **Outcome**: the decisions the README already presents as settled have
  records, so a cold-start agent finds them by index instead of by prose.
- **Scope**: `docs/adr/`, `docs/adr/README.md` (generated by
  `scripts/adr-index.py`). Candidate subjects, each an as-built Accepted ADR
  grounded strictly in current code: the GhostCell brand + slotmap topology
  seam; the `Scalar` sealed-trait precision seam and the `Real` default; the
  exact-predicate boundary and where exactness ends (GWN); the CSG operand
  normalization band `0.5..=10.0`; the GAIA-LINT-1 ratchet as the conformance
  policy. Non-goals: inventing rationale — backfill is trigger-driven, one ADR
  per item that touches its scope, never a speculative sweep.
- **Acceptance oracle**: each new ADR cites its board item and the code it
  describes; `python scripts/adr-index.py check` passes; no ADR restates a
  README paragraph without naming the rejected alternative.
- **Dependencies**: GAIA-001/002/003 each carry one of the candidates.
- **Risk / change class**: [docs] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `docs/adr/README.md` indexes two ADRs (0001 channel-path validation,
0002 boundary quality criteria) for a 73 214-line kernel whose README § Core
Architecture presents six distinct architectural decisions as settled.

---

## GAIA-011 — Domain book: teach the geometry, not only the contract

- **Outcome**: the book teaches robust computational geometry from the ground
  up — exact predicate arithmetic, the Delaunay empty-circumsphere property
  and Ruppert termination, winding-number membership, manifold/Euler
  invariants — and then maps gaia's abstractions onto that theory. The current
  contract and gallery chapters stay; they become the applied layer.
- **Scope**: `docs/book/SUMMARY.md` and new chapters. Non-goals: migrating
  `CHECKLIST.md` status prose into the book; figures stay generated by
  `src/bin/book_mesh_gallery`, never hand-assembled.
- **Acceptance oracle**: `mdbook test` passes with runnable samples in each new
  chapter; each theory chapter cites a resolved primary reference (Shewchuk
  1997 for the predicates, Ruppert 1995 for refinement, Jacobson et al. for the
  generalized winding number) with a locator, and states its domain of
  validity.
- **Dependencies**: GAIA-001 (the GWN chapter needs the derived band).
- **Risk / change class**: [docs] [patch] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `docs/book/SUMMARY.md` lists six chapters totalling 289 lines —
mesh-generation contract, Atlas ownership, gallery, two figure manifests, and
watertightness diagnostics. All describe what gaia guarantees; none derives
why the algorithms hold.

---

## GAIA-012 — Benchmark and example runtime budgets

- **Outcome**: the three criterion benches and the CI-safe examples carry
  enforced finite budgets, so a performance regression fails a gate instead of
  being noticed by hand.
- **Scope**: `.github/workflows/ci.yml`, `benches/{csg_performance,
  hex_to_tet_performance, tpms_performance}.rs`. Non-goals: adding benches;
  changing bench workloads to fit a budget (that is instrument tuning).
- **Acceptance oracle**: CI smoke-runs the bench binaries in single-iteration
  mode (`cargo test --benches` / criterion `--test`) inside the committed
  30 s nextest budget, and the CI-safe examples run within it; a committed
  per-binary wall-clock bound exists for full timing runs.
- **Dependencies**: GAIA-006 (example targets must exist before they can be
  budgeted).
- **Risk / change class**: [perf] [verification] [patch] — S.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `.config/nextest.toml` budgets tests only; `.github/workflows/ci.yml`
never invokes a bench or example target beyond `clippy --all-targets`
type-checking, so no bench body has ever been executed by a gate.

---

## GAIA-013 — GAIA-LINT-1 ratchet burn-down and file-size debt

- **Outcome**: the counted allow-list in `Cargo.toml` shrinks monotonically and
  the 42 files past the 500-line target are split along operation-family lines.
- **Scope**: the `[lints.clippy]` ratchet table in `Cargo.toml`; the largest
  offenders first —
  `src/application/csg/arrangement/adversarial_tests.rs` (2324),
  `src/application/csg/boolean/indexed.rs` (2046),
  `src/domain/mesh/indexed.rs` (1379),
  `src/application/csg/corefine.rs` (1131). Non-goals: mechanical slicing that
  breaks domain cohesion; raising any count.
- **Acceptance oracle**: each increment lowers at least one measured count in
  the ratchet table and never raises one; the `too_many_lines` count (101) and
  the >500-line file count (42) both fall. The ratchet's own comment claims
  "4 production sites remain" for `unwrap_used`; the current count is 2
  (`src/application/quality/normals.rs:231`, `:243`) — correct the stale
  number in the first increment that touches the table.
- **Dependencies**: GAIA-003 retires the largest class (1268 cast lints).
- **Risk / change class**: [arch] [patch] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `Cargo.toml` `[lints.clippy]` holds ~2400 measured hits across 30
allowed classes; `find src -name '*.rs' | xargs wc -l` shows 42 files over
500 lines.

---

## GAIA-014 — Edition 2024

- **Outcome**: the crate builds on edition 2024, gaining
  `unsafe_op_in_unsafe_fn`, let-chains, and the current resolver behaviour the
  stack's other members already assume.
- **Scope**: `Cargo.toml` `edition`, plus whatever `cargo fix --edition`
  surfaces. Non-goals: raising the pinned toolchain (1.97.0 already supports
  it).
- **Acceptance oracle**: `cargo clippy --all-targets --all-features -- -D
  warnings` and the full nextest suite pass on edition 2024 with no new
  ratchet entries; `manual_let_else` (51 allowed) drops as let-chains land.
- **Dependencies**: none.
- **Risk / change class**: [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `Cargo.toml:10` `edition = "2021"`; `rust-toolchain.toml` pins
1.97.0.

---

## GAIA-015 — Constrained 3-D refinement and remeshing (carried forward)

- **Outcome**: the two P1 capability gaps the repo's own 2026-08-04 audit
  recorded are either delivered behind a consumer-driven acceptance contract or
  explicitly declared out of scope in the README so no reader infers them.
- **Scope**: `docs/mesh_library_gap_audit.md` rows "3-D constrained
  refinement" and "Remeshing/repair"; `src/application/delaunay/dim3/`.
  Non-goals: adding a public refinement API before the predicate contract,
  sizing-field contract, feature-protection criteria, and termination gates
  are specified — that gate is already recorded in `CHECKLIST.md` Phase 52.
- **Acceptance oracle**: either a sizing-field + radius-edge refinement loop
  with a proven termination bound and a boundary-feature protection test, or a
  README scope statement naming both as non-goals with the consumer driver
  that would reopen them.
- **Dependencies**: GAIA-002 (predicate contract), GAIA-001 (tolerance
  discipline).
- **Risk / change class**: [arch] [minor] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: re-verified 2026-08-20 against the current tree —
`rg -li 'remesh|decimat|advancing_front|SizingField' src` returns zero files;
`sliver` appears only in CSG fragment classification and quality *measurement*,
never in an optimization pass. The audit's finding still holds.
