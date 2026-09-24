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
- **Dependencies**: none — the generic predicate seam landed with GAIA-002
  (PR #73; ADR 0005).
- **Risk / change class**: [arch] [patch] — L.
- **Status**: todo. Draft PR #78 carries the NURBS derivative-overflow
  correction. Owner: root.

Evidence: `src/domain/core/scalar.rs:114` `pub type Real = f64;`; 848 `Real`
sites vs 130 `T: Scalar` sites; `IndexedMesh` appears 431× without a type
argument vs 36× as `IndexedMesh<T>`. The Cargo.toml ratchet block names the
same root cause: "the real fix is a `Scalar`-parameterized conversion
boundary, not blanket `as` casts".

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
- **Dependencies**: GAIA-003 carries the remaining precision candidate; the
  predicate boundary and GWN band landed as ADRs 0005 and 0004.
- **Risk / change class**: [docs] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `docs/adr/README.md` indexes five ADRs (0001 channel-path validation,
0002 boundary quality criteria, 0003 indexed CSG repair modules, 0004 GWN
thresholds, 0005 predicate boundary) for a 73 214-line kernel whose README § Core
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
- **Dependencies**: none; the GWN band decision is recorded in ADR 0004.
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

<a id="GAIA-013"></a>
## GAIA-013 — GAIA-LINT-1 ratchet burn-down and file-size debt

- **Outcome**: the counted allow-list in `Cargo.toml` shrinks monotonically and
  the source files past the 500-line target are split along operation-family
  lines.
- **Scope**: the `[lints.clippy]` ratchet table in `Cargo.toml`; the largest
  offenders first —
  `src/application/csg/arrangement/adversarial_tests.rs` (2324),
  `src/application/csg/boolean/indexed.rs` (2076 at branch base),
  `src/domain/mesh/indexed.rs` (1490),
  `src/application/csg/corefine.rs` (1131). Non-goals: mechanical slicing that
  breaks domain cohesion; raising any count.
- **Acceptance oracle**: each increment lowers at least one measured count in
  the ratchet table and never raises one; this increment lowers forced-warning
  `too_many_lines` emissions from 101 to 100 and source files over 500 lines
  from 41 at branch base to 40. Correct stale measurements when touching the
  table: `too_many_arguments` is 14 emissions, and `unwrap_used` has two
  production sites (`src/application/quality/normals.rs:235`, `:247`).
- **Dependencies**: GAIA-003 retires the largest class (1268 cast lints).
- **ADR**: 0003 — Align CSG repair module ownership with directory paths.
- **Risk / change class**: [arch] [patch] — L.
- **Status**: todo. In review — the delivery PRs are the hosting fact. **Owner**: root.
- **Lease**: root — `Cargo.toml`, `rustfmt.toml`, `examples/csg/cube_cube.rs`, `examples/debug_stl.rs`, `src/application/csg/arrangement/adversarial_tests.rs`, `src/application/csg/arrangement/boolean_csg.rs`, `src/application/csg/boolean/indexed.rs`, `src/application/csg/boolean/indexed/csg.rs`, `src/application/csg/boolean/indexed/repair/`, `src/application/csg/boolean/indexed/repair/mod.rs`, `src/application/csg/boolean/indexed_tests.rs`, `docs/adr/README.md`, `docs/adr/0003-indexed-csg-repair-modules.md`, `src/application/csg/clip/polygon2d/cdt.rs`, `src/application/csg/corefine.rs`, `src/application/delaunay/dim2/pslg/graph.rs`, `src/application/delaunay/dim2/smoothing/laplacian.rs`, `src/application/delaunay/dim2/triangulation/bowyer_watson.rs`, `src/application/delaunay/dim3/tetrahedralize.rs`, `src/application/hierarchy/hex_to_tet.rs`, `src/application/quality/normals.rs`, `src/application/welding/welder.rs`, `src/bin/book_mesh_gallery/render.rs`, `src/domain/topology/orientation.rs`, `src/infrastructure/spatial/ssvdag/boolean.rs`, `src/infrastructure/spatial/ssvdag/core.rs`, `src/infrastructure/spatial/ssvdag/rasterize.rs`, `src/infrastructure/storage/attribute.rs`, `src/infrastructure/storage/edge_store.rs`, `src/infrastructure/storage/vertex_pool.rs` — `2026-09-23T08:28:43-04:00`.

Evidence: `cargo clippy --all-targets --all-features -- --force-warn
clippy::too_many_lines` emits 100 diagnostics at this revision; Cargo.toml at
the branch base recorded 101. The source inventory had 41 files over 500
lines at branch base and has 40 now.

---

<a id="GAIA-014"></a>
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
- **Status**: todo. In review — the delivery PRs are the hosting fact. **Owner**: root.

Evidence: delivered with [GAIA-013](#GAIA-013): the manifest now uses edition
2024 and resolver 3 with the pinned Rust 1.97.0 toolchain. Strict all-target
Clippy and the full nextest suite pass.

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
- **Dependencies**: none — the predicate contract landed with GAIA-002
  (PR #73).
- **Risk / change class**: [arch] [minor] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: re-verified 2026-08-20 against the current tree —
`rg -li 'remesh|decimat|advancing_front|SizingField' src` returns zero files;
`sliver` appears only in CSG fragment classification and quality *measurement*,
never in an optimization pass. The audit's finding still holds.

---

<a id="GAIA-016"></a>
## GAIA-016 — Pin exact floating-point test values

- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: exact IEEE-value contracts compare representations without emitting
  `clippy::float_cmp` diagnostics.
- priority: verification
- needs: none
- scope: `src/application/quality/normals.rs`, `src/domain/core/scalar.rs`,
  `src/domain/geometry/nurbs/knot.rs`; test assertions only.
- acceptance: strict all-target Clippy emits 44 `float_cmp` diagnostics, the
  recorded ceiling does not rise, and affected tests pass.
- basis: `34f0229`
- next: deliver in PR #86.

<a id="GAIA-017"></a>
## GAIA-017 — Parse float-comparison diagnostics

- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: Cargo JSON diagnostics become validated records with workspace-local
  source and macro-expansion chains.
- priority: verification
- needs: GAIA-016
- scope: `scripts/float_cmp_diagnostics.py` and focused tests.
- acceptance: malformed messages fail closed; external packages are excluded;
  local absolute, relative, generated, and expanded sources have exact tests.
- basis: `34f0229`
- next: implement the parser and its value-semantic tests under 400 changed lines.

<a id="GAIA-018"></a>
## GAIA-018 — Reject hidden float-comparison diagnostics

- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: Rust lint attributes cannot conceal the measured diagnostic class.
- priority: verification
- needs: GAIA-016
- scope: `scripts/float_cmp_suppressions.py` and focused tests.
- acceptance: comments and strings are ignored; broad allows and non-local
  `float_cmp` suppression are rejected; narrow item expectations are accepted.
- basis: `34f0229`
- next: implement the scanner and adversarial lexical cases under 400 lines.

<a id="GAIA-019"></a>
## GAIA-019 — Enforce the float-comparison ceiling

- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: the manifest ceiling is machine-checked against diagnostics and its
  accepted base value.
- priority: verification
- needs: GAIA-017, GAIA-018
- scope: `scripts/float_cmp_budget.py` manifest, base, and count rules plus tests.
- acceptance: missing, duplicate, malformed, or raised ceilings fail; 44
  emissions pass and 45 fail; shallow-history base resolution is tested.
- basis: `34f0229`
- next: implement the pure budget rules and CLI-free tests under 400 lines.

<a id="GAIA-020"></a>
## GAIA-020 — Gate float-comparison growth in CI

- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: the repository gate runs strict Clippy once and rejects diagnostic or
  suppression growth on pull requests and main.
- priority: verification
- needs: GAIA-019
- scope: `.github/workflows/ci.yml`, Cargo execution in
  `scripts/float_cmp_budget.py`, and shared helpers in `scripts/lockfile.py`.
- acceptance: the script reports the 44-emission inventory, checks the base
  ceiling and suppressions, and the hosted gate passes on its exact revision.
- basis: `34f0229`
- next: integrate the runner and workflow under 400 changed lines.

<a id="GAIA-021"></a>
## GAIA-021 — Validate finite rational NURBS weights

- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: rational curves and surfaces reject non-finite weights at their
  construction boundary while preserving the positive-weight contract.
- priority: correctness
- needs: GAIA-003
- scope: `src/domain/geometry/nurbs/{curve,surface}.rs`, public construction
  error enums, tests, Rustdoc, and the breaking-change migration guide.
- acceptance: NaN and positive infinity return typed exhaustive error variants;
  finite positive weights remain accepted; cargo-semver-checks classifies the
  enum change and a major-version migration documents exhaustive-match updates.
- basis: `34f0229`
- next: specify the public error migration before implementation.
