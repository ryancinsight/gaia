# Backlog

Shared state and ownership board for `gaia-mesh` (import path `gaia`).
This board is the ordered queue of open work. GAIA-022 folds the legacy
`CHECKLIST.md` execution state into this queue and removes the duplicate.

Schema per item: **outcome**, **priority**, **needs**, **scope**,
**acceptance oracle**, and **next step**; audited items also record **basis**.

Seeded 2026-08-20 by the Atlas gap audit (`atlas-gap-audit`) at
`4980732`. Every item cites the evidence that opened it.

Triage order: correctness → architecture → verification → tightening → feature.

---

<a id="GAIA-003"></a>
## GAIA-003 — Collapse the `Real` alias onto the `Scalar` seam

- status: todo

- outcome: `T: Scalar` is the internal precision contract; `Real` remains only
  as the public default type parameter.
- priority: architecture
- needs: none; the predicate seam landed in PR #73 / ADR 0005.
- scope: `src/domain/core/scalar.rs` and its consumer modules; preserve public
  defaults and migrate one module family per verified increment.
- acceptance: eliminate internal `Real` sites (initial 848, current baseline
  793 hits across 98 Rust files) and reduce cast ratchets to zero (initial
  759/297/99, current 437/160/53); each slice updates callers and passes its
  gate.
- basis: `e43c3e6` (current main; `Real` references remeasured).
- next: make the CSG arrangement generic over the PSLG scalar.

Progress (Phase 19, 2026-09-30):
- Added `Scalar::from_usize(n: usize)` and `Scalar::from_index(k: i64)` to
  the `Scalar` trait — eunomia-backed replacement for `i as Real` index casts.
  Implemented via `<Self as Scalar>::from_f64(n as f64)` following eunomia's
  `FloatElement` widening-seam contract. Documented with IEEE-754 exactness
  bound; 37 doctests pass.
- Added aequitas-typed physical-quantity constructors to `constants.rs`:
  `length_m()`, `length_mm()`, `angle_rad()`, `angle_deg()`, and default
  channel dimension accessors using `aequitas::systems::si::quantities`.
- Phase 19 subagent replacing 56+ `as Real` index casts with `from_usize()`.

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
- **Status**: todo. Delivered Phase 14 (2026-09-29): two cross-thread tests added to `cell.rs` (Send/Sync theorems documented inline); `miri` CI job added under nightly toolchain with `-Zmiri-strict-provenance`. **Owner**: root.

Evidence: `rg 'unsafe ' src` returns exactly `cell.rs:34` and `cell.rs:40`
(`unsafe impl Send`/`Sync` for `GhostCell`); `.github/workflows/ci.yml` now
includes a `miri` job targeting `infrastructure::permission::cell::tests`.

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
- **Status**: todo. Incrementally delivered — Phase 13 (2026-09-29) removed four lint classes and discharged `unwrap_used`; remaining: cast-precision ratchet (blocked by GAIA-003), file-size splits, `too_many_lines` 58. **Owner**: root.

Progress (Phase 24, 2026-10-01):
- Split 13 `too_many_lines` violators across geometry primitives, GLTF/PLY export, branching tubes, arrangement patching, indexed orientation repair, and indexed-merge compaction.
- Forced-warning `too_many_lines` baseline reduced 30→18; `Cargo.toml` ratchet comment remeasured to 18.
- `cargo clippy --lib` is clean; `cargo test --lib` passed 1139/1139.

Progress (Phase 25, 2026-10-01):
- Split 10 remaining complex CSG `too_many_lines` violators across arrangement, coplanar, and indexed-repair code paths without touching the concurrently edited forbidden files.
- Forced-warning `too_many_lines` baseline reduced 18→8; `Cargo.toml` ratchet comment remeasured to 8.
- `cargo clippy --lib` is clean; `cargo test --lib` passed 1140/1140.
- Full `cargo fmt --check` is still blocked by formatting diffs in the concurrently edited forbidden files (`boundary.rs`, `helix_sweep.rs`, `stadium_prism.rs`, `ply.rs`), so this phase only formatted the touched CSG files directly.

Progress (Phase 26, 2026-10-02):
- Split `src/application/csg/corefine/face.rs` into documented helpers for snap-segment dedup, edge/interior Steiner collection, boundary fallback preparation, PSLG construction, and CDT face assembly; `corefine_face` is now below the `too_many_lines` threshold and no longer emits a forced-warning diagnostic.
- Forced-warning `too_many_lines` measurement reduced 9→5 via the Phase 26 command; `Cargo.toml` ratchet comment remeasured to 5.
- `cargo clippy --lib` is clean and `cargo test --lib` passed 1139/1139. `cargo fmt --check` is still blocked by parent-owned formatting diffs in `src/application/csg/arrangement/seam.rs`.
- `src/application/csg/boolean/indexed/repair/merge.rs` remains on the residual list, but this pass left it untouched because that file is concurrently owned by the parent process.

Evidence: `cargo clippy --all-targets --all-features -- --force-warn
clippy::too_many_lines` emits 58 diagnostics at this revision; `unwrap_production`
ratchet baseline auto-tightened 2→0 on push.

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
- **Status**: todo. Delivered with GAIA-013 / Phase 13. **Owner**: root.

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

- Status: todo. Delivered Phase 13 (test assertions) + Phase 17 (full discharge). `float_cmp = "allow"` removed from ratchet; 5 production sites carry `#[expect]` with documented reasons (tie-breaking, identity check, edge guard, convex hull pivot); test-code sites covered by `#![cfg_attr(test, expect(clippy::float_cmp))]` in lib.rs. Count: 44 → 25 (P13) → 0 (P17). **Owner**: root.
- priority: verification

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

<a id="GAIA-022"></a>
## GAIA-022 — Consolidate the legacy execution checklist

- status: todo
- outcome: every open deliverable has one complete record in this queue.
- priority: verification
- needs: none
- scope: `CHECKLIST.md`, `backlog.md`, and docs linking to checklist anchors.
- acceptance: revalidate every unchecked entry against the tree and hosting;
  map it to an existing item or file a complete item; remove stale links;
  delete `CHECKLIST.md`; leave no references to it.
- basis: `e43c3e6` (current main).
- next: compare each unchecked entry with the current tree, backlog, and PRs.


<a id="GAIA-023"></a>
## GAIA-023 — Unformatted landings outpace fmt fixes; enforce the gate on the pushing side

- status: todo
- outcome: `main` stays green on `cargo fmt --check` for longer than one landing cycle, because unformatted code cannot reach it.
- priority: correctness
- needs: none
- scope: `.githooks/pre-push` installation and bypass discipline; no source changes.
- acceptance: three consecutive `main` pushes each pass the fmt job with no fmt-fix commit between them.
- basis: `eb98a58` (current main).
- evidence (2026-09-30, measured): `main` flip-flopped green 15:31 -> red 15:36 on fresh `Diff in` lines (`multi_mesh_resolution.rs:377`, `merge.rs:100/150/320/328/353`) from code landed in between; an earlier round (tube/collapse/merge/welder/constants) was fixed the same way and superseded within the hour. A file-level fmt PR cannot converge while landings outpace it — closed gaia#130 unmerged for exactly this reason after verifying its content was correct but stale. The hook carries a documented `SKIP_LOCAL_GATE=1` escape and installs opt-in per checkout (`install-hooks`), so either path lets unformatted code through.
- next: decide the enforcement (mandatory hook install via the setup path, removing or gating the skip, or a required fmt status check that blocks merge rather than reporting after it) and implement that decision; do not open another file-level fmt PR.

<a id="GAIA-RESCUE-QUEUE"></a>
## GAIA-RESCUE-QUEUE — Complete or close the open rescue PRs

- status: todo
- outcome: each listed rescue PR is completed onto current main (ported, verified, merged) or closed once its diff resolves empty against main.
- priority: correctness
- needs: none
- scope: the rescue head refs below; the board of this repository.
- acceptance: no open `rescue/` PR in `ryancinsight/gaia` remains unaccounted for.
- basis: `07611f2` (current main).
- queue (draft parking records, none a claim):
  - `ryancinsight/gaia#134` (`rescue/audit-gaia-20260928`): one commit splitting oversized modules and extracting test bodies; 98 files, +13284/-13215, code (csg, delaunay, geometry, quality, storage); fails `cargo fmt --check`.
  - `ryancinsight/gaia#120` (`rescue/gaia-audit-20260929`): tracked and untracked work from a detached audit tree plus the rescue-only pre-push exception; 97 files, +13287/-13204, near-identical file set to #134; resolve against #134 first.
  - `ryancinsight/gaia#121` (`rescue/gaia-arch008-20260929`): ARCH008 branch tip plus dirty backlog and repair hunks; 3 files, +12/-16 (`.githooks/pre-push`, `backlog.md`, `csg/boolean/indexed/repair/vertices.rs`).
  - `ryancinsight/gaia#123` (`rescue/gaia-manifest-parallel-20260929`): `moirai-runtime` dependency enables `parallel`; 2 files, +1/-1 (`Cargo.toml`, `csg/arrangement.rs`); verification blocked by a Moirai/Mnemosyne API mismatch.
- next: classify #123 and #121 first (small), then diff #134 against #120 and port the surviving union onto main.
