# ADR 0011: Version the Leto 0.51 geometry boundary

- Status: Accepted
- Date: 2026-11-19
- Change class: [major]

## Context

Gaia 0.6.10 exposes Leto points and vectors and requires Aequitas 0.3.
A standalone caller using Leto 0.51 points and vectors fails with E0308
when it passes them to the published Gaia plane constructor. With the
merged Gaia and Leto source pair, the same caller constructs the plane
through `(0,0,3)` with normal `(0,0,2)` and obtains offset −3.

## Decision

Release Gaia 0.7.0 with Leto 0.51.0, Aequitas 0.4.0, and Eunomia 0.12.1.
Keep library and fuzz lockfiles on the same provider revisions. Consumers
that share Gaia geometry or scalar traits advance their provider
requirements together. The algorithms and 3MF archive representation
remain the same.

A patch release would preserve a version range whose callers can supply
incompatible geometry types. Re-exporting the old types would keep two
competing definitions at a public boundary.

## Verification

Compare with the published 0.6.10 API using cargo-semver-checks. Verify
the clean package and the existing native, documentation, and Miri
checks. After publishing, resolve the exact registry versions in a new
Cargo project and rerun the plane construction and 3MF archive checks.

## Aequitas 0.5 registry revision

Gaia 0.8 carries the Leto 0.52 and Aequitas 0.5 identities already selected
by the integrated source. The original 0.7 registry archive retains its
published requirements. RITK and other registry consumers advance to 0.8
for the single-provider graph. Geometry, file formats, and numerical
acceptance tests follow the same contracts.
