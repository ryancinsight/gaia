# Gaia

Gaia is the standalone repository for the `cfd-mesh` Rust crate.

`cfd-mesh` provides watertight CFD mesh generation primitives, indexed and half-edge mesh representations, exact geometric predicates, Boolean CSG operations, quality analysis, and mesh I/O.

## Layout

- `crates/cfd-mesh`: the library crate published and consumed as `cfd-mesh`

## Features

- `parallel`: rayon-backed mesh operations
- `stl-io`, `vtk-io`, `three-mf-io`: format-specific I/O
- `millifluidic`: millifluidic-specific builders
- `cfdrs-integration`: optional bridge layer for CFDrs schematic and blueprint types

## Build

```sh
cargo check -p cfd-mesh --lib --no-default-features
cargo check -p cfd-mesh --examples --no-default-features
```

To validate the optional CFDrs bridge:

```sh
cargo check -p cfd-mesh --lib --no-default-features --features cfdrs-integration
cargo check -p cfd-mesh --examples --no-default-features --features cfdrs-integration
```
