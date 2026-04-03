# Gaia

Watertight CFD mesh generation for millifluidic devices. Built entirely in Rust, with an emphasis on exactly-computable geometry operations rather than floating-point approximations.

## Features
- **CSG Boolean Operations**: Exact precision Union, Intersection, and Difference
- **Delaunay Corefinement**: Topologically guaranteed triangulations
- **Watertight Output**: Guaranteed manifold topologies, avoiding the non-manifold edges typical of standard CAD kernels
- **Multiple Supported Formats**: Import and Export functionality for STL and VTK (legacy ASCII).

## Development Strategy
The exact architectural roadmap and requirements can be found in `agents.md`.
