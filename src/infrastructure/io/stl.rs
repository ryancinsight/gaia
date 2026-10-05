//! STL import and export.
//!
//! Supports both ASCII and binary STL formats.

use std::io::{BufRead, BufReader, Read, Write};

use eunomia::NumericElement;

use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Real, Scalar, Vector3r};
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::{FaceData, FaceStore};
use crate::infrastructure::storage::vertex_pool::VertexPool;

use super::parse;

/// Write an indexed mesh as ASCII STL.
///
/// Generic over the scalar type `T`; coordinates are written as `f64` so the
/// output precision is independent of the input scalar width.
///
/// # Errors
///
/// Returns [`MeshError::Io`] if writing any header, facet, or vertex record
/// to `writer` fails.
pub fn write_ascii_stl<W: Write, T: Scalar>(
    writer: &mut W,
    name: &str,
    vertex_pool: &VertexPool<T>,
    face_store: &FaceStore,
) -> MeshResult<()> {
    writeln!(writer, "solid {name}").map_err(MeshError::Io)?;

    for (_, face) in face_store.iter_enumerated() {
        let a = vertex_pool.position(face.vertices[0]);
        let b = vertex_pool.position(face.vertices[1]);
        let c = vertex_pool.position(face.vertices[2]);

        let normal =
            crate::domain::geometry::normal::triangle_normal(a, b, c).unwrap_or_else(|| {
                let mut z = leto::geometry::Vector3::zeros();
                z.z = <T as crate::domain::core::scalar::Scalar>::from_f64(1.0);
                z
            });

        writeln!(
            writer,
            "  facet normal {:.7} {:.7} {:.7}",
            <T as NumericElement>::to_f64(normal.x),
            <T as NumericElement>::to_f64(normal.y),
            <T as NumericElement>::to_f64(normal.z),
        )
        .map_err(MeshError::Io)?;
        writeln!(writer, "    outer loop").map_err(MeshError::Io)?;
        for p in [&a, &b, &c] {
            writeln!(
                writer,
                "      vertex {:.7} {:.7} {:.7}",
                <T as NumericElement>::to_f64(p.x),
                <T as NumericElement>::to_f64(p.y),
                <T as NumericElement>::to_f64(p.z),
            )
            .map_err(MeshError::Io)?;
        }
        writeln!(writer, "    endloop").map_err(MeshError::Io)?;
        writeln!(writer, "  endfacet").map_err(MeshError::Io)?;
    }

    writeln!(writer, "endsolid {name}").map_err(MeshError::Io)?;
    Ok(())
}

/// Write an indexed mesh as binary STL.
///
/// Generic over the scalar type `T`; vertex coordinates and normals are stored
/// as `f32` per the binary STL specification.
///
/// # Errors
///
/// Returns [`MeshError::Io`] if writing the header, triangle count, triangle
/// payload, or attribute bytes to `writer` fails.
///
/// # Panics
///
/// Panics if the triangle count exceeds the binary STL `u32` header field.
pub fn write_binary_stl<W: Write, T: Scalar>(
    writer: &mut W,
    vertex_pool: &VertexPool<T>,
    face_store: &FaceStore,
) -> MeshResult<()> {
    // 80-byte header
    let header = [0u8; 80];
    writer.write_all(&header).map_err(MeshError::Io)?;

    // Number of triangles
    let n_triangles = u32::try_from(face_store.len()).expect("triangle count fits in u32");
    writer
        .write_all(&n_triangles.to_le_bytes())
        .map_err(MeshError::Io)?;

    for (_, face) in face_store.iter_enumerated() {
        let a = vertex_pool.position(face.vertices[0]);
        let b = vertex_pool.position(face.vertices[1]);
        let c = vertex_pool.position(face.vertices[2]);

        let normal =
            crate::domain::geometry::normal::triangle_normal(a, b, c).unwrap_or_else(|| {
                let mut z = leto::geometry::Vector3::zeros();
                z.z = <T as crate::domain::core::scalar::Scalar>::from_f64(1.0);
                z
            });

        // Normal (3 × f32) — eunomia's to_f32() is the explicit precision-reduction path
        write_f32(writer, normal.x.to_f32())?;
        write_f32(writer, normal.y.to_f32())?;
        write_f32(writer, normal.z.to_f32())?;

        // Vertices (3 × 3 × f32)
        for p in [&a, &b, &c] {
            write_f32(writer, p.x.to_f32())?;
            write_f32(writer, p.y.to_f32())?;
            write_f32(writer, p.z.to_f32())?;
        }

        // Attribute byte count
        writer
            .write_all(&0u16.to_le_bytes())
            .map_err(MeshError::Io)?;
    }

    Ok(())
}

/// Read an ASCII STL into the vertex pool and face store.
///
/// # Errors
///
/// Returns [`MeshError::Io`] if the input cannot be read line-by-line,
/// [`MeshError::Other`] if a vertex field is not a valid number, and
/// [`MeshError::InvalidCoordinate`] if a parsed vertex contains NaN or
/// infinity.
pub fn read_ascii_stl<R: Read>(
    reader: R,
    vertex_pool: &mut VertexPool,
    face_store: &mut FaceStore,
    region: RegionId,
) -> MeshResult<usize> {
    let buf = BufReader::new(reader);
    let mut count = 0usize;
    let mut verts: Vec<Point3r> = Vec::with_capacity(3);
    // Position of the next `vertex` record in the file, so a non-finite
    // coordinate is reported against the file rather than against a mesh id
    // that welding has not assigned yet.
    let mut ordinal = 0usize;

    for line in buf.lines() {
        let line = line.map_err(MeshError::Io)?;
        let trimmed = line.trim();

        if trimmed.starts_with("vertex") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                verts.push(parse::parse_point([parts[1], parts[2], parts[3]], ordinal)?);
                ordinal += 1;
            }
        }

        if trimmed.starts_with("endfacet") && verts.len() == 3 {
            let normal =
                crate::domain::geometry::normal::triangle_normal(&verts[0], &verts[1], &verts[2])
                    .unwrap_or_else(Vector3r::z);

            let v0 = vertex_pool.insert_or_weld(verts[0], normal);
            let v1 = vertex_pool.insert_or_weld(verts[1], normal);
            let v2 = vertex_pool.insert_or_weld(verts[2], normal);

            face_store.push(FaceData {
                vertices: [v0, v1, v2],
                region,
            });

            count += 1;
            verts.clear();
        }
    }

    Ok(count)
}

/// Write a single f32 in little-endian.
fn write_f32<W: Write>(w: &mut W, v: f32) -> MeshResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(MeshError::Io)
}

// =============================================================================
//  Low-level binary STL reader (to VertexPool + FaceStore)
// =============================================================================

/// Read a binary STL into the vertex pool and face store.
///
/// Binary STL format: 80-byte header, u32 triangle count, then for each
/// triangle: 12-byte normal, 3 × 12-byte vertices, 2-byte attribute count.
/// Vertex normals are recomputed from face geometry rather than read from the
/// file (the spec does not require them to be correct).
///
/// # Errors
///
/// Returns [`MeshError::Io`] if the binary records cannot be read,
/// or [`MeshError::InvalidCoordinate`] if any decoded vertex coordinate is NaN
/// or infinite.
pub fn read_binary_stl<R: Read>(
    reader: R,
    vertex_pool: &mut VertexPool,
    face_store: &mut FaceStore,
    region: RegionId,
) -> MeshResult<usize> {
    let mut buffered_reader = BufReader::new(reader);
    let mut header = [0u8; 80];
    buffered_reader
        .read_exact(&mut header)
        .map_err(MeshError::Io)?;
    let mut count_bytes = [0u8; 4];
    buffered_reader
        .read_exact(&mut count_bytes)
        .map_err(MeshError::Io)?;
    let triangle_count = u32::from_le_bytes(count_bytes) as usize;

    for triangle in 0..triangle_count {
        // Skip the stored normal (12 bytes) — we recompute it.
        let mut skip = [0u8; 12];
        buffered_reader
            .read_exact(&mut skip)
            .map_err(MeshError::Io)?;

        let mut verts = [Point3r::new(0.0, 0.0, 0.0); 3];
        for (index, vert) in verts.iter_mut().enumerate() {
            let mut vbuf = [0u8; 12];
            buffered_reader
                .read_exact(&mut vbuf)
                .map_err(MeshError::Io)?;
            let x_coord = Real::from(f32::from_le_bytes([vbuf[0], vbuf[1], vbuf[2], vbuf[3]]));
            let y_coord = Real::from(f32::from_le_bytes([vbuf[4], vbuf[5], vbuf[6], vbuf[7]]));
            let z_coord = Real::from(f32::from_le_bytes([vbuf[8], vbuf[9], vbuf[10], vbuf[11]]));
            // `f32::from_le_bytes` accepts every bit pattern, so the NaN and
            // infinity encodings are reachable from a well-formed 50-byte
            // record; the ordinal is this vertex's position in the file.
            *vert = parse::finite_point(
                Point3r::new(x_coord, y_coord, z_coord),
                triangle.saturating_mul(3).saturating_add(index),
            )?;
        }
        // Skip attribute byte count (2 bytes).
        let mut attr = [0u8; 2];
        buffered_reader
            .read_exact(&mut attr)
            .map_err(MeshError::Io)?;

        let normal =
            crate::domain::geometry::normal::triangle_normal(&verts[0], &verts[1], &verts[2])
                .unwrap_or_else(Vector3r::z);
        let v0 = vertex_pool.insert_or_weld(verts[0], normal);
        let v1 = vertex_pool.insert_or_weld(verts[1], normal);
        let v2 = vertex_pool.insert_or_weld(verts[2], normal);
        face_store.push(FaceData {
            vertices: [v0, v1, v2],
            region,
        });
    }
    Ok(triangle_count)
}

// =============================================================================
//  High-level IndexedMesh helpers
// =============================================================================

/// Read an STL file (auto-detecting ASCII vs binary) into a new [`IndexedMesh`].
///
/// Detection is based on the binary record-size invariant:
/// `file_bytes == 84 + triangle_count * 50`.  Any file that satisfies this
/// is parsed as binary; everything else is attempted as ASCII.
/// All faces are tagged with `RegionId(0)` (wall).
///
/// # Errors
///
/// Returns the same read or coordinate-validation errors as
/// [`read_binary_stl`] or [`read_ascii_stl`], depending on the detected format.
pub fn read_stl<R: Read>(reader: R) -> MeshResult<IndexedMesh> {
    let mut data = Vec::new();
    // read_to_end needs the Read trait in scope — it is via `use std::io::Read`.
    BufReader::new(reader)
        .read_to_end(&mut data)
        .map_err(MeshError::Io)?;

    let region = RegionId::from_usize(0);
    let mut mesh = IndexedMesh::new();

    let is_binary = data.len() >= 84
        && data.len()
            == 84 + u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize * 50;

    if is_binary {
        read_binary_stl(
            std::io::Cursor::new(data),
            &mut mesh.vertices,
            &mut mesh.faces,
            region,
        )?;
    } else {
        read_ascii_stl(
            std::io::Cursor::new(data),
            &mut mesh.vertices,
            &mut mesh.faces,
            region,
        )?;
    }
    Ok(mesh)
}

/// Write an [`IndexedMesh`] as ASCII STL (convenience wrapper).
///
/// Generic over scalar `T`; existing callers with `IndexedMesh<f64>` are unchanged.
///
/// # Errors
///
/// Returns [`MeshError::Io`] if emitting the ASCII STL stream fails.
pub fn write_stl_ascii<W: Write, T: Scalar>(
    writer: &mut W,
    name: &str,
    mesh: &IndexedMesh<T>,
) -> MeshResult<()> {
    write_ascii_stl(writer, name, &mesh.vertices, &mesh.faces)
}

/// Write an [`IndexedMesh`] as binary STL (convenience wrapper).
///
/// Generic over scalar `T`; existing callers with `IndexedMesh<f64>` are unchanged.
///
/// # Errors
///
/// Returns [`MeshError::Io`] if emitting the binary STL stream fails.
pub fn write_stl_binary<W: Write, T: Scalar>(
    writer: &mut W,
    mesh: &IndexedMesh<T>,
) -> MeshResult<()> {
    write_binary_stl(writer, &mesh.vertices, &mesh.faces)
}

// =============================================================================
//  Fuzz entry point
// =============================================================================

/// Fuzz entry point for STL parsing.
///
/// Accepts arbitrary bytes and attempts to parse them as STL.  This function
/// must **never panic** — all errors are returned as `Err`.  Suitable as the
/// inner body of a `cargo-fuzz` target.
///
/// # Example (in a fuzz target)
/// ```rust,no_run
/// #![no_main]
/// // libfuzzer_sys::fuzz_target!(|data: &[u8]| {
/// //     let _ = gaia::infrastructure::io::stl::fuzz_read_stl(data);
/// // });
/// ```
///
/// # Errors
///
/// Returns the same parse and coordinate-validation errors as [`read_stl`].
pub fn fuzz_read_stl(data: &[u8]) -> MeshResult<IndexedMesh> {
    read_stl(std::io::Cursor::new(data))
}

#[cfg(test)]
#[path = "tests_stl.rs"]
mod tests;
