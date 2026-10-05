use super::*;
use crate::test_support::assert_rejects;

// ── ASCII round-trip ──────────────────────────────────────────────────

#[test]
fn ascii_stl_round_trip_indexed() {
    let mut mesh = IndexedMesh::new();
    let v0 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
    let v1 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
    let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
    mesh.add_face(v0, v1, v2);

    let mut buf = Vec::new();
    write_stl_ascii(&mut buf, "test", &mesh).unwrap();

    let mesh2 = read_stl(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(mesh2.face_count(), 1);
    assert_eq!(mesh2.vertex_count(), 3);
}

// ── Binary round-trip ─────────────────────────────────────────────────

#[test]
fn binary_stl_round_trip_indexed() {
    let mut mesh = IndexedMesh::new();
    let v0 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
    let v1 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
    let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
    mesh.add_face(v0, v1, v2);

    let mut buf = Vec::new();
    write_stl_binary(&mut buf, &mesh).unwrap();

    let mesh2 = read_stl(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(mesh2.face_count(), 1);
    assert_eq!(mesh2.vertex_count(), 3);
}

// ── f32 mesh export ───────────────────────────────────────────────────

#[test]
fn ascii_stl_exports_f32_mesh() {
    use leto::geometry::Point3;

    let mut mesh = IndexedMesh::<f32>::new();
    let v0 = mesh.add_vertex_pos(Point3::new(0.0_f32, 0.0, 0.0));
    let v1 = mesh.add_vertex_pos(Point3::new(1.0_f32, 0.0, 0.0));
    let v2 = mesh.add_vertex_pos(Point3::new(0.0_f32, 1.0, 0.0));
    mesh.add_face(v0, v1, v2);

    let mut buf = Vec::new();
    write_stl_ascii(&mut buf, "f32-test", &mesh).unwrap();
    let s = std::str::from_utf8(&buf).unwrap();
    assert!(s.contains("solid f32-test"));
    assert!(s.contains("vertex"));
}

#[test]
fn binary_stl_exports_f32_mesh() {
    use leto::geometry::Point3;

    let mut mesh = IndexedMesh::<f32>::new();
    let v0 = mesh.add_vertex_pos(Point3::new(0.0_f32, 0.0, 0.0));
    let v1 = mesh.add_vertex_pos(Point3::new(1.0_f32, 0.0, 0.0));
    let v2 = mesh.add_vertex_pos(Point3::new(0.0_f32, 1.0, 0.0));
    mesh.add_face(v0, v1, v2);

    let mut buf = Vec::new();
    write_stl_binary(&mut buf, &mesh).unwrap();
    // Binary STL: 80-byte header + 4-byte count + 1 triangle × 50 bytes = 134
    assert_eq!(buf.len(), 134);
}

/// A one-triangle binary STL whose first vertex has x-coordinate `first_x`.
///
/// Exactly `84 + 1 * 50` bytes, which is the invariant `read_stl` uses to
/// decide a file is binary.
fn one_triangle_binary_stl(first_x: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(134);
    data.extend_from_slice(&[0u8; 80]); // header
    data.extend_from_slice(&1_u32.to_le_bytes()); // triangle count
    data.extend_from_slice(&[0u8; 12]); // the stored normal, which is ignored
    for (x, y, z) in [
        (first_x, 0.0_f32, 0.0_f32),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ] {
        data.extend_from_slice(&x.to_le_bytes());
        data.extend_from_slice(&y.to_le_bytes());
        data.extend_from_slice(&z.to_le_bytes());
    }
    data.extend_from_slice(&[0u8; 2]); // attribute byte count
    data
}

/// `f32::from_le_bytes` accepts every bit pattern, so the NaN and infinity
/// encodings are reachable from a well-formed 50-byte record. The binary
/// reader parses no text and so had no syntax check that could fail.
#[test]
fn binary_stl_non_finite_vertex_is_an_error() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let data = one_triangle_binary_stl(value);
        assert_rejects(
            &read_stl(std::io::Cursor::new(&data[..])),
            "invalid coordinate at vertex 0",
        );
    }
}

#[test]
fn ascii_stl_non_finite_vertex_is_an_error() {
    for spelling in ["nan", "inf", "-inf"] {
        let body = format!(
            "solid t\nfacet normal 0 0 1\nouter loop\n\
             vertex {spelling} 0 0\nvertex 1 0 0\nvertex 0 1 0\n\
             endloop\nendfacet\nendsolid t\n"
        );
        assert_rejects(
            &read_stl(std::io::Cursor::new(body.as_bytes())),
            "invalid coordinate at vertex 0",
        );
    }
}

// ── Fuzz entry point never panics ─────────────────────────────────────

#[test]
fn fuzz_target_handles_empty_input() {
    let result = fuzz_read_stl(b"");
    // May succeed (empty mesh) or return an error — must not panic.
    let _ = result;
}

#[test]
fn fuzz_target_handles_truncated_binary() {
    let result = fuzz_read_stl(&[0u8; 84]);
    let _ = result;
}
