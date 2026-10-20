#!/usr/bin/env python3
"""Resolve a lock-free Gaia consumer and verify its real 3MF archive.

The consumer lives outside Atlas's development overlay and starts without a
lockfile. Python's independent ZIP/XML readers check the exported model, and
the resolved graph must carry the first-party `consus-zip` container and no
third-party `zip` crate.
The same probe accepts a normalized `cargo package` directory as its input.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET
import zipfile


CONSUMER = """use gaia::domain::core::scalar::Point3r;
use gaia::infrastructure::io::three_mf::write_3mf;
use gaia::IndexedMesh;
use std::io::Cursor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh = IndexedMesh::new();
    let a = mesh.add_vertex_pos(Point3r::new(3.0, 0.0, 0.0));
    let b = mesh.add_vertex_pos(Point3r::new(0.0, 4.0, 0.0));
    let c = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 5.0));
    mesh.add_face(a, b, c);
    let mut writer = Cursor::new(Vec::new());
    write_3mf(&mut writer, &mesh)?;
    std::fs::write(std::env::args().nth(1).expect("archive output path"), writer.into_inner())?;
    Ok(())
}
"""


def verify_archive(path):
    model_ns = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == [
            "3D/3dmodel.model", "[Content_Types].xml", "_rels/.rels"
        ]
        assert archive.testzip() is None, "all archive CRCs must match"
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
        types = ET.fromstring(archive.read("[Content_Types].xml"))
        assert {item.attrib["Extension"]: item.attrib["ContentType"] for item in types} == {
            "rels": "application/vnd.openxmlformats-package.relationships+xml",
            "model": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        }
        relationships = ET.fromstring(archive.read("_rels/.rels"))
        assert len(relationships) == 1
        assert relationships[0].attrib == {
            "Target": "/3D/3dmodel.model", "Id": "rel-1",
            "Type": "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel",
        }
        model = ET.fromstring(archive.read("3D/3dmodel.model"))
        vertices = model.findall(f".//{model_ns}vertex")
        assert [tuple(float(v.attrib[axis]) for axis in "xyz") for v in vertices] == [
            (3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 5.0)
        ]
        triangles = model.findall(f".//{model_ns}triangle")
        assert [tuple(int(t.attrib[key]) for key in ("v1", "v2", "v3")) for t in triangles] == [
            (0, 1, 2)
        ]
        assert model.find(f"{model_ns}build/{model_ns}item").attrib == {"objectid": "1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=Path("Cargo.toml"))
    args = parser.parse_args()
    manifest = args.manifest_path.resolve()
    toolchain = tomllib.loads((manifest.parent / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    cargo = ["cargo", f"+{toolchain}"]
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", str(manifest.parent / "target"))
    with tempfile.TemporaryDirectory(prefix="gaia-fresh-consumer-") as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "Cargo.toml").write_text(
            '[package]\nname = "gaia-fresh-consumer"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n[dependencies]\n'
            'gaia = { package = "gaia-mesh", path = ' + json.dumps(str(manifest.parent)) + ' }\n'
        )
        (root / "src/main.rs").write_text(CONSUMER)
        output = root / "triangle.3mf"
        subprocess.run(cargo + ["run", "--quiet", "--", str(output)], cwd=root,
                       env=env, check=True, timeout=300)
        verify_archive(output)
        metadata = json.loads(subprocess.check_output(
            # Metadata includes dependencies for targets the native run did
            # not build, so a cold runner may still need their sources.
            cargo + ["metadata", "--locked", "--format-version=1"],
            cwd=root, env=env, timeout=60,
        ))
        versions = {}
        for package in metadata["packages"]:
            versions.setdefault(package["name"], []).append(package["version"])
        assert "zip" not in versions, versions["zip"]
        container = versions.get("consus-zip", [])
        assert len(container) == 1, container
        print(f"Fresh consumer and independent 3MF payload checks pass with consus-zip {container[0]}.")


if __name__ == "__main__":
    main()
