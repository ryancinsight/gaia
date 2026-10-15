#!/usr/bin/env python3
"""Reproduce the Gaia 0.5.3 -> 0.6 public provider-trait boundary.

First run `cargo publish --dry-run --locked -p gaia-mesh` on a clean tree.
This probe uses that normalized package, not source overrides or mock traits.
Requires Python 3.9+ and Rust 1.97.0. Neither case publishes anything.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(command, cwd, timeout, env=None):
    result = subprocess.run(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        timeout=timeout, check=False,
    )
    return result.returncode, result.stdout


def metadata(cargo, directory):
    code, output = run(
        cargo + ["metadata", "--no-deps", "--offline", "--format-version=1"],
        directory, 60,
    )
    if code:
        raise RuntimeError("Cargo metadata failed:\n" + output)
    return json.loads(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packaged-source", type=Path)
    parser.add_argument("--expected-source", help="Require this package VCS SHA")
    parser.add_argument("--evidence-dir", type=Path, help="Keep manifests, locks, and logs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cargo = ["cargo", "+1.97.0"]
    packaged = args.packaged_source
    if packaged is None:
        target = Path(metadata(cargo, root)["target_directory"])
        packaged = target / "package" / "gaia-mesh-0.6.0"
    packaged = packaged.resolve()
    package = next(p for p in metadata(cargo, packaged)["packages"] if p["name"] == "gaia-mesh")
    if package["version"] != "0.6.0":
        raise RuntimeError("Expected the normalized Gaia 0.6.0 package")
    dependencies = {d["name"]: d for d in package["dependencies"] if d["kind"] is None}
    for name, requirement in [("leto", "^0.46.0"), ("eunomia", "^0.10.0")]:
        dependency = dependencies[name]
        if dependency["req"] != requirement or not dependency["source"].startswith("registry+"):
            raise RuntimeError("Expected the coherent normalized registry provider: " + name)
    vcs = json.loads((packaged / ".cargo_vcs_info.json").read_text(encoding="utf-8"))
    if vcs["git"].get("dirty", False):
        raise RuntimeError("The normalized package must come from a clean Git tree")
    expected_source = args.expected_source
    if expected_source is None:
        code, output = run(["git", "rev-parse", "HEAD"], root, 60)
        if code == 0:
            expected_source = output.strip()
    if expected_source and vcs["git"]["sha1"] != expected_source:
        raise RuntimeError("The package VCS source does not match --expected-source")
    fixture = (root / "scripts/regressions/scalar_provider_boundary.rs").read_bytes()
    result = {"baseline": "gaia-mesh 0.5.3", "candidate": "gaia-mesh 0.6.0",
              "candidate_vcs": vcs, "fixture_sha256": hashlib.sha256(fixture).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="gaia-provider-boundary-") as temporary:
        work = Path(temporary)
        env = dict(os.environ, CARGO_TARGET_DIR=str(work / "target"))
        for case in ["baseline", "candidate"]:
            directory = work / case
            (directory / "src").mkdir(parents=True)
            (directory / "src/main.rs").write_bytes(fixture)
            target = 'version = "=0.5.3"' if case == "baseline" else "path = " + json.dumps(str(packaged))
            manifest = (
                '[package]\nname = "gaia-scalar-boundary-probe"\nversion = "0.0.0"\n'
                'edition = "2024"\npublish = false\n\n[workspace]\n\n[dependencies]\n'
                'gaia = { package = "gaia-mesh", ' + target + ' }\n'
                'eunomia = "=0.9.2"\n'
            )
            (directory / "Cargo.toml").write_text(manifest, encoding="utf-8")
            code, output = run(cargo + ["build"], directory, 300, env)
            (directory / "build.log").write_text(output, encoding="utf-8")
            result[case + "_build_exit"] = code
            if case == "baseline":
                if code:
                    raise RuntimeError("Published baseline did not build:\n" + output)
                code, output = run(cargo + ["run", "--locked", "--offline"], directory, 60, env)
                (directory / "run.log").write_text(output, encoding="utf-8")
                result["baseline_run_exit"] = code
                if code:
                    raise RuntimeError("Published baseline assertions failed:\n" + output)
            elif not (code == 101 and "error[E0277]" in output and "old_numeric(value)" in output
                      and "multiple different versions of crate `eunomia`" in output
                      and "due to 1 previous error" in output):
                raise RuntimeError("The expected public RealField identity loss was not reproduced:\n" + output)
        if args.evidence_dir:
            args.evidence_dir.mkdir(parents=True, exist_ok=True)
            saved = Path(tempfile.mkdtemp(prefix="run-", dir=args.evidence_dir.resolve()))
            for case in ["baseline", "candidate"]:
                shutil.copytree(work / case, saved / case)
            result["evidence"] = str(saved)
        result["conclusion"] = "The public Scalar supertrait identity changes; Gaia 0.6 is required."
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
