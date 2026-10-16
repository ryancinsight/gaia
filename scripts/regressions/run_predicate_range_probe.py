"""Reproduce GAIA-026 against Cargo-JSON-selected real libraries.

First build unchanged Gaia with --locked --lib --message-format=json, saving
stdout to a JSONL file and stderr separately. Pass that JSONL here. The script
returns the regression test exit code (101 while the demonstrated defects are
unrepaired); observer/provider logs retain every value even when tests fail.
No dependency selection, production source or normal test setting is changed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts-json', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--optimized', action='store_true')
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
artifacts = {}
for line in args.artifacts_json.read_text().splitlines():
    message = json.loads(line)
    if message.get('reason') != 'compiler-artifact':
        continue
    name = message['target']['name']
    if name not in ['gaia', 'leto', 'geometry_predicates']:
        continue
    for filename in message['filenames']:
        if filename.endswith('.rlib'):
            artifacts[name] = filename
assert set(artifacts) == {'gaia', 'leto', 'geometry_predicates'}, artifacts
(args.output_dir / 'artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
source = Path(__file__).parent
results = {}
for name, filename, libraries, flags in [
    ('observe', 'predicate_range_probe.rs', ['gaia', 'leto'], []),
    ('regressions', 'predicate_range_probe.rs', ['gaia', 'leto'], ['--test']),
    ('provider', 'predicate_provider_probe.rs', ['gaia', 'geometry_predicates'], []),
]:
    executable = args.output_dir / name
    command = ['rustc', '+1.97.0', '--edition=2024', str(source / filename),
               '-L', 'dependency=' + str(Path(artifacts['leto']).parent)]
    if args.optimized:
        command.append('-O')
    for library in libraries:
        command += ['--extern', library + '=' + artifacts[library]]
    subprocess.run(command + flags + ['-o', str(executable)], check=True, timeout=60)
    with (args.output_dir / (name + '.log')).open('w') as log:
        result = subprocess.run([str(executable)], stdout=log, stderr=subprocess.STDOUT, timeout=60)
    results[name] = result.returncode
    print(name, result.returncode)
(args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
sys.exit(results['regressions'])
