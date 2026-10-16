"""Compare actual stored-direction slab classification with a Fraction oracle.

Run separately compiled public-API drivers from tests/support on the same
finite IEEE corpus. Certify the equal-endpoint and positive-infinity-hit
postfilter and detect new errors elsewhere. Report any residual discrepancies.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
parser.add_argument('--head', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def encode(value, width):
    return int.from_bytes(struct.pack('>f' if width == 32 else '>d', value), 'big')


def decode(bits, width):
    return struct.unpack('>f' if width == 32 else '>d', bits.to_bytes(width // 8, 'big'))[0]


fixtures = []


def add(width, label, origin, direction, lower, upper):
    fixtures.append((width, label, [encode(x, width) for x in origin + direction + lower + upper]))


for width, fraction_bits, minimum, maximum, exponents in [
    (32, 23, 0x00800000, 0x7f7fffff, [-126, -100, -1, 0, 1, 64, 100, 125, 126, 127]),
    (64, 52, 0x0010000000000000, 0x7fefffffffffffff,
     [-1022, -600, -142, -1, 0, 1, 200, 600, 1021, 1022, 1023]),
]:
    anchors = {4, 5, minimum - 1, minimum, minimum + 1, maximum - 1, maximum}
    for exponent in exponents:
        pivot = encode(math.ldexp(1.0, exponent), width)
        anchors.update([max(4, pivot - 1), pivot, min(maximum, pivot + 1)])
    for pivot in sorted(anchors):
        values = [decode(pivot - i, width) for i in range(4)]
        top, previous, below, lowest = values
        patterns = [(top, top, previous, previous),
                    (lowest, below, previous, top),
                    (top, top, top, top),
                    (lowest, top, below, previous),
                    (lowest, previous, previous, top)]
        for pattern, (xmin, xmax, ymin, ymax) in enumerate(patterns):
            for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                for rotation in range(3):
                    rotate = lambda a: a[rotation:] + a[:rotation]
                    lower = [min(sx*xmin, sx*xmax), min(sy*ymin, sy*ymax), -1.0]
                    upper = [max(sx*xmin, sx*xmax), max(sy*ymin, sy*ymax), 1.0]
                    add(width, f'boundary-{pivot:x}-{pattern}-{sx}-{sy}-{rotation}',
                        [0.0]*3, rotate([float(sx), float(sy), 0.0]), rotate(lower), rotate(upper))
    large = decode(maximum, width)
    previous = decode(maximum - 1, width)
    for origin in [[large, large, 0.0], [large, previous, 0.0]]:
        for target in [[-large, -large, 0.0], [-large, -previous, 0.0]]:
            add(width, 'original-difference-overflow', origin, [-1.0, -1.0, 0.0], target, target)
    for direction in [[1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 1.0, 1.0]]:
        for target in [[large/2, large, 0.0], [large, large, large], [large, previous, large]]:
            add(width, 'unequal-or-three-active-directions', [0.0]*3, direction, target, target)
    rng = random.Random(3000 + width)

    def random_value():
        bits = rng.getrandbits(width)
        magnitude = bits & ((1 << (width-1))-1)
        if magnitude > maximum:
            bits = (bits & (1 << (width-1))) | maximum
        return decode(bits, width)

    for index in range(4096):
        origin = [random_value() for _ in range(3)]
        direction = [random_value() for _ in range(3)]
        if not any(direction):
            direction[0] = 1.0
        corners = [[random_value() for _ in range(3)] for _ in range(2)]
        lower = [min(x, y) for x, y in zip(*corners)]
        upper = [max(x, y) for x, y in zip(*corners)]
        add(width, f'full-bits-{index}', origin, direction, lower, upper)
    for index in range(512):
        direction = [1.0+rng.random() for _ in range(3)]
        lower = [decode(maximum-rng.randrange(64), width) for _ in range(3)]
        upper = [decode(min(maximum, encode(v, width)+rng.randrange(8)), width) for v in lower]
        add(width, f'collapsed-candidate-{index}', [0.0]*3, direction, lower, upper)

text = ''.join(str(width)+' '+ ' '.join(f'{v:x}' for v in values)+'\n' for width, _, values in fixtures)
(out/'input.tsv').write_text(text)
outputs = {}
for mode, executable in [('base', args.base), ('head', args.head)]:
    run = subprocess.run([str(Path(executable).resolve())], input=text, capture_output=True, text=True, check=True)
    (out/(mode+'.tsv')).write_text(run.stdout)
    outputs[mode] = [[int(v) for v in line.split()] for line in run.stdout.splitlines()]
    assert len(outputs[mode]) == len(fixtures)


def expected(values, directions, width):
    values = [Fraction(decode(v, width)) for v in values]
    directions = [Fraction(decode(v, width)) for v in directions]
    entry, exit = Fraction(0), None
    for origin, direction, lower, upper in zip(values[:3], directions, values[6:9], values[9:12]):
        if lower > upper:
            return False
        if not direction:
            if not lower <= origin <= upper:
                return False
            continue
        left, right = sorted(((lower-origin)/direction, (upper-origin)/direction))
        entry = max(entry, left)
        exit = right if exit is None else min(exit, right)
        if entry > exit:
            return False
    return exit is None or entry <= exit


counts = {str(width): {'cases': 0, 'collapsed_candidates': 0, 'corrected_misses': 0,
                       'retained_true_collapsed_hits': 0, 'preexisting_other_mismatches': 0} for width in [32, 64]}
residual = []
for index, ((width, label, values), base, head) in enumerate(zip(fixtures, outputs['base'], outputs['head'])):
    assert base[0] == head[0] == index
    assert base[1:4] == head[1:4], 'stored normalization changes'
    truth = expected(values, head[1:4], width)
    row = counts[str(width)]
    row['cases'] += 1
    collapsed = bool(base[4]) and (base[5] == base[6]
                                 or base[5] == encode(float('inf'), width))
    if collapsed:
        row['collapsed_candidates'] += 1
        assert bool(head[4]) == truth, (index, label, truth, base, head)
        if truth:
            row['retained_true_collapsed_hits'] += 1
            assert head[4:] == base[4:]
        else:
            row['corrected_misses'] += 1
    else:
        assert head[4:] == base[4:], 'changes outside bounded postfilter'
    assert not (bool(base[4]) == truth and bool(head[4]) != truth), 'new classification error'
    if bool(head[4]) != truth:
        assert not collapsed
        row['preexisting_other_mismatches'] += 1
        residual.append({'index': index, 'width': width, 'label': label, 'input_bits': values,
                         'stored_direction_bits': head[1:4], 'expected_hit': truth,
                         'base': base, 'head': head})
for row in counts.values():
    assert row['corrected_misses'] > 0 and row['retained_true_collapsed_hits'] > 0
(out/'residual-native-rounding.json').write_text(json.dumps(residual, indent=2)+'\n')
receipt = {'status': 'BOUNDED_POSTFILTER_PASS; NO_NEW_CLASSIFICATION_ERROR',
           'corpus_sha256': hashlib.sha256(text.encode()).hexdigest(), 'counts': counts,
           'oracle': 'Python Fraction original stored-coordinate subtraction/division and interval clipping; normalized direction read from each actual public Ray.',
           'scope': 'Every equal-endpoint or positive-infinity-entry candidate matches exact classification; true-hit returned intervals retained. Other native behavior unchanged. Any residual errors are reported, not certified as correct.'}
(out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
