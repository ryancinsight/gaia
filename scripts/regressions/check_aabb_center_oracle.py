"""Check finite overflow midpoints exactly and preserve other native center bits."""
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


def decode(bits, width):
    return struct.unpack('>f' if width == 32 else '>d', bits.to_bytes(width // 8, 'big'))[0]


def encode(value, width):
    return int.from_bytes(struct.pack('>f' if width == 32 else '>d', value), 'big')


def power(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def rounded_bits(value, width):
    """Round an exact rational once, nearest/ties-to-even, without float casts."""
    fraction_bits, bias = (23, 127) if width == 32 else (52, 1023)
    sign = (1 << (width - 1)) if value < 0 else 0
    value = abs(value)
    if not value:
        return sign
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power(exponent):
        exponent -= 1
    exponent = max(exponent, 1 - bias)
    scaled = value / power(exponent - fraction_bits)
    coefficient, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (
            2 * remainder == scaled.denominator and coefficient % 2):
        coefficient += 1
    if coefficient == 1 << (fraction_bits + 1):
        coefficient >>= 1
        exponent += 1
    if coefficient < 1 << fraction_bits:
        return sign | coefficient
    exponent_field = exponent + bias
    assert exponent_field < 2 * bias + 1, 'finite midpoint must be representable'
    return sign | (exponent_field << fraction_bits) | (coefficient - (1 << fraction_bits))


fixtures = []


def add(width, label, lower, upper):
    fixtures.append((width, label, [encode(x, width) for x in lower + upper]))


for width, fraction_bits, minimum, maximum, exponents in [
    (32, 23, 0x00800000, 0x7f7fffff, [-126, -100, -1, 0, 1, 64, 103, 125, 126, 127]),
    (64, 52, 0x0010000000000000, 0x7fefffffffffffff,
     [-1022, -600, -1, 0, 1, 600, 970, 1021, 1022, 1023]),
]:
    anchors = {0, 1, 2, 3, minimum - 1, minimum, minimum + 1, maximum - 1, maximum}
    for exponent in exponents:
        pivot = encode(math.ldexp(1.0, exponent), width)
        anchors.update([pivot - 1, pivot, min(maximum, pivot + 1)])
    signed = sorted({decode(bits | sign, width) for bits in anchors
                     for sign in [0, 1 << (width - 1)]})
    large, tiny = decode(maximum, width), decode(1, width)
    for a in signed:
        for b in signed:
            lo, hi = min(a, b), max(a, b)
            add(width, 'boundary', [lo, -large, tiny], [hi, large, tiny])
    rng = random.Random(3200 + width)

    def random_value():
        bits = rng.getrandbits(width)
        magnitude = min(bits & ((1 << (width - 1)) - 1), maximum)
        return decode((bits & (1 << (width - 1))) | magnitude, width)

    for index in range(8192):
        corners = [[random_value() for _ in range(3)] for _ in range(2)]
        lower = [min(a, b) for a, b in zip(*corners)]
        upper = [max(a, b) for a, b in zip(*corners)]
        add(width, f'full-bits-{index}', lower, upper)
    for index in range(2048):
        values = [decode(maximum - rng.randrange(1 << (fraction_bits + 1)), width)
                  for _ in range(2)]
        lo, hi = sorted(values)
        add(width, f'large-same-sign-{index}', [lo, -hi, tiny], [hi, -lo, tiny])
        # The integer quantizer must retain every stored finite nonzero value.
        for value in values:
            assert rounded_bits(Fraction(value), width) == encode(value, width)
    # Legacy inverted boxes remain outside the finite ordered recovery contract,
    # including an inverted ordinary axis next to an overflowing valid axis.
    for inverted_axis in range(3):
        for overflow_axis in range(3):
            if overflow_axis == inverted_axis:
                continue
            for sign in [-1.0, 1.0]:
                for inverted_overflow in [False, True]:
                    lower, upper = [tiny] * 3, [tiny] * 3
                    lower[overflow_axis] = upper[overflow_axis] = sign * large
                    if inverted_overflow:
                        lower[inverted_axis] = large if sign > 0 else -large * 0.5
                        upper[inverted_axis] = large * 0.5 if sign > 0 else -large
                    else:
                        lower[inverted_axis], upper[inverted_axis] = 1.0, -1.0
                    add(width, 'inverted-observation', lower, upper)
    for a in [0.0, -0.0]:
        for b in [0.0, -0.0]:
            add(width, 'signed-zero', [a] * 3, [b] * 3)
    infinity = encode(float('inf'), width)
    specials = [infinity, infinity | (1 << (width - 1)), infinity | 1,
                infinity | (1 << (fraction_bits - 1)) | 0x123,
                infinity | (1 << (width - 1)) | 1,
                infinity | (1 << (width - 1)) | (1 << (fraction_bits - 1)) | 0x123]
    for bits in specials:
        for upper_bound in [False, True]:
            for axis in range(3):
                for mixed_overflow in [False, True]:
                    lower = [large if mixed_overflow else -1.0] * 3
                    upper = [large if mixed_overflow else 1.0] * 3
                    (upper if upper_bound else lower)[axis] = decode(bits, width)
                    add(width, 'nonfinite-observation', lower, upper)

text = ''.join(str(width) + ' ' + ' '.join(f'{v:x}' for v in values) + '\n'
               for width, _, values in fixtures)
(out / 'input.tsv').write_text(text)
outputs = {}
for label, executable in [('base', args.base), ('head', args.head)]:
    run = subprocess.run([str(Path(executable).resolve())], input=text,
                         capture_output=True, text=True, check=True)
    (out / (label + '.tsv')).write_text(run.stdout)
    outputs[label] = [[int(v) for v in line.split()] for line in run.stdout.splitlines()]
    assert len(outputs[label]) == len(fixtures)

counts = {str(width): {'boxes': 0, 'finite_boxes': 0, 'repaired_coordinates': 0,
                      'unchanged_coordinates': 0, 'nonfinite_boxes_preserved': 0, 'inverted_boxes_preserved': 0}
          for width in [32, 64]}
for index, ((width, label, values), base, head) in enumerate(
        zip(fixtures, outputs['base'], outputs['head'])):
    assert base[0] == head[0] == index
    coordinates = [decode(bits, width) for bits in values]
    finite_box = all(math.isfinite(x) for x in coordinates)
    row = counts[str(width)]
    row['boxes'] += 1
    row['finite_boxes'] += finite_box
    if not finite_box:
        assert base == head, (index, label, 'legacy nonfinite input changed')
        row['nonfinite_boxes_preserved'] += 1
        continue
    if any(lo > hi for lo, hi in zip(coordinates[:3], coordinates[3:])):
        assert base == head, (index, label, 'legacy inverted input changed')
        row['inverted_boxes_preserved'] += 1
        continue
    for axis, (old, new) in enumerate(zip(base[1:], head[1:])):
        lo, hi = coordinates[axis], coordinates[axis + 3]
        assert lo <= hi
        if math.isinf(decode(old, width)):
            expected = rounded_bits((Fraction(lo) + Fraction(hi)) / 2, width)
            assert new == expected, (index, label, axis, old, new, expected)
            assert lo <= decode(new, width) <= hi
            row['repaired_coordinates'] += 1
        else:
            assert new == old, (index, label, axis, 'ordinary arithmetic changed')
            row['unchanged_coordinates'] += 1
for row in counts.values():
    assert row['repaired_coordinates'] > 0 and row['nonfinite_boxes_preserved'] > 0
receipt = {'status': 'FINITE_OVERFLOW_MIDPOINT_PASS; OTHER_NATIVE_BITS_PRESERVED',
           'counts': counts, 'corpus_sha256': hashlib.sha256(text.encode()).hexdigest(),
           'oracle': 'Fraction original-coordinate mean, explicit integer nearest/ties-even IEEE encoding; no intermediate float rounding.',
           'scope': 'Finite ordered-box overflow coordinates match exact rounded means. All other native center bits, including nonfinite and inverted boxes, remain unchanged. No universal correctly-rounded midpoint or new invalid-box policy claim.'}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
