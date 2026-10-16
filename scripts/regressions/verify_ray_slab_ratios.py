"""Independently certify the finite-MAX ray/slab regression quotients."""

from fractions import Fraction
import json
import struct


def exact(bits, width):
    code = ">f" if width == 32 else ">d"
    return Fraction(*struct.unpack(code, bits.to_bytes(width // 8, "big"))[0].as_integer_ratio())


CASES = [
    (32, 0x7EEE7650, 0x3EEE7651, 0x7F7FFFFF),
    (64, 0x7FDF14C616BC880A, 0x3FDF14C616BC880B, 0x7FEFFFFFFFFFFFFF),
]
results = []
for width, delta, component, maximum in CASES:
    ratio = exact(delta, width) / exact(component, width)
    largest = exact(maximum, width)
    previous = exact(maximum - 1, width)
    assert (largest + previous) / 2 < ratio <= largest
    results.append({"width": width, "delta_bits": hex(delta),
                    "component_bits": hex(component),
                    "correct_round_to_nearest_result": "finite MAX"})
print(json.dumps(results, indent=2))
