"""Independent full-coordinate Bareiss oracle for finite stored-bit predicates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import struct

def exact_integer(bits):
    value = struct.unpack('>d', bits.to_bytes(8, 'big'))[0]
    numerator, denominator = value.as_integer_ratio()
    return numerator << (1074 - (denominator.bit_length() - 1))

def determinant(matrix):
    # Fraction-free elimination of the full homogeneous matrix, independent
    # from the Rust translated-matrix Leibniz enumeration.
    data = [row[:] for row in matrix]
    previous = 1
    parity = 1
    for column in range(len(data) - 1):
        pivot_row = next((row for row in range(column, len(data)) if data[row][column]), None)
        if pivot_row is None:
            return 0
        if pivot_row != column:
            data[column], data[pivot_row] = data[pivot_row], data[column]
            parity = -parity
        pivot = data[column][column]
        for row in range(column + 1, len(data)):
            for other in range(column + 1, len(data)):
                numerator = data[row][other] * pivot - data[row][column] * data[column][other]
                assert numerator % previous == 0
                data[row][other] = numerator // previous
            data[row][column] = 0
        previous = pivot
    return parity * data[-1][-1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cases-per-operation', type=int, default=1024)
args = parser.parse_args()
rng = random.Random(0x26D1)
counts = Counter()
endpoint_exponents = [0, 1, 2, 879, 880, 881, 882, 1223, 1224, 1225, 2045, 2046]
with args.output.open('w') as output:
    for operation, count, dimension in [('orient2d', 3, 2), ('orient3d', 4, 3), ('incircle', 4, 2), ('insphere', 5, 3)]:
        for index in range(args.cases_per_operation):
            common = rng.randrange(2047)
            points = []
            for _ in range(count):
                point = []
                for _ in range(dimension):
                    mode = index % 4
                    exponent = rng.choice(endpoint_exponents) if mode == 0 else common if mode == 1 else rng.randrange(2047)
                    fraction = rng.getrandbits(52) if mode != 3 else rng.choice([0, 1, (1 << 52) - 1])
                    bits = (rng.getrandbits(1) << 63) | (exponent << 52) | fraction
                    if mode == 3 and rng.randrange(3) == 0:
                        bits &= 1 << 63  # Both signed zeros are exact inputs.
                    point.append(bits)
                points.append(point)
            if index % 17 == 0:
                points[1] = points[0][:]
            if index % 29 == 0:
                points[-1] = points[0][:]
            integers = [[exact_integer(bits) for bits in point] for point in points]
            matrix = [point + ([sum(value * value for value in point)] if operation in ['incircle', 'insphere'] else []) + [1] for point in integers]
            value = determinant(matrix)
            expected = (value > 0) - (value < 0)
            if operation == 'orient3d':
                expected = -expected  # Geometry's right-hand convention.
            counts[(operation, expected)] += 1
            row = {'kind': operation, 'points': [[f'{bits:016x}' for bits in point] for point in points], 'expected': expected}
            output.write(json.dumps(row, separators=(',', ':')) + '\n')
summary = {'seed': '0x26D1', 'method': 'exact integer ratios from Python float.as_integer_ratio; full homogeneous determinants by fraction-free Bareiss elimination', 'cases_per_operation': args.cases_per_operation, 'total': sum(counts.values()), 'counts': {f'{operation}:{sign}': count for (operation, sign), count in counts.items()}, 'corpus_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
args.output.with_suffix('.receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
