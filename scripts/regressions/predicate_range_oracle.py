"""Independent Fraction determinant checks for the analytical range fixtures."""
from fractions import Fraction
from itertools import permutations
import json
from pathlib import Path

def determinant(matrix):
    size = len(matrix)
    result = Fraction(0)
    for permutation in permutations(range(size)):
        parity = sum(permutation[i] > permutation[j] for i in range(size) for j in range(i+1,size))
        term = Fraction(-1 if parity % 2 else 1)
        for row, column in enumerate(permutation):
            term *= matrix[row][column]
        result += term
    return result

def power(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1,2 ** -exponent)

rows = []
for exponent in [0,600,-600,400,-400,300,-300,240,-240,-149,126]:
    s = power(exponent)
    matrices = {
        'orient2d': [[0,0,1],[s,0,1],[0,s,1]],
        'orient3d.right_hand': [[s,0,0],[0,s,0],[0,0,s]],
        'incircle': [[s,0,s*s,1],[0,s,s*s,1],[-s,0,s*s,1],[0,0,0,1]],
        'insphere.shewchuk': [[s,0,0,s*s,1],[0,s,0,s*s,1],[0,0,s,s*s,1],[-s,0,0,s*s,1],[0,0,0,0,1]],
    }
    expected = [s**2,s**3,2*s**4,2*s**5]
    for (operation,matrix), analytical in zip(matrices.items(),expected):
        actual = determinant(matrix)
        assert actual == analytical > 0
        rows.append({'operation':operation,'scale_exponent':exponent,'numerator_bits':actual.numerator.bit_length(),'denominator_bits':actual.denominator.bit_length(),'sign':1,'exact_formula_verified':True})
h=power(600)
points=[[0,0,1],[1,0,1],[h,h,1]]
for permutation in permutations(range(3)):
    parity=sum(permutation[i]>permutation[j] for i in range(3) for j in range(i+1,3))
    value=determinant([points[i] for i in permutation])
    assert value == (-1 if parity%2 else 1)*h
    rows.append({'operation':'orient2d.finite_determinant','permutation':list(permutation),'sign':-1 if parity%2 else 1,'exact_magnitude':'2^600','exact_formula_verified':True})
path=Path(__file__).with_name('predicate_range_oracle.json')
path.write_text(json.dumps({'method':'independent Leibniz determinants over Python Fraction; no floating point or provider calls','cases':rows},indent=2)+'\n')
print(len(rows),'exact determinants verified')
