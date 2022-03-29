from typing import Callable, List

from program01 import mult

Matrix = List[List[float]]


def sub(a: Matrix, b: Matrix, _callback: Callable[[int, int], None] = None) -> Matrix:
    return add(a, neg(b), _callback)


def add(a: Matrix, b: Matrix, _callback: Callable[[int, int], None] = None) -> Matrix:
    _callback(0, len(a) ** 2)
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def neg(a: Matrix) -> Matrix:
    return [[-x for x in ra] for ra in a]


def inverse(a: Matrix, _c: Callable[[int, int], None] = None) -> Matrix:

    if len(a[0]) == 2:
        idet = 1 / (a[0][0] * a[1][1] - a[0][1] * a[1][0])
        return [[a[1][1] * idet, - a[0][1] * idet],
                [- a[1][0] * idet, a[0][0] * idet]]

        # divide input matrix into blocks
    _size = int(len(a) / 2)
    a11 = [i[:_size] for i in a[:_size]]
    a12 = [i[_size:len(a)] for i in a[:_size]]
    a21 = [i[:_size] for i in a[_size:len(a)]]
    a22 = [i[_size:len(a)] for i in a[_size:len(a)]]

    ai11 = inverse(a11, _c)
    s22 = sub(a22, mult(a21, mult(ai11, a12, _c), _c), _c)
    si22 = inverse(s22, _c)

    i = [[1 if y == x else 0 for y in range(_size)] for x in range(_size)]
    b11 = mult(ai11, add(i, mult(mult(mult(a12, si22, _c), a21, _c), ai11, _c), _c), _c)
    b12 = neg(mult(mult(ai11, a12, _c), si22, _c))
    b21 = neg(mult(mult(si22, a21, _c), ai11, _c))
    b22 = si22

    # unwrap blocks and return matrix
    return [*[[*_c1, *_c2] for _c1, _c2 in zip(b11, b12)],
            *[[*_c1, *_c2] for _c1, _c2 in zip(b21, b22)]]
