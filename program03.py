from typing import List, Callable

import numpy as np

from program02 import inverse, sub

Matrix = List[List[float]]


def determinant(U):
    det = 1
    for i in range(len(U)):
        for j in range(len(U)):
            if i == j:
                det = det * U[i][j]
    print("Determinant: {}".format(det))


def eigenvalues(U):
    eigenvalues = [U[i][i] for i in range(len(U))]
    print("Eigenvalues:")
    for i in eigenvalues:
        print(i, end=", ")


def zeros(_size) -> Matrix:
    return [[0. for _ in range(_size)] for _ in range(_size)]


def lu_decomposition(a: Matrix, _c: Callable[[int, int], None] = None) -> (Matrix, Matrix):
    from test_program02 import A
    mult = A.mult

    if len(a[0]) == 2:
        l11, l22 = 1, 1
        u11 = a[0][0]
        u12 = a[0][1]
        l21 = a[1][0] / a[0][0]
        u22 = (a[0][1] - (l21 * u12)) / l22
        _c(2, 1)
        return [[l11, 0], [l21, l22]], \
               [[u11, u12], [0, u22]]

    # divide input matrix into blocks
    _size = int(len(a) / 2)
    a11 = [i[:_size] for i in a[:_size]]
    a12 = [i[_size:len(a)] for i in a[:_size]]
    a21 = [i[:_size] for i in a[_size:len(a)]]
    a22 = [i[_size:len(a)] for i in a[_size:len(a)]]

    l11, u11 = lu_decomposition(a11, _c)
    ui11 = inverse(u11, _c)
    l21 = mult(a21, ui11, _c)
    li11 = inverse(l11, _c)
    u12 = mult(li11, a12, _c)
    l22 = sub(a22, mult(mult(mult(a21, ui11, _c), li11, _c), a12, _c), _c)
    l22, u22 = lu_decomposition(l22, _c)

    L = [*[[*_l1, *_l2] for _l1, _l2 in zip(l11, zeros(len(l11)))],
         *[[*_l1, *_l2] for _l1, _l2 in zip(l21, l22)]]

    U = [*[[*_u1, *_u2] for _u1, _u2 in zip(u11, u12)],
         *[[*_u1, *_u2] for _u1, _u2 in zip(zeros(len(u22)), u22)]]
    return L, U
