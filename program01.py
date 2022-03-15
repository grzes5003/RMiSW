from typing import Callable


Matrix = list[list[int]]


def binet(a: Matrix, b: Matrix, _callback: Callable[[int, int], None] = None):
    """
    Implementation of recursive binet matrix multiplication algorithm.
    Multiplies any matrix 2^I x 2^I; I > 0
    :param a: A matrix
    :param b: B matrix
    :param _callback: function for counting (multiplications, additions)
    :return: result of A*B multiplication
    """
    if len(a[0]) == 2:
        # calculate trivial multiplication solution
        c11 = a[0][0] * b[0][0] + a[0][1] * b[1][0]
        c12 = a[0][0] * b[0][1] + a[0][1] * b[1][1]
        c21 = a[1][0] * b[0][0] + a[1][1] * b[1][0]
        c22 = a[1][0] * b[0][1] + a[1][1] * b[1][1]
        _callback(4 * 2, 4)
        # print(f'{Int.__mul__.calls}, {Int.__add__.calls}')
        return [[c11, c12],
                [c21, c22]]

    # select blocks
    _size = int(len(a) / 2)
    a11, b11 = [[i[:_size] for i in x[:_size]] for x in [a, b]]
    a12, b12 = [[i[_size:len(a)] for i in x[:_size]] for x in [a, b]]
    a21, b21 = [[i[:_size] for i in x[_size:len(a)]] for x in [a, b]]
    a22, b22 = [[i[_size:len(a)] for i in x[_size:len(a)]] for x in [a, b]]

    # calculate each block
    c11 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a11, b11, _callback), binet(a12, b21, _callback))]
    c12 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a11, b12, _callback), binet(a12, b22, _callback))]
    c21 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a21, b11, _callback), binet(a22, b21, _callback))]
    c22 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a21, b12, _callback), binet(a22, b22, _callback))]
    _callback(0, 4 * len(c11))

    # unwrap blocks and return matrix
    return [*[[*_c1, *_c2] for _c1, _c2 in zip(c11, c12)],
            *[[*_c1, *_c2] for _c1, _c2 in zip(c21, c22)]]


import time
import numpy as np

counter = 0


def split_matrix(matrix):
    n = len(matrix)

    m = int(n / 2)
    return matrix[:m, :m], matrix[:m, m:], matrix[m:, :m], matrix[m:, m:], m


def strassen(x, y):
    if len(x) == 1:
        global counter
        counter = counter + 1
        return x * y

    a, b, c, d, m = split_matrix(x)
    e, f, g, h, m = split_matrix(y)
    z = np.zeros(shape=(2 * m, 2 * m))
    z = z.astype(int)
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    z[: m, : m] = p5 + p4 - p2 + p6
    z[: m, m:] = p1 + p2
    z[m:, : m] = p3 + p4
    z[m:, m:] = p1 + p5 - p3 - p7

    return z


if __name__ == "__main__":
    x = np.array([[1, 0, 1, 0], [1, 2, 1, 0], [1, 3, 4, 2], [1, 2, 1, 1]])
    y = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
    # x = np.array([[1, 1], [0, 2]])
    # y = np.array([[2, 1], [1, 0]])
    print('Multiplication with Strassen algorithm: ')
    start_time = time.time()
    print(strassen(x, y))
    print(str(round(((time.time() - start_time) * 1000), 3)) + " ms")
    print("Operations amount: " + str(counter))
