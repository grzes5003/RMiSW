from typing import Callable, List

from program02 import sub, add
from util import generate_charts
import time
import numpy as np

Matrix = List[List[float]]


def binet(a: Matrix, b: Matrix, _callback: Callable[[int, int], None] = None):
    """
    Implementation of recursive binet matrix multiplication algorithm.
    Multiplies any matrix 2^I x 2^I; I > 0
    :param a: A matrix
    :param b: B matrix
    :param _callback: function for counting (multiplications, additions)
    :return: result of A*B multiplication
    """
    if len(a) == 2:
        # calculate trivial multiplication solution
        c11 = a[0][0] * b[0][0] + a[0][1] * b[1][0]
        c12 = a[0][0] * b[0][1] + a[0][1] * b[1][1]
        c21 = a[1][0] * b[0][0] + a[1][1] * b[1][0]
        c22 = a[1][0] * b[0][1] + a[1][1] * b[1][1]
        _callback(4 * 2, 4)
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


counter = 0


def split_matrix(matrix):
    n = len(matrix)
    m = int(n / 2)

    _size = int(len(matrix) / 2)
    matrix11 = [i[:_size] for i in matrix[:_size]]
    matrix12 = [i[_size:len(matrix)] for i in matrix[:_size]]
    matrix21 = [i[:_size] for i in matrix[_size:len(matrix)]]
    matrix22 = [i[_size:len(matrix)] for i in matrix[_size:len(matrix)]]
    return matrix11, matrix12, matrix21, matrix22, m


def strassen(x: Matrix, y: Matrix, _c: Callable[[int, int], None] = None):
    # x = np.array(x)
    # y = np.array(y)
    if len(x) == 1:
        global counter
        _c(1, 0)
        counter = counter + 1
        return [[x[0][0] * y[0][0]]]

    a, b, c, d, m = split_matrix(x)
    e, f, g, h, m = split_matrix(y)
    z = np.zeros(shape=(2 * m, 2 * m))
    z = z.astype(int).tolist()
    p1 = strassen(a, sub(f, h, _c), _c)
    p2 = strassen(add(a, b, _c), h, _c)
    p3 = strassen(add(c, d, _c), e, _c)
    p4 = strassen(d, sub(g, e, _c), _c)
    p5 = strassen(add(a, d, _c), add(e, h, _c), _c)
    p6 = strassen(sub(b, d, _c), add(g, h, _c), _c)
    p7 = strassen(sub(a, c, _c), add(e, f, _c), _c)

    c11 = add(sub(add(p5, p4, _c), p2, _c), p6, _c)
    c12 = add(p1, p2, _c)
    c21 = add(p3, p4, _c)
    c22 = sub(sub(add(p1, p5, _c), p3, _c), p7, _c)
    # _callback(0, 18)

    return [*[[*_c1, *_c2] for _c1, _c2 in zip(c11, c12)],
            *[[*_c1, *_c2] for _c1, _c2 in zip(c21, c22)]]


if __name__ == "__main__":
    y_time = []
    y_multi_operations = []
    y_addition_operations = []
    x_axis = []
    for i in range(1, 10, 1):
        counter = 0
        print("Matrix multiplication of " + str(pow(2, i)) + "x" + str(pow(2, i)) + " size")
        x = np.random.randint(10, size=(pow(2, i), pow(2, i)))
        y = np.random.randint(10, size=(pow(2, i), pow(2, i)))
        start_time = time.time()
        print(strassen(x, y))
        print("Time of multiplication: " + str(round((time.time() - start_time), 3)) + " s")
        print("Multiplication operations amount: " + str(counter))
        print("Addition operations amount: " + str(pow(4, i)))
        x_axis.append(pow(2, i))
        y_time.append(round(((time.time() - start_time) * 1000), 3))
        y_multi_operations.append(counter)
        y_addition_operations.append(pow(4, i))

    generate_charts(x_axis, y_time, y_multi_operations, y_addition_operations)
