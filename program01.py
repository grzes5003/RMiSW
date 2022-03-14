Matrix = list[list[float]]


def binet(a: Matrix, b: Matrix):
    """
    Implementation of recursive binet matrix multiplication algorithm.
    Multiplies any matrix 2^I x 2^I; I > 0
    :param a: A matrix
    :param b: B matrix
    :return: result of A*B multiplication
    """
    if len(a[0]) == 2:
        # calculate trivial multiplication solution
        c11 = a[0][0] * b[0][0] + a[0][1] * b[1][0]
        c12 = a[0][0] * b[0][1] + a[0][1] * b[1][1]
        c21 = a[1][0] * b[0][0] + a[1][1] * b[1][0]
        c22 = a[1][0] * b[0][1] + a[1][1] * b[1][1]
        return [[c11, c12],
                [c21, c22]]

    # select blocks
    _size = int(len(a)/2)
    a11, b11 = [[i[:_size] for i in x[:_size]] for x in [a, b]]
    a12, b12 = [[i[_size:len(a)] for i in x[:_size]] for x in [a, b]]
    a21, b21 = [[i[:_size] for i in x[_size:len(a)]] for x in [a, b]]
    a22, b22 = [[i[_size:len(a)] for i in x[_size:len(a)]] for x in [a, b]]

    # calculate each block
    c11 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a11, b11), binet(a12, b21))]
    c12 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a11, b12), binet(a12, b22))]
    c21 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a21, b11), binet(a22, b21))]
    c22 = [[x + y for x, y in zip(i, j)] for i, j in zip(binet(a21, b12), binet(a22, b22))]

    # unwrap blocks and return matrix
    return [*[[*_c1, *_c2] for _c1, _c2 in zip(c11, c12)],
            *[[*_c1, *_c2] for _c1, _c2 in zip(c21, c22)]]


def strassen(a: Matrix, b: Matrix, c: Matrix = []):
    ...
