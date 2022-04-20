import time
import unittest
from unittest import TestCase

import numpy as np
from numpy.linalg import LinAlgError

from program01 import binet, strassen
from program02 import Matrix
from program03 import lu_decomposition
from test_program01 import Counter
from test_program02 import A


class Program02Test(TestCase):
    a = [
        [[4, 3], [6, 3]],

        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 17]],

        [[1, 62, 14, 51],
         [54, 5, 0, 46],
         [1, 2, 66, 2],
         [41, 42, 323, 4]],

        [[2, 8, 3, 1, 2, 5, 8, 0],
         [5, 1, 3, 2, 8, 1, 6, 2],
         [7, 5, 2, 8, 2, 2, 3, 1],
         [2, 0, 6, 5, 3, 6, 3, 1],
         [9, 3, 6, 5, 0, 2, 8, 5],
         [6, 5, 5, 3, 9, 1, 9, 4],
         [7, 1, 2, 6, 6, 3, 1, 9],
         [9, 6, 2, 7, 5, 8, 6, 2]]
    ]

    @staticmethod
    def matmul(a: Matrix, b: Matrix, *args) -> Matrix:
        return np.matmul(np.array(a), np.array(b)).tolist()

    def setUp(self) -> None:
        np.random.seed(1)
        self.counter = Counter()

    @staticmethod
    def get_rnd_matrix(n: int) -> Matrix:
        return np.random.randint(5, size=(2 ** n, 2 ** n)).tolist()

    def test_LU(self):
        A.mult = binet
        for a in self.a:
            l, u = lu_decomposition(a, self.counter.callback)
            x = A.mult(l, u, self.counter.callback)
            np.testing.assert_almost_equal(np.array(x), np.array(a))

    def test_binet_bench(self):
        A.mult = binet
        self.benchmark()

    def test_compare(self):
        A.mult = binet
        self.compare()

    def test_strassen_bench(self):
        A.mult = strassen
        self.benchmark()

    def test_get_matrixes(self):
        print(Program02Test.get_rnd_matrix(8))

    def benchmark(self):
        score, calls = {}, {}
        self.counter.reset()
        for i in range(1, 16):
            while True:
                try:
                    a = Program02Test.get_rnd_matrix(i)
                    tic = time.perf_counter()
                    l, u = lu_decomposition(a, self.counter.callback)

                    score[i] = time.perf_counter() - tic
                    calls[i] = self.counter.reset()
                except (LinAlgError, ZeroDivisionError):
                    continue
                break
            print(score)
            print(calls)
            self.assertEqual(len(l), 2 ** i)

    def compare(self):
        from numpy.linalg import det, eigvals
        from numpy import matmul, diagonal

        det_res, eig_res = {}, {}
        for i in range(1, 8):
            while True:
                try:
                    a = Program02Test.get_rnd_matrix(i)
                    l, u = lu_decomposition(a, self.counter.callback)

                    det_res[i] = np.prod(diagonal(l))*np.prod(diagonal(u))
                    eig_res[i] = eigvals(matmul(l, u)).tolist()
                except (LinAlgError, ZeroDivisionError):
                    continue
                break
            print(det_res)
            print(eig_res)

    def test_check(self):
        from numpy.linalg import det, eigvals

        for i in range(1, 8):
            while True:
                try:
                    a = Program02Test.get_rnd_matrix(i)
                    l, u = lu_decomposition(a, self.counter.callback)
                    res = det(l)*det(u)
                except (LinAlgError, ZeroDivisionError):
                    continue
                break
            np.testing.assert_almost_equal(np.array(res), det(a))


if __name__ == '__main__':
    unittest.main()
