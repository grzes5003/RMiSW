import time
import unittest
from scipy import linalg
from typing import Any, Callable
from unittest import TestCase

import numpy as np
from numpy.linalg import LinAlgError

from program01 import binet, strassen
from program02 import add, sub, inverse, Matrix
from program03 import lu_decomposition
from test_program01 import Counter, Program01Test
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
         [41, 42, 323, 4]]
    ]

    def setUp(self) -> None:
        np.random.seed(0)
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

    def test_strassen_bench(self):
        A.mult = strassen
        self.benchmark()

    def benchmark(self):
        score, calls = {}, {}
        self.counter.reset()
        for i in range(1, 16):
            while True:
                try:
                    a = Program02Test.get_rnd_matrix(i)
                    tic = time.perf_counter()
                    c = lu_decomposition(a, self.counter.callback)

                    score[i] = time.perf_counter() - tic
                    calls[i] = self.counter.reset()
                except (LinAlgError, ZeroDivisionError):
                    continue
                break
            print(score)
            print(calls)
            self.assertEqual(len(c), 2 ** i)


if __name__ == '__main__':
    unittest.main()