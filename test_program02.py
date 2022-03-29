import time
import unittest
from typing import Any, Callable
from unittest import TestCase

import numpy as np
from numpy.linalg import LinAlgError

from program01 import binet, strassen
from program02 import add, sub, inverse, Matrix
from test_program01 import Counter, Program01Test


class A:
    @staticmethod
    def mult(*args, **kwargs) -> Matrix:
        raise NotImplementedError


class Program02Test(TestCase):
    a = [
        [[1, 0],
         [0, 1]],

        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],

        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 17]],

        [[1, 62, 14, 51],
         [54, 5, 0, 46],
         [1, 2, 66, 2],
         [41, 42, 323, 4]],

        [[6, 7, 13, 4, 7, 13, 12, 6],
         [5, 11, 4, 13, 10, 14, 7, 11],
         [6, 2, 5, 11, 6, 5, 13, 6],
         [7, 12, 6, 1, 13, 11, 10, 4],
         [5, 15, 6, 9, 12, 8, 14, 9],
         [4, 10, 1, 5, 2, 8, 9, 2],
         [13, 11, 6, 6, 4, 12, 12, 8],
         [7, 9, 2, 8, 13, 4, 5, 10]]
    ]

    b = [
        [[4, 1],
         [2, 2]],

        [[0, 0, 0, 1],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 0, 0]],

        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 17]],

        [[1, 521, 34, 123],
         [24, -1, 41, 22],
         [0, 41, 233, -1],
         [1, 4, 4, 83]],

        [[5, 11, 9, 14, 14, 5, 8, 2],
         [14, 6, 2, 6, 14, 4, 13, 12],
         [12, 14, 6, 12, 14, 13, 6, 10],
         [8, 8, 3, 10, 14, 4, 14, 13],
         [4, 12, 3, 2, 12, 15, 10, 3],
         [8, 14, 9, 12, 3, 3, 8, 14],
         [6, 7, 14, 8, 3, 8, 10, 11],
         [12, 14, 14, 4, 1, 13, 13, 5]]
    ]

    def setUp(self) -> None:
        np.random.seed(0)
        self.counter = Counter()

    @staticmethod
    def get_rnd_matrix(n: int) -> Matrix:
        return np.random.randint(5, size=(2 ** n, 2 ** n)).tolist()

    def test_add(self):
        [self.assertEqual(np.add(np.array(_a), np.array(_b)).tolist(), add(_a, _b, self.counter.callback))
         for (_a, _b) in zip(self.a, self.b)]

    def test_sub(self):
        [self.assertEqual(np.subtract(np.array(_a), np.array(_b)).tolist(), sub(_a, _b, self.counter.callback))
         for (_a, _b) in zip(self.a, self.b)]

    def test_inverse(self):
        for _ in range(10):
            self.counter.reset()
            while True:
                try:
                    a = Program02Test.get_rnd_matrix(2)
                    ai = np.linalg.inv(a).tolist()
                    np.testing.assert_almost_equal(ai, inverse(a, self.counter.callback))
                except (LinAlgError, ZeroDivisionError):
                    continue
                break

    def test_idt(self):
        print([np.linalg.det(_a) for _a in self.a])

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
                    c = inverse(a, self.counter.callback)

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
