import time
import unittest
from typing import Callable, Any

from program01 import binet, strassen
import numpy as np


class Counter:
    def __init__(self):
        self.additions = 0
        self.multiplications = 0

    def callback(self, mults: int, adds: int):
        self.additions += adds
        self.multiplications += mults

    def reset(self) -> [int, int]:
        a, m = self.additions, self.multiplications
        self.additions, self.multiplications = 0, 0
        return a, m


class Program01Test(unittest.TestCase):
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
         [13, 14, 15, 16]],

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
         [13, 14, 15, 16]],

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

    c = [
        [[4, 1],
         [2, 2]],

        [[0, 0, 0, 1],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 0, 0]],

        [[90, 100, 110, 120],
         [202, 228, 254, 280],
         [314, 356, 398, 440],
         [426, 484, 542, 600]],

        [[1540, 1237, 6042, 5706],
         [220, 28313, 2225, 10570],
         [51, 3233, 15502, 267],
         [1053, 34578, 78391, 5976]],

        [[592, 756, 548, 612, 585, 561, 645, 643],
         [657, 800, 538, 602, 656, 564, 814, 709],
         [420, 553, 450, 466, 468, 434, 566, 490],
         [531, 677, 460, 506, 587, 525, 632, 554],
         [683, 781, 568, 590, 709, 632, 838, 714],
         [362, 385, 309, 358, 357, 245, 438, 430],
         [619, 753, 593, 660, 632, 507, 727, 648],
         [483, 610, 402, 410, 557, 506, 639, 446]]
    ]

    def setUp(self) -> None:
        np.random.seed(0)
        self.counter = Counter()

    @staticmethod
    def get_rnd_matrices(n: int):
        return [np.random.randint(5, size=(2 ** n, 2 ** n)).tolist() for _ in range(2)]

    def test_binet(self):
        [self.assertEqual(_c, binet(_a, _b, self.counter.callback))
         for (_a, _b, _c) in zip(self.a, self.b, self.c)]

    def test_strassen(self):
        [self.assertEqual(_c, strassen(_a, _b, self.counter.callback))
         for (_a, _b, _c) in zip(self.a, self.b, self.c)]

    def test_binet_bench(self):
        self.benchmark(binet)

    def test_strassen_bench(self):
        self.benchmark(strassen)

    def benchmark(self, func: Callable[[Any, Any, Callable], Any]):
        score = {}
        calls = {}
        self.counter.reset()
        for i in range(1, 16):
            a, b = Program01Test.get_rnd_matrices(i)
            tic = time.perf_counter()
            c = func(a, b, self.counter.callback)

            score[i] = time.perf_counter() - tic
            calls[i] = self.counter.reset()

            print(score)
            print(calls)
            self.assertEqual(len(c), 2 ** i)


if __name__ == '__main__':
    unittest.main()
