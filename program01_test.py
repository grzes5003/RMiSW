import unittest
from program01 import binet, strassen


class Program01Test(unittest.TestCase):
    a = [
        [[1, 0],
         [0, 1]],
    ]

    b = [
        [[4, 1],
         [2, 2]],
    ]

    c = [
        [[4, 1],
         [2, 2]],
    ]

    def test_binet(self):
        [self.assertEqual(_c, binet(_a, _b))
         for (_a, _b, _c) in zip(self.a, self.b, self.c)]

    def test_strassen(self):
        [self.assertEqual(_c, strassen(_a, _b))
         for (_a, _b, _c) in zip(self.a, self.b, self.c)]


if __name__ == '__main__':
    unittest.main()
