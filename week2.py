import unittest

import DDCS


class MyTestCase(unittest.TestCase):
    def test_linear_regression(self):
        xarr = [-1.0, -0.3, 0.3, 1.0]
        yarr = [10.3, 5.3, -0.2, -5.3]
        w0, w1 = DDCS.linear_regression(xarr, yarr)
        self.assertAlmostEqual(w0, 2.53, places=2)
        self.assertAlmostEqual(w1, -7.91, places=2)

    def test_regularised(self):
        xarr = [-2, -1, 0, 1, 2]
        yarr = [-6.2, -2.6, 0.5, 2.7, 5.7]
        w0, w1 = DDCS.regularised(xarr, yarr, sigma=1, lam=2)
        self.assertAlmostEqual(w0, 0.0143, places=4)
        self.assertAlmostEqual(w1, 2.425, places=3)


if __name__ == '__main__':
    unittest.main()
