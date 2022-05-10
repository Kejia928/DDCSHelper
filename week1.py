import unittest
import DDCS as DDCS


class MyTestCase(unittest.TestCase):
    def test_stats(self):
        mean, median, std, variance = DDCS.statistic([-3, 2, 4, 6, -2, 0, 5])
        self.assertAlmostEqual(mean, 1.71, places=2)
        self.assertAlmostEqual(median, 2.0, places=2)
        self.assertAlmostEqual(std, 3.5, places=2)
        self.assertAlmostEqual(variance, 12.24, places=2)

    def test_manhattan_distance(self):
        self.assertEqual(DDCS.manhattan_distance([4, 5, 6], [2, -1, 3]), 11)

    def test_euclidean_distance(self):
        self.assertEqual(DDCS.euclidean_distance([4, 5, 6], [2, -1, 3]), 7.0)

    def test_n_norm(self):
        self.assertAlmostEqual(DDCS.p_norm_distance([4, 5, 6], [2, -1, 3], 3), 6.3, places=1)

    def test_chebyshev_distance(self):
        self.assertEqual(DDCS.chebyshev_distance([4, 5, 6], [2, -1, 3]), 6)

    def test_edit_distance(self):
        self.assertEqual(DDCS.edit_distance("water", "further"), 4)

    def test_hamming_distance(self):
        self.assertEqual(DDCS.hamming_distance("weather", "further"), 3)


if __name__ == '__main__':
    unittest.main()
