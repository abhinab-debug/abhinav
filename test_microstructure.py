import unittest
import numpy as np
import pandas as pd
from physics import reconstruct_sides, calculate_wnp, calculate_gini
from processing import sanitize_floorsheet

class TestMicrostructure(unittest.TestCase):
    def test_reconstruct_sides(self):
        rates = np.array([100, 101, 101, 99, 99, 100])
        # 101-100 = 1 -> 1
        # 101-101 = 0 -> prev (1)
        # 99-101 = -2 -> -1
        # 99-99 = 0 -> prev (-1)
        # 100-99 = 1 -> 1
        expected = np.array([1, 1, 1, -1, -1, 1])
        np.testing.assert_array_equal(reconstruct_sides(rates), expected)

    def test_calculate_wnp(self):
        quantities = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]) # 11 elements
        # 95th percentile of 11 elements will pick 1000
        sides = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        wnp = calculate_wnp(quantities, sides)
        self.assertEqual(wnp, 1000.0)

    def test_calculate_gini(self):
        vols = np.array([100, 100, 100])
        gini = calculate_gini(vols)
        self.assertAlmostEqual(gini, 0.0)

class TestProcessing(unittest.TestCase):
    def test_sanitize_floorsheet(self):
        df = pd.DataFrame({
            'Symbol': ['NABIL', 'NABIL-PO', 'MF1-MF', 'D85'],
            'Date': ['2026-05-22', '2026-05-22', '2026-05-22', '2026-05-22']
        })
        sanitized = sanitize_floorsheet(df)
        self.assertEqual(len(sanitized), 1)
        self.assertEqual(sanitized['Symbol'].iloc[0], 'NABIL')

    def test_jan3_purge(self):
        df = pd.DataFrame({
            'Symbol': ['NABIL', 'NABIL'],
            'Date': ['2026-01-02', '2026-01-03']
        })
        sanitized = sanitize_floorsheet(df)
        self.assertEqual(len(sanitized), 1)
        self.assertEqual(sanitized['Date'].iloc[0], '2026-01-02')

if __name__ == '__main__':
    unittest.main()
