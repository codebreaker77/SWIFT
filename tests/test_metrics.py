"""
Unit tests for metric computation (EER).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest
from src.metrics import compute_eer, compute_eer_from_probabilities

class TestMetrics(unittest.TestCase):
    def test_eer_perfect_separation(self):
        # Perfect model: real items have p_fake ~ 0.0, fake items have p_fake ~ 1.0
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p_fake = np.array([0.01, 0.02, 0.05, 0.03, 0.95, 0.98, 0.92, 0.99])
        
        eer, threshold = compute_eer_from_probabilities(y_true, p_fake)
        self.assertAlmostEqual(eer, 0.0, places=2)

    def test_eer_overlapping_distributions(self):
        np.random.seed(42)
        real_p_fake = np.random.normal(0.4, 0.15, 100)
        fake_p_fake = np.random.normal(0.6, 0.15, 100)
        
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        p_fake = np.concatenate([real_p_fake, fake_p_fake])
        
        eer, threshold = compute_eer_from_probabilities(y_true, p_fake)
        self.assertGreater(eer, 0.0)
        self.assertLess(eer, 0.3)

if __name__ == '__main__':
    unittest.main()
