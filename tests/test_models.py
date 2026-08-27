"""
Unit tests for model architecture modules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from src.config import ModelConfig
from src.models.efficientnet_b0 import EfficientNetB0Baseline

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.model = EfficientNetB0Baseline(self.config, pretrained=False)
        self.model.eval()

    def test_forward_pass_shape(self):
        # Input batch of 2 single-channel 224x224 LFCC images
        x = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            logits = self.model(x)
            probs = self.model.predict_proba(x)
            
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(probs.shape, (2, 2))
        
        # Softmax probabilities sum to 1.0 across classes
        prob_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(2), atol=1e-5))

if __name__ == '__main__':
    unittest.main()
