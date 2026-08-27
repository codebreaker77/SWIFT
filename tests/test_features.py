"""
Unit tests for feature extraction modules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import unittest
from src.config import AudioConfig, LFCCConfig
from src.features.lfcc import LFCCExtractor, extract_lfcc_batch

class TestLFCCExtractor(unittest.TestCase):
    def setUp(self):
        self.audio_cfg = AudioConfig()
        self.lfcc_cfg = LFCCConfig()
        self.extractor = LFCCExtractor(self.audio_cfg, self.lfcc_cfg)

    def test_single_audio_extraction_shape(self):
        # 2 seconds of 16kHz sine wave audio (32,000 samples)
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        lfcc_tensor = self.extractor(dummy_audio)
        
        # Output shape must be (1, 224, 224)
        self.assertEqual(lfcc_tensor.shape, (1, 224, 224))
        self.assertTrue(isinstance(lfcc_tensor, torch.Tensor))

    def test_value_range_normalization(self):
        dummy_audio = np.random.randn(32000).astype(np.float32)
        lfcc_tensor = self.extractor(dummy_audio)
        
        # Normalized pixel values must be within [0.0, 1.0]
        self.assertGreaterEqual(lfcc_tensor.min().item(), 0.0)
        self.assertLessEqual(lfcc_tensor.max().item(), 1.0)

    def test_batch_extraction(self):
        # Batch of 4 audio waveforms
        dummy_batch = torch.randn(4, 32000)
        batch_lfcc = extract_lfcc_batch(dummy_batch, self.extractor)
        
        self.assertEqual(batch_lfcc.shape, (4, 1, 224, 224))

if __name__ == '__main__':
    unittest.main()
