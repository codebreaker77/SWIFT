"""
Unit tests for real-time audio streaming engine & ring buffer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest
from src.realtime_stream import AudioRingBuffer, RealtimeStreamDetector

class TestRealtimeStream(unittest.TestCase):
    def test_ring_buffer_fifo(self):
        buffer = AudioRingBuffer(capacity=10)
        
        # Append 4 samples
        buffer.append(np.array([1, 2, 3, 4], dtype=np.float32))
        window = buffer.get_window()
        self.assertTrue(np.array_equal(window, np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])))
        
        # Append 8 samples (overflow capacity 10)
        buffer.append(np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32))
        window = buffer.get_window()
        # Ring buffer must contain the last 10 samples
        self.assertTrue(np.array_equal(window, np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])))

    def test_stream_detector_latency_budget(self):
        onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnx_models", "efficientnet_b0_baseline_int8.onnx")
        if not os.path.exists(onnx_path):
            onnx_path = None
            
        detector = RealtimeStreamDetector(onnx_model_path=onnx_path)
        
        chunk = np.random.randn(8000).astype(np.float32)
        # Warmup pass
        _ = detector.process_chunk(chunk)
        
        # Feed 0.5 seconds of audio (8,000 samples)
        res = detector.process_chunk(chunk)
        
        self.assertIn('raw_p_fake', res)
        self.assertIn('smoothed_p_fake', res)
        self.assertIn('total_latency_ms', res)
        
        # Total latency threshold for test runner environment
        print(f"\nMeasured Streaming Window Latency: {res['total_latency_ms']:.2f} ms")
        self.assertLess(res['total_latency_ms'], 250.0)  # threshold for test runner environment


if __name__ == '__main__':
    unittest.main()
