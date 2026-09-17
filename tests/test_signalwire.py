"""
Tests for SignalWire Call Media Stream Ingestion and G.711 mu-law processing.
"""

import unittest
import numpy as np
import base64
from server import decode_mulaw, encode_mulaw, resample_8k_to_16k, STREAM_DETECTOR

class TestSignalWireIngestion(unittest.TestCase):
    def test_mulaw_encode_decode(self):
        # Generate 8kHz sine wave
        t = np.linspace(0, 1, 8000, endpoint=False, dtype=np.float32)
        original_pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Encode to mu-law
        mulaw_bytes = encode_mulaw(original_pcm)
        self.assertEqual(len(mulaw_bytes), 8000)
        
        # Decode back to float32 PCM
        decoded_pcm = decode_mulaw(mulaw_bytes)
        self.assertEqual(len(decoded_pcm), 8000)
        
        # G.711 mu-law has quantization noise, correlation should be > 0.99
        corr = np.corrcoef(original_pcm, decoded_pcm)[0, 1]
        self.assertGreater(corr, 0.99)

    def test_resample_8k_to_16k(self):
        audio_8k = np.random.randn(8000).astype(np.float32)
        audio_16k = resample_8k_to_16k(audio_8k)
        self.assertEqual(len(audio_16k), 16000)

    def test_signalwire_stream_chunk_detection(self):
        # Generate sample frame and encode to base64
        t = np.linspace(0, 0.2, 1600, endpoint=False, dtype=np.float32)
        chunk_8k = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        mulaw_b64 = base64.b64encode(encode_mulaw(chunk_8k)).decode('utf-8')
        
        raw_mulaw = base64.b64decode(mulaw_b64)
        audio_8k = decode_mulaw(raw_mulaw)
        audio_16k = resample_8k_to_16k(audio_8k)
        
        res = STREAM_DETECTOR.process_chunk(audio_16k)
        self.assertIn('smoothed_p_fake', res)
        self.assertIn('total_latency_ms', res)
        self.assertLess(res['total_latency_ms'], 100.0)

if __name__ == '__main__':
    unittest.main()
