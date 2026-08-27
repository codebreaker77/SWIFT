import numpy as np
import scipy.io.wavfile as wavfile
import os

if not os.path.exists("noise_samples"):
    os.makedirs("noise_samples")

for i in range(150):
    sr = 16000
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration))
    white = np.random.randn(len(t))
    pink = np.cumsum(np.random.randn(len(t)))
    pink = pink - np.mean(pink)
    pink = pink / np.max(np.abs(pink))
    
    if np.random.rand() > 0.5:
        freq = np.random.randint(200, 2000)
        sine = np.sin(2 * np.pi * freq * t)
    else:
        sine = np.zeros(len(t))
        
    mixed = white * np.random.rand() + pink * np.random.rand() + sine * np.random.rand()
    mixed = mixed / np.max(np.abs(mixed))
    
    wavfile.write(f"noise_samples/noise_{i}.wav", sr, (mixed * 32767).astype(np.int16))
print("Generated 150 synthetic noise samples in 'noise_samples/'")
