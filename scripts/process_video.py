from moviepy import VideoFileClip
import librosa
import soundfile as sf
import os
import numpy as np

# Extract audio using moviepy
print("Extracting audio from train.mp4...")
try:
    clip = VideoFileClip("train.mp4")
    clip.audio.write_audiofile("temp_audio.wav", logger=None)
except Exception as e:
    print(f"Error extracting audio: {e}")
    exit(1)

# Load audio at 32kHz
print("Loading audio...")
sr = 32000
y, _ = librosa.load("temp_audio.wav", sr=sr)

# Remove silence
print("Removing silence...")
intervals = librosa.effects.split(y, top_db=25)
y_non_silent = []
for start, end in intervals:
    y_non_silent.extend(y[start:end])
    
y_non_silent = np.array(y_non_silent)

# Split into 2.0 second chunks (64000 samples)
chunk_size = 64000
chunks = [y_non_silent[i:i + chunk_size] for i in range(0, len(y_non_silent), chunk_size)]

os.makedirs("public/samples", exist_ok=True)
count = 0
for i, chunk in enumerate(chunks):
    if len(chunk) >= 32000: # at least 1.0 second
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        sf.write(f"public/samples/gptlive_spoof_{i}.wav", chunk, sr)
        count += 1
print(f"Saved {count} GPT Live chunks to public/samples/")
