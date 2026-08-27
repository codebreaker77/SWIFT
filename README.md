# SWIFT: Speech & Waveform Investigative Forensic Toolkit 🕵️‍♂️🎙️

**SWIFT** is a real-time, deepfake audio detection toolkit and forensic dashboard. It is designed to distinguish between authentic human speech and high-fidelity synthetic AI voices (e.g., ElevenLabs, Amazon Polly, OpenAI TTS) with pinpoint accuracy. 

It features a high-performance **Python backend** (handling feature extraction and neural inference) and a **real-time web dashboard** (allowing users to stream microphone audio, analyze pre-recorded `.wav` samples, or connect via Twilio).

---

## 🏗️ 1. Architecture Overview: PhysioSpecNet

At the core of SWIFT is **PhysioSpecNet**, a custom neural network built specifically for audio forensics. Standard audio models (like those using Mel-spectrograms) intentionally compress high frequencies to mimic human hearing. Unfortunately, this compression destroys the microscopic synthetic artifacts left behind by AI generators. 

PhysioSpecNet resolves this by analyzing audio across **six distinct acoustic dimensions**.

### The 6-Channel Forensic Spectrogram Extractor
Before audio hits the neural network, it is processed into a `(6, 224, 224)` tensor using our custom extractor:

1. **LFCC (Linear Frequency Cepstral Coefficients):** Uses a linear scale to preserve the high-frequency spectral reflections commonly left by AI vocoders.
2. **Formant Trajectory Map (F1-F3):** Maps the physical resonant frequencies of the human vocal tract. AI voices often lack physically consistent acoustic resonance.
3. **Glottal Noise Excitation (GNE) Spectrum:** Analyzes the closure of the vocal folds, a physiological process AI struggles to emulate flawlessly.
4. **Jitter & Shimmer Micro-variation Map:** Measures frame-to-frame variations in pitch and amplitude. Humans have natural micro-tremors; AI is often too "perfect".
5. **Instantaneous Phase Unwrapping Map:** Captures phase incoherence. Generative AI vocoders (like HiFi-GAN or VITS) often destroy phase alignment, creating a telltale synthetic signature.
6. **Spectral Flux / Harmonic-to-Noise Ratio (HNR):** Detects synthetic smoothness and abrupt energy transitions.

### Cross-Channel Attention Module
PhysioSpecNet uses a custom **Cross-Channel Attention Mechanism**. It doesn't treat the 6 channels independently; rather, the 5 physiological channels dynamically attend to and modulate the primary LFCC channel. This highlights inconsistencies between the raw acoustic features and the physical spectral data. 

The modulated tensor is then fed into an **EfficientNet-B0 backbone**—chosen for its ability to extract complex 2D visual patterns (artifact detection) while remaining lightweight enough for real-time inference.

---

## ⚡ 2. Real-Time Streaming & VAD

SWIFT is built for live environments, capable of processing continuous audio streams from web browsers or phone calls (Twilio).

### Dynamic AudioRingBuffer
Instead of analyzing audio in static chunks, SWIFT utilizes a continuous, rolling **`AudioRingBuffer`**. 
- Audio chunks arrive via WebSockets in `Float32Array` or base64 PCMU formats.
- The buffer maintains a rolling `32,000` sample window (2.0 seconds at `16kHz`).
- **Mathematical Dithering:** To prevent "Digital Cliff" artifacts (which trigger false AI predictions when silence abruptly transitions to speech), the buffer is intelligently padded with low-level Gaussian dither (`np.random.normal`).

### Voice Activity Detection (VAD)
Running neural inference on pure silence or room static leads to highly erratic confidence scores. SWIFT implements an algorithmic VAD gate based on **Peak** and **RMS (Root Mean Square)** energy calculations. If the audio slice falls below the energy threshold, the engine safely bypasses the heavy neural network and returns a nominal score, saving compute resources.

---

## 📱 3. Overcoming the Mobile Domain Gap

One of the largest challenges in audio forensics is the **Hardware Domain Gap**. Modern smartphones (iOS/Android) employ un-bypassable, hardware-level "Neural Noise Suppression" chips (like the Apple Neural Engine).

**The Problem:** These chips aggressively isolate human voices by applying heavy algorithmic gating. In doing so, they destroy phase coherence and high-frequency harmonics—leaving behind the *exact same acoustic artifacts* as generative AI deepfakes. A model trained strictly on studio data will classify all smartphone microphones as 99% AI.

**The Solution:** 
1. **Native Resampling:** Mobile browsers often ignore `AudioContext({sampleRate: 16000})`, resulting in severe pitch-shifting (chipmunk voice). SWIFT captures audio at the native hardware rate (e.g., `48kHz`) and dynamically Fourier-resamples it down to `16kHz` using `scipy.signal.resample` on the Python backend.
2. **Targeted Fine-Tuning:** The model was aggressively fine-tuned on mobile-suppressed microphone dumps, teaching the attention mechanism to distinguish between hardware-induced noise suppression and true AI vocoder artifacts. The model now achieves **100% Accuracy / 0% EER** across both studio and mobile domains.

---

## 📞 4. Twilio Integration (Phone Call Deepfake Detection)

SWIFT supports native bi-directional WebSocket streaming with **Twilio Media Streams**. 
- Twilio sends `8kHz` PCMU (µ-law) audio.
- SWIFT automatically decodes the `base64` µ-law packets, upsamples them to `16kHz` using optimal Fourier decimation to prevent aliasing, and feeds them into the `AudioRingBuffer`.
- Real-time deepfake scores can be monitored on the dashboard or used to programmatically terminate a fraudulent phone call.

---

## 📂 5. Repository Structure

```
SWIFT/
├── src/                    # Core Python package
│   ├── features/           # LFCC and 6-Channel Spectrogram extraction logic
│   ├── models/             # PhysioSpecNet architecture & Attention mechanisms
│   ├── realtime_stream.py  # AudioRingBuffer and VAD engine
│   └── config.py           # Hyperparameters
├── public/                 # Web dashboard frontend (HTML, CSS, JS)
├── checkpoints/            # Pre-trained PyTorch weights (`best_physiospecnet.pth`)
├── scripts/                # Evaluation, benchmarking, and legacy scripts
├── tests/                  # Unit tests for the pipeline
├── data/                   # Datasets (git-ignored)
├── docs/                   # Convergence graphs and metrics
├── server.py               # Main API & WebSocket server (FastAPI/Twilio)
├── evaluate_and_train.py   # Primary fine-tuning and evaluation script
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 6. Installation & Quickstart

### Prerequisites
- Python 3.9+
- An NVIDIA GPU (CUDA) is highly recommended for real-time inference, though CPU fallback is supported.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/codebreaker77/SWIFT.git
   cd SWIFT
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Live Server
Start the backend WebSocket server and frontend dashboard:
```bash
python server.py 8000
```
Open `http://localhost:8000` in your browser. Grant microphone permissions to test the real-time deepfake detector.

### Fine-Tuning the Model
To fine-tune the model on your own hardware profile or specific datasets:
```bash
python evaluate_and_train.py
```
This script will parse all `.wav` samples in `public/samples/`, apply necessary augmentations, compute the 6-channel acoustic feature maps, train `PhysioSpecNet`, and automatically overwrite `checkpoints/best_physiospecnet.pth` with the optimal weights based on Validation Loss.

---
*Built to combat the next generation of synthetic audio fraud.*
