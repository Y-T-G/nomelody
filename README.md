# vocal-isolate 🎙️✨

Lightweight, on-device AI vocal and instrumental stem separator for videos and audio files. Powered by **ONNX Runtime** and fine-tuned **MDX-Net** architecture.

Runs completely offline without cloud APIs or gigabytes of heavy PyTorch dependencies.

---

## Features

- **Video-First**: Extracts audio, separates stems, and muxes directly back into your video without re-encoding the video stream (`-c:v copy`).
- **Zero PyTorch Footprint**: Uses ONNX Runtime with multi-threaded CPU and ARM NEON acceleration (XNNPACK).
- **Audio & Video Support**: Supports MP4, MKV, MOV, WebM, FLAC, WAV, MP3, and more.
- **Vocal & Instrumental Extraction**: Isolate clean vocals, extract karaoke / instrumental backing, or produce both in a single command.
- **Android / Termux Native**: Seamlessly integrates with Termux on Android, including automatic `content call` indexing into the Android MediaStore.
- **Auto Model Management**: Automatically downloads the fine-tuned [`UVR-MDX-NET-Voc_FT.onnx`](https://github.com/TRvlvr/model_repo) (~64 MB) on first launch.

---

## Installation

### Termux (Android)

```bash
# 1. Install prerequisites
pkg install -y ffmpeg python-onnxruntime python-numpy curl

# 2. Clone repository
git clone https://github.com/Y-T-G/vocal-isolate.git
cd vocal-isolate

# 3. Install executable into ~/bin
./install.sh
```

### Linux / macOS / Windows

```bash
# 1. Ensure ffmpeg is installed
# macOS: brew install ffmpeg
# Debian/Ubuntu: sudo apt install ffmpeg

# 2. Clone repository & install python dependencies
git clone https://github.com/Y-T-G/vocal-isolate.git
cd vocal-isolate
pip install -r requirements.txt

# 3. Run directly or link to PATH
chmod +x vocal_isolate.py
sudo ln -s $(pwd)/vocal_isolate.py /usr/local/bin/vocal-isolate
```

---

## Usage

### 1. Remove music / isolate voice (default)
```bash
vocal-isolate input_video.mp4
# Output: input_video (vocals).mp4
```

### 2. Remove voice / extract instrumental backing
```bash
vocal-isolate --inst input_video.mp4
# Output: input_video (instrumental).mp4
```

### 3. Generate both vocals and instrumental versions
```bash
vocal-isolate --both input_video.mp4
```

### 4. Batch processing
```bash
vocal-isolate clip1.mp4 clip2.mp4 podcast.mkv
```

### 5. Custom output file
```bash
vocal-isolate speech.mp4 -o voice_only.mp4
```

---

## CLI Options

```text
usage: vocal-isolate [-h] [-o OUTPUT] [--inst] [--both] [--model MODEL] [-t THREADS] inputs [inputs ...]

Local AI stem separator (vocal and instrumental isolation for videos and audio)

positional arguments:
  inputs                Input video or audio file(s)

options:
  -h, --help            Show this help message and exit
  -o, --output OUTPUT   Explicit output file path (single file only)
  --inst, --instrumental
                        Keep instrumental instead of vocals
  --both                Generate both vocal and instrumental versions
  --model MODEL         Path to custom MDX-Net ONNX model file
  -t, --threads THREADS Inference worker threads (default: 4)
```

---

## How It Works

1. **Audio Extraction**: `ffmpeg` extracts the input audio track into a 44.1 kHz stereo PCM stream.
2. **STFT Spectrogram**: Audio chunks are converted into frequency-domain spectrograms using Short-Time Fourier Transform ($N_{\text{fft}} = 6144$, $\text{hop} = 1024$, $\text{bins} = 3072$).
3. **Neural Inference**: The ONNX MDX-Net model predicts vocal frequency masks across stereo time-frequency tiles.
4. **iSTFT & Synthesis**: Inverse STFT reconstructs time-domain audio with Hann overlap-add blending and volume compensation.
5. **Lossless Video Muxing**: The original video track is paired with the processed audio stream into the final container.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
