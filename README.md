# nomelody 🎙️✨

Lightweight, on-device AI voice isolation for videos and audio files. Removes background music and melody while preserving speech, powered by **ONNX Runtime** and the fine-tuned **MDX-Net** architecture.

Runs completely offline on your device without cloud APIs or gigabytes of heavy PyTorch dependencies.

---

## Features

- **Video-First**: Extracts audio, strips background music, and muxes the voice track cleanly back into your video without re-encoding the video stream (`-c:v copy`).
- **Zero PyTorch Footprint**: Runs via ONNX Runtime with multi-threaded CPU and ARM NEON acceleration (XNNPACK).
- **Format Support**: Supports MP4, MKV, MOV, WebM, FLAC, WAV, MP3, and more.
- **Android / Termux Native**: Works seamlessly inside Termux on Android, with automatic MediaStore indexing (`content call`) on shared storage.
- **Auto Model Management**: Automatically downloads the fine-tuned [`UVR-MDX-NET-Voc_FT.onnx`](https://github.com/TRvlvr/model_repo) (~64 MB) on first run.

---

## Installation

### Termux (Android)

```bash
# 1. Install prerequisites
pkg install -y ffmpeg python-onnxruntime python-numpy curl

# 2. Clone repository
git clone https://github.com/Y-T-G/nomelody.git
cd nomelody

# 3. Install executable into ~/bin
./install.sh
```

### Linux / macOS / Windows

```bash
# 1. Ensure ffmpeg is installed
# macOS: brew install ffmpeg
# Debian/Ubuntu: sudo apt install ffmpeg

# 2. Clone repository & install python dependencies
git clone https://github.com/Y-T-G/nomelody.git
cd nomelody
pip install -r requirements.txt

# 3. Run directly or link to PATH
chmod +x nomelody.py
sudo ln -s $(pwd)/nomelody.py /usr/local/bin/nomelody
```

---

## Usage

### 1. Remove background music from video (default)
```bash
nomelody input_video.mp4
# Output: input_video (nomelody).mp4
```

### 2. Batch processing
```bash
nomelody clip1.mp4 clip2.mp4 podcast.mkv
```

### 3. Custom output file
```bash
nomelody speech.mp4 -o voice_only.mp4
```

---

## CLI Options

```text
usage: nomelody [-h] [-o OUTPUT] [--model MODEL] [-t THREADS] inputs [inputs ...]

nomelody: Local AI vocal isolation for videos and audio (removes background music/melody)

positional arguments:
  inputs                Input video or audio file(s)

options:
  -h, --help            Show this help message and exit
  -o, --output OUTPUT   Explicit output file path (single file only)
  --model MODEL         Path to custom MDX-Net ONNX model file
  -t, --threads THREADS Inference worker threads (default: 4)
```

---

## How It Works

1. **Audio Extraction**: `ffmpeg` extracts the input audio track into a 44.1 kHz stereo PCM stream.
2. **STFT Spectrogram**: Audio chunks are converted into frequency-domain spectrograms using Short-Time Fourier Transform ($N_{\text{fft}} = 6144$, $\text{hop} = 1024$, $\text{bins} = 3072$).
3. **Neural Inference**: The ONNX MDX-Net model predicts vocal frequency masks across stereo time-frequency tiles.
4. **iSTFT & Synthesis**: Inverse STFT reconstructs time-domain speech with Hann overlap-add blending and volume compensation.
5. **Lossless Video Muxing**: The original video track is paired with the isolated voice stream into the final container.

---

## Acknowledgments & Credits

This tool relies on vocal separation models trained by the open-source audio research community:

- **[Kimberley Jensen](https://github.com/KimberleyJensen)** – Implemented training pipelines and fine-tuned the MDX-Net vocal separation models (`Voc_FT` / `Kim_Vocal`).
- **[Anjok07](https://github.com/Anjok07)** and **[TRvlvr](https://github.com/TRvlvr)** – Creators and core maintainers of [Ultimate Vocal Remover (UVR)](https://github.com/Anjok07/ultimatevocalremovergui) and the [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo) checkpoint repository.
- **[Kuielab & Woosung Choi](https://github.com/kuielab)** – Original creators of the [MDX-Net](https://github.com/kuielab/mdx-net) architecture (Sony Music Demixing Challenge 2021).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
