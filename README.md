# nomelody 🎙️✨

Lightweight, on-device AI voice isolation for videos and audio files. Removes background music and melody while preserving speech, powered by **ONNX Runtime** and the fine-tuned **MDX-Net** architecture.

Runs completely offline on your device without cloud APIs or gigabytes of heavy PyTorch dependencies.

---

## Features

- **Video-First**: Extracts audio, strips background music, and muxes the voice track cleanly back into your video without re-encoding the video stream (`-c:v copy`).
- **Flexible Acceleration**: Supports both **ONNX Runtime** (multi-threaded CPU/NEON) and **Google LiteRT GPU** (OpenCL hardware acceleration on Qualcomm Adreno / Mali GPUs).
- **Zero Heavy PyTorch**: Runs completely offline without multi-gigabyte PyTorch/CUDA installs.
- **Format Support**: Supports MP4, MKV, MOV, WebM, FLAC, WAV, MP3, and more.
- **Android / Termux Native**: Works seamlessly inside Termux on Android, with automatic MediaStore indexing (`content call`) on shared storage.
- **Auto Model & Delegate Management**: Automatically fetches model weights (`UVR-MDX-NET-Voc_FT.onnx` or `uvr_mdx_voc_ft_fp16.tflite`) and GPU delegates on first run.

---

## Installation

### Termux (Android)

```bash
# 1. Install prerequisites (for default ONNX or LiteRT)
pkg install -y ffmpeg python-numpy curl
pkg install -y python-onnxruntime     # for ONNX CPU backend (default)
pkg install -y python-tflite-runtime  # for LiteRT GPU / CPU backend

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

### 1. Default (Fast multi-threaded CPU via ONNX Runtime)
```bash
nomelody input_video.mp4
# Output: input_video (nomelody).mp4
```

### 2. LiteRT GPU Acceleration (Adreno / OpenCL)
```bash
nomelody --backend litert-gpu input_video.mp4
```

### 3. LiteRT CPU (XNNPACK)
```bash
nomelody --backend litert-cpu input_video.mp4
```

### 4. Batch processing
```bash
nomelody clip1.mp4 clip2.mp4 podcast.mkv
```

### 5. Custom output file
```bash
nomelody speech.mp4 -o voice_only.mp4
```

---

## CLI Options

```text
usage: nomelody [-h] [-o OUTPUT] [--backend {onnx,litert-gpu,litert-cpu}] [--model MODEL] [-t THREADS] inputs [inputs ...]

nomelody: Local AI vocal isolation for videos and audio (removes background music/melody)

positional arguments:
  inputs                Input video or audio file(s)

options:
  -h, --help            Show this help message and exit
  -o, --output OUTPUT   Explicit output file path (single file only)
  --backend {onnx,litert-gpu,litert-cpu}
                        Inference engine: 'onnx' (CPU NEON, default), 'litert-gpu' (OpenCL GPU delegate), 'litert-cpu' (LiteRT XNNPACK CPU)
  --model MODEL         Path to custom model file (.onnx or .tflite)
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
