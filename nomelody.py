#!/usr/bin/env python3
"""
nomelody: On-device voice isolation for video and audio (removes background music / melody).
Supports ONNX Runtime and Google LiteRT (TensorFlow Lite) with OpenCL GPU acceleration on Android.

Model & Architecture Credits:
- Model Weights: UVR-MDX-NET-Voc_FT fine-tuned by Kimberley Jensen (KimberleyJSN).
- UVR Ecosystem: Anjok07 & TRvlvr (https://github.com/Anjok07/ultimatevocalremovergui).
- MDX-Net Architecture: Kuielab & Woosung Choi (https://github.com/kuielab/mdx-net).
"""

import os
import sys
import warnings
import argparse
import subprocess
import wave
import tempfile
import ctypes
import numpy as np

# Suppress runtime warnings on Android/Termux
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = os.path.expanduser("~/.local/share/models")
LIB_DIR = os.path.expanduser("~/.local/lib")

ONNX_MODEL_NAME = "UVR-MDX-NET-Voc_FT.onnx"
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, ONNX_MODEL_NAME)
ONNX_MODEL_URL = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{ONNX_MODEL_NAME}"

TFLITE_MODEL_NAME = "uvr_mdx_voc_ft_fp16.tflite"
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)
TFLITE_MODEL_URL = f"https://huggingface.co/skytrix/kirexa-mdx-tflite/resolve/main/{TFLITE_MODEL_NAME}"

GPU_DELEGATE_SO = os.path.join(LIB_DIR, "libtflite_gpu_delegate.so")
GPU_DELEGATE_AAR_URL = "https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite-gpu/2.16.1/tensorflow-lite-gpu-2.16.1.aar"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".ts", ".m4v"}


class GpuDelegateOptionsV2(ctypes.Structure):
    _fields_ = [
        ("is_precision_loss_allowed", ctypes.c_int32),
        ("inference_preference", ctypes.c_int32),
        ("inference_priority1", ctypes.c_int32),
        ("inference_priority2", ctypes.c_int32),
        ("inference_priority3", ctypes.c_int32),
        ("experimental_flags", ctypes.c_int32),
        ("max_delegated_partitions", ctypes.c_int32),
        ("model_token", ctypes.c_char_p),
        ("serialization_dir", ctypes.c_char_p),
    ]


class LiteRtGpuDelegate:
    def __init__(self, allow_fp16: bool = True):
        self.lib = ctypes.CDLL(GPU_DELEGATE_SO)
        self.lib.TfLiteGpuDelegateOptionsV2Default.restype = GpuDelegateOptionsV2
        self.lib.TfLiteGpuDelegateV2Create.argtypes = [ctypes.POINTER(GpuDelegateOptionsV2)]
        self.lib.TfLiteGpuDelegateV2Create.restype = ctypes.c_void_p
        self.lib.TfLiteGpuDelegateV2Delete.argtypes = [ctypes.c_void_p]

        opts = self.lib.TfLiteGpuDelegateOptionsV2Default()
        if allow_fp16:
            opts.is_precision_loss_allowed = 1
        self._delegate_ptr = self.lib.TfLiteGpuDelegateV2Create(ctypes.byref(opts))

    def _get_native_delegate_pointer(self):
        return self._delegate_ptr

    def __del__(self):
        if getattr(self, "_delegate_ptr", None):
            self.lib.TfLiteGpuDelegateV2Delete(self._delegate_ptr)
            self._delegate_ptr = None


def ensure_file(path: str, url: str, desc: str, min_size: int = 1_000_000):
    if os.path.isfile(path) and os.path.getsize(path) >= min_size:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"Downloading {desc} to {path}...")
    subprocess.run(["curl", "-L", "-o", path, url], check=True)


def ensure_gpu_delegate():
    if os.path.isfile(GPU_DELEGATE_SO) and os.path.getsize(GPU_DELEGATE_SO) > 500_000:
        return
    os.makedirs(LIB_DIR, exist_ok=True)
    print("Downloading LiteRT GPU acceleration delegate (libtflite_gpu_delegate.so)...")
    with tempfile.NamedTemporaryFile(suffix=".aar") as tmp_aar:
        subprocess.run(["curl", "-sL", "-o", tmp_aar.name, GPU_DELEGATE_AAR_URL], check=True)
        subprocess.run(
            ["unzip", "-p", tmp_aar.name, "jni/arm64-v8a/libtensorflowlite_gpu_jni.so"],
            stdout=open(GPU_DELEGATE_SO, "wb"),
            check=True,
        )


def scan_mediastore(path: str):
    abs_p = os.path.abspath(path)
    if abs_p.startswith(("/storage/emulated/0", "/sdcard", "/data/media/0")):
        cmd = f'content call --uri content://media --method scan_file --arg "{abs_p}"'
        subprocess.run(["su", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ModelRunner:
    def __init__(self, backend: str, custom_model: str = None, threads: int = 4):
        self.backend = backend
        self.threads = threads

        if backend == "onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                print("Error: onnxruntime not found. Install via 'pkg install python-onnxruntime'.", file=sys.stderr)
                sys.exit(1)
            model_path = custom_model or ONNX_MODEL_PATH
            ensure_file(model_path, ONNX_MODEL_URL, "ONNX model (Voc_FT)")
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = threads
            opts.intra_op_num_threads = threads
            self.session = ort.InferenceSession(model_path, sess_options=opts)
            self.infer_fn = lambda x: self.session.run(None, {"input": x})[0][0]

        elif backend in ("litert-gpu", "litert-cpu"):
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                print("Error: tflite_runtime not found. Install via 'pkg install python-tflite-runtime'.", file=sys.stderr)
                sys.exit(1)

            model_path = custom_model or TFLITE_MODEL_PATH
            ensure_file(model_path, TFLITE_MODEL_URL, "LiteRT FP16 model (Voc_FT)")

            delegates = []
            if backend == "litert-gpu":
                ensure_gpu_delegate()
                self.gpu_del = LiteRtGpuDelegate(allow_fp16=True)
                delegates.append(self.gpu_del)

            self.interp = tflite.Interpreter(
                model_path=model_path,
                num_threads=threads if backend == "litert-cpu" else None,
                experimental_delegates=delegates if delegates else None,
            )
            self.interp.allocate_tensors()
            self.in_idx = self.interp.get_input_details()[0]["index"]
            self.out_idx = self.interp.get_output_details()[0]["index"]

            def _tflite_infer(x):
                self.interp.set_tensor(self.in_idx, x)
                self.interp.invoke()
                return self.interp.get_tensor(self.out_idx)[0]

            self.infer_fn = _tflite_infer

    def run(self, model_in: np.ndarray) -> np.ndarray:
        return self.infer_fn(model_in)


def process_file(input_file: str, runner: ModelRunner, output_file: str = None):
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}", file=sys.stderr)
        return False

    base, ext = os.path.splitext(input_file)
    is_video = ext.lower() in VIDEO_EXTS

    if not output_file:
        output_file = f"{base} (nomelody){ext}"

    print(f"\nProcessing: {os.path.basename(input_file)} [Backend: {runner.backend}]")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in_wav = os.path.join(tmpdir, "input.wav")
        tmp_out_wav = os.path.join(tmpdir, "stem.wav")

        res = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_file,
                "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
                tmp_in_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if res.returncode != 0:
            print("Failed to decode audio track via ffmpeg", file=sys.stderr)
            return False

        with wave.open(tmp_in_wav, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio.reshape(-1, 2).T
        total_samples = audio.shape[1]
        print(f"  Duration: {total_samples/sr:.2f}s | Sample rate: {sr}Hz stereo")

        n_fft = 6144
        hop = 1024
        dim_f = 3072
        dim_t = 256
        chunk_size = hop * (dim_t - 1)
        step = chunk_size // 2

        window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
        win_sq = window ** 2
        pad = n_fft // 2

        padded_audio = np.pad(audio, ((0, 0), (0, chunk_size)), mode="constant")
        recon = np.zeros((2, padded_audio.shape[1]), dtype=np.float32)
        weight = np.zeros(padded_audio.shape[1], dtype=np.float32)
        chunk_win = np.hanning(chunk_size).astype(np.float32)

        starts = list(range(0, total_samples, step))
        total_chunks = len(starts)

        for idx, start in enumerate(starts):
            end = start + chunk_size
            chunk = padded_audio[:, start:end]

            c_pad = np.pad(chunk, ((0, 0), (pad, pad)), mode="reflect")
            frames = np.lib.stride_tricks.sliding_window_view(c_pad, n_fft, axis=-1)[:, ::hop, :]
            spec = np.fft.rfft(frames * window, n=n_fft, axis=-1)
            spec = np.transpose(spec, (0, 2, 1))

            spec = spec[:, :dim_f, :dim_t]
            spec[:, :3, :] = 0.0

            model_in = np.stack(
                [spec.real[0], spec.imag[0], spec.real[1], spec.imag[1]], axis=0
            )[None, ...].astype(np.float32)

            pred = runner.run(model_in)

            pred_c0 = pred[0] + 1j * pred[1]
            pred_c1 = pred[2] + 1j * pred[3]
            pred_spec = np.stack([pred_c0, pred_c1], axis=0)

            pred_spec_padded = np.pad(pred_spec, ((0, 0), (0, 1), (0, 0)), mode="constant")
            pred_frames = np.transpose(pred_spec_padded, (0, 2, 1))

            iframes = np.fft.irfft(pred_frames, n=n_fft, axis=-1) * window

            chunk_recon = np.zeros((2, pad + chunk_size + pad), dtype=np.float32)
            chunk_w = np.zeros(pad + chunk_size + pad, dtype=np.float32)
            for f_i in range(dim_t):
                s_pos = f_i * hop
                chunk_recon[:, s_pos : s_pos + n_fft] += iframes[:, f_i, :]
                chunk_w[s_pos : s_pos + n_fft] += win_sq

            mask = chunk_w > 1e-6
            chunk_recon[:, mask] /= chunk_w[mask]
            chunk_out = chunk_recon[:, pad : pad + chunk_size]

            recon[:, start:end] += chunk_out * chunk_win
            weight[start:end] += chunk_win
            print(f"\r  Removing melody: [{idx + 1}/{total_chunks}] chunks", end="", flush=True)

        print("")
        valid = weight > 1e-6
        recon[:, valid] /= weight[valid]
        vocal = recon[:, :total_samples] * 1.035

        peak = np.max(np.abs(vocal))
        if peak > 0.98:
            vocal = vocal * (0.98 / peak)

        out_data = (vocal.T * 32767.0).astype(np.int16)
        with wave.open(tmp_out_wav, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(out_data.tobytes())

        if is_video:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", input_file, "-i", tmp_out_wav,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    output_file,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_out_wav, "-c:a", "flac", output_file],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    scan_mediastore(output_file)
    print(f"  Saved: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="nomelody: Local AI vocal isolation for videos and audio (removes background music/melody)"
    )
    parser.add_argument("inputs", nargs="+", help="Input video or audio file(s)")
    parser.add_argument("-o", "--output", help="Explicit output file path (single file only)")
    parser.add_argument(
        "--backend",
        choices=["onnx", "litert-gpu", "litert-cpu"],
        default="onnx",
        help="Inference engine: 'onnx' (CPU NEON, default), 'litert-gpu' (LiteRT OpenCL GPU delegate), 'litert-cpu' (LiteRT XNNPACK CPU)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to custom model file (.onnx or .tflite)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Inference worker threads (default: 4)",
    )
    args = parser.parse_args()

    runner = ModelRunner(backend=args.backend, custom_model=args.model, threads=args.threads)

    for inp in args.inputs:
        process_file(inp, runner=runner, output_file=args.output)


if __name__ == "__main__":
    main()
