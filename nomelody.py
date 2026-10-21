#!/usr/bin/env python3
"""
nomelody: On-device voice isolation for video and audio (removes background music / melody).
Supports ONNX Runtime (FP32 & calibrated Static INT8) and Google LiteRT with OpenCL GPU acceleration.

Model & Architecture Credits:
- Model Weights: UVR-MDX-NET-Voc_FT fine-tuned by Kimberley Jensen (KimberleyJSN).
- UVR Ecosystem: Anjok07 & TRvlvr (https://github.com/Anjok07/ultimatevocalremovergui).
- MDX-Net Architecture: Kuielab & Woosung Choi (https://github.com/kuielab/mdx-net).
"""

import os
import sys
import time
import json
import queue
import socket
import shutil
import threading
import webbrowser
import warnings
import argparse
import subprocess
import tempfile
import ctypes
import http.server
import numpy as np

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Suppress runtime warnings on Android/Termux
warnings.filterwarnings("ignore", category=UserWarning)

# Multithreaded FFT via scipy (pocketfft) with fallback to numpy.fft
try:
    import scipy.fft as _sfft

    def fft_rfft(x, n=None, axis=-1):
        return _sfft.rfft(x, n=n, axis=axis, workers=-1)

    def fft_irfft(x, n=None, axis=-1):
        return _sfft.irfft(x, n=n, axis=axis, workers=-1)
except ImportError:
    def fft_rfft(x, n=None, axis=-1):
        return np.fft.rfft(x, n=n, axis=axis)

    def fft_irfft(x, n=None, axis=-1):
        return np.fft.irfft(x, n=n, axis=axis)

MODEL_DIR = os.path.expanduser("~/.local/share/models")
LIB_DIR = os.path.expanduser("~/.local/lib")

RELEASE_TAG = "v0.1.0"
ASSET_BASE_URL = f"https://github.com/Y-T-G/nomelody/releases/download/{RELEASE_TAG}"

ONNX_MODEL_NAME = "UVR-MDX-NET-Voc_FT.onnx"
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, ONNX_MODEL_NAME)
ONNX_MODEL_URL = f"{ASSET_BASE_URL}/{ONNX_MODEL_NAME}"

INT8_MODEL_NAME = "UVR-MDX-NET-Voc_FT_int8_calibrated.onnx"
INT8_MODEL_PATH = os.path.join(MODEL_DIR, INT8_MODEL_NAME)
INT8_MODEL_URL = f"{ASSET_BASE_URL}/{INT8_MODEL_NAME}"

TFLITE_MODEL_NAME = "uvr_mdx_voc_ft_fp16.tflite"
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)
TFLITE_MODEL_URL = f"{ASSET_BASE_URL}/{TFLITE_MODEL_NAME}"

TFLITE_FP32_NAME = "uvr_mdx_voc_ft_fp32.tflite"
TFLITE_FP32_PATH = os.path.join(MODEL_DIR, TFLITE_FP32_NAME)
TFLITE_FP32_URL = f"{ASSET_BASE_URL}/{TFLITE_FP32_NAME}"

TFLITE_INT8_NAME = "uvr_mdx_voc_ft_int8.tflite"
TFLITE_INT8_PATH = os.path.join(MODEL_DIR, TFLITE_INT8_NAME)
TFLITE_INT8_URL = f"{ASSET_BASE_URL}/{TFLITE_INT8_NAME}"

GPU_DELEGATE_SO = os.path.join(LIB_DIR, "libtflite_gpu_delegate.so")
GPU_DELEGATE_URL = f"{ASSET_BASE_URL}/libtflite_gpu_delegate.so"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".ts", ".m4v"}


def get_optimal_threads(requested: int = None) -> int:
    if requested and requested > 0:
        return requested
    try:
        count = os.cpu_count()
        if count:
            return min(8, count)
    except Exception:
        pass
    return 4


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


def ensure_int8_model(fp32_path: str, int8_path: str):
    if os.path.isfile(int8_path) and os.path.getsize(int8_path) >= 10_000_000:
        return
    try:
        ensure_file(int8_path, INT8_MODEL_URL, "Calibrated static INT8 model (17 MB)", min_size=10_000_000)
    except Exception as e:
        print(f"Direct download of INT8 model failed ({e}), falling back to local calibration...")
        ensure_file(fp32_path, ONNX_MODEL_URL, "ONNX FP32 base model (Voc_FT)")
        print("Calibrating static INT8 quantized model for high CPU throughput...")
        try:
            from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

            class SyntheticAudioCalibrator(CalibrationDataReader):
                def __init__(self):
                    # Frequency-sloped calibration data simulating real audio spectrogram energy decay
                    freq_profile = 1.0 / np.linspace(1, 20, 3072)[:, None].astype(np.float32)
                    self.data = [
                        (np.random.randn(1, 4, 3072, 256).astype(np.float32) * freq_profile)
                        for _ in range(8)
                    ]
                    self.enum = iter(self.data)

                def get_next(self):
                    t = next(self.enum, None)
                    return {"input": t} if t is not None else None

            quantize_static(
                model_input=fp32_path,
                model_output=int8_path,
                calibration_data_reader=SyntheticAudioCalibrator(),
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                per_channel=True,
            )
            print(f"  Generated calibrated INT8 model ({os.path.getsize(int8_path)/1e6:.1f}MB)")
        except Exception as err:
            print(f"Static quantization failed ({err}), falling back to FP32 model.")


def ensure_gpu_delegate():
    if os.path.isfile(GPU_DELEGATE_SO) and os.path.getsize(GPU_DELEGATE_SO) > 500_000:
        return
    ensure_file(
        GPU_DELEGATE_SO,
        GPU_DELEGATE_URL,
        "LiteRT GPU acceleration delegate (libtflite_gpu_delegate.so)",
        min_size=500_000,
    )


def scan_mediastore(path: str):
    abs_p = os.path.abspath(path)
    if abs_p.startswith(("/storage/emulated/0", "/sdcard", "/data/media/0")):
        cmd = f'content call --uri content://media --method scan_file --arg "{abs_p}"'
        subprocess.run(["su", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ModelRunner:
    def __init__(
        self,
        backend: str,
        custom_model: str = None,
        threads: int = None,
        batch_size: int = 1,
        precision: str = None,
        device: str = "auto",
    ):
        self.backend = backend
        self.device = device
        self.threads = get_optimal_threads(threads)
        self.batch_size = max(1, batch_size)
        if precision is None:
            if backend in ("qnn-dsp", "qti-dsp"):
                precision = "int8"
            elif backend == "litert-gpu":
                precision = "fp16"
            else:
                precision = "fp32"
        self.precision = precision

        if backend in ("onnx", "onnx-cpu"):
            try:
                import onnxruntime as ort
            except ImportError:
                print("Error: onnxruntime not found. Install via 'pkg install python-onnxruntime'.", file=sys.stderr)
                sys.exit(1)

            if custom_model:
                model_path = custom_model
            elif precision == "int8":
                model_path = INT8_MODEL_PATH
                ensure_int8_model(ONNX_MODEL_PATH, INT8_MODEL_PATH)
                if not os.path.isfile(model_path):
                    model_path = ONNX_MODEL_PATH
            else:
                if precision == "fp16":
                    print("Notice: FP16 precision requested on ONNX backend; using FP32 ONNX model.")
                model_path = ONNX_MODEL_PATH
                ensure_file(model_path, ONNX_MODEL_URL, "ONNX model (Voc_FT)")

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.intra_op_num_threads = self.threads
            opts.inter_op_num_threads = 1

            available_providers = ort.get_available_providers()
            if backend == "onnx-cpu" or device == "cpu":
                providers = ["CPUExecutionProvider"]
            elif device in ("dsp", "npu"):
                providers = [("NnapiExecutionProvider", {"nnapi_accelerator_name": "qti-dsp"}), "CPUExecutionProvider"]
            elif device == "gpu":
                preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]
                providers = [p for p in preferred if p in available_providers]
                if not providers:
                    print("Warning: GPU device requested, but no GPU provider found in onnxruntime. Falling back to CPU.")
                    providers = ["CPUExecutionProvider"]
            else:
                preferred = [
                    "CUDAExecutionProvider",
                    "OpenVINOExecutionProvider",
                    "DmlExecutionProvider",
                    "CoreMLExecutionProvider",
                    "CPUExecutionProvider",
                ]
                providers = [p for p in preferred if p in available_providers] or available_providers

            # Attempt each provider in priority order with graceful fallback
            session = None
            last_err = None
            for p in providers:
                try:
                    session = ort.InferenceSession(model_path, sess_options=opts, providers=[p])
                    break
                except Exception as e:
                    last_err = e
                    continue

            if session is None:
                raise last_err

            self.session = session
            self.input_name = self.session.get_inputs()[0].name
            active_provider = self.session.get_providers()[0]
            print(f"  Engine: ONNX Runtime [{active_provider}] | Precision: {precision.upper()} | Threads: {self.threads} | Batch size: {self.batch_size}")

            def _onnx_infer(x: np.ndarray) -> np.ndarray:
                return self.session.run(None, {self.input_name: x})[0]

            self.infer_fn = _onnx_infer

        elif backend == "openvino":
            try:
                import openvino as ov
            except ImportError:
                print("Error: openvino not found. Install via 'pip install openvino'.", file=sys.stderr)
                sys.exit(1)

            if custom_model:
                model_path = custom_model
            elif precision == "int8":
                model_path = INT8_MODEL_PATH
                ensure_int8_model(ONNX_MODEL_PATH, INT8_MODEL_PATH)
                if not os.path.isfile(model_path):
                    model_path = ONNX_MODEL_PATH
            else:
                model_path = ONNX_MODEL_PATH
                ensure_file(model_path, ONNX_MODEL_URL, "ONNX model (Voc_FT)")

            core = ov.Core()
            ov_device = "CPU"
            if self.device == "gpu" and "GPU" in core.available_devices:
                ov_device = "GPU"
            elif self.device in ("cpu", "auto"):
                ov_device = "CPU"

            config = {}
            if self.threads:
                config["INFERENCE_NUM_THREADS"] = str(self.threads)
            if self.precision == "fp16":
                config["INFERENCE_PRECISION_HINT"] = "f16"
            elif self.precision == "fp32":
                config["INFERENCE_PRECISION_HINT"] = "f32"

            model = core.read_model(model_path)
            self.compiled_model = core.compile_model(model, ov_device, config)
            active_device = core.get_property(ov_device, "FULL_DEVICE_NAME") if ov_device in core.available_devices else ov_device
            print(f"  Engine: OpenVINO [{ov_device}: {active_device}] | Precision: {self.precision.upper()} | Threads: {self.threads} | Batch size: {self.batch_size}")

            def _openvino_infer(x: np.ndarray) -> np.ndarray:
                B = x.shape[0]
                if B == 1:
                    return self.compiled_model([x])[0]
                outs = [self.compiled_model([x[i : i + 1]])[0] for i in range(B)]
                return np.concatenate(outs, axis=0)

            self.infer_fn = _openvino_infer

        elif backend in ("qnn-dsp", "qti-dsp"):
            try:
                import onnxruntime as ort
            except ImportError:
                print("Error: onnxruntime not found. Install via 'pkg install python-onnxruntime'.", file=sys.stderr)
                sys.exit(1)

            if custom_model:
                model_path = custom_model
            elif self.precision == "int8":
                model_path = INT8_MODEL_PATH
                ensure_int8_model(ONNX_MODEL_PATH, INT8_MODEL_PATH)
                if not os.path.isfile(model_path):
                    model_path = ONNX_MODEL_PATH
            else:
                model_path = ONNX_MODEL_PATH
                ensure_file(model_path, ONNX_MODEL_URL, "ONNX model (Voc_FT)")

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.intra_op_num_threads = self.threads
            opts.inter_op_num_threads = 1

            providers = [("NnapiExecutionProvider", {"nnapi_accelerator_name": "qti-dsp"}), "CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            active_provider = self.session.get_providers()[0]
            print(f"  Engine: Qualcomm QNN/Hexagon DSP [{active_provider}] | Precision: {self.precision.upper()} | Threads: {self.threads} | Batch size: {self.batch_size}")

            def _qnn_infer(x: np.ndarray) -> np.ndarray:
                return self.session.run(None, {self.input_name: x})[0]

            self.infer_fn = _qnn_infer

        elif backend in ("litert-gpu", "litert-cpu"):
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                try:
                    import ai_edge_litert.interpreter as tflite
                except ImportError:
                    try:
                        import tensorflow.lite as tflite
                    except ImportError:
                        print("Error: LiteRT runtime not found. Install via 'pkg install python-tflite-runtime' (Android) or 'pip install ai-edge-litert'.", file=sys.stderr)
                        sys.exit(1)

            if custom_model:
                model_path = custom_model
            elif precision == "int8":
                model_path = TFLITE_INT8_PATH
                ensure_file(model_path, TFLITE_INT8_URL, "LiteRT INT8 model (Voc_FT)")
            elif precision == "fp16":
                model_path = TFLITE_MODEL_PATH
                ensure_file(model_path, TFLITE_MODEL_URL, "LiteRT FP16 model (Voc_FT)")
            else:
                model_path = TFLITE_FP32_PATH
                ensure_file(model_path, TFLITE_FP32_URL, "LiteRT FP32 model (Voc_FT)")

            delegates = []
            if backend == "litert-gpu":
                ensure_gpu_delegate()
                self.gpu_del = LiteRtGpuDelegate(allow_fp16=True)
                delegates.append(self.gpu_del)

            self.interp = tflite.Interpreter(
                model_path=model_path,
                num_threads=self.threads if backend == "litert-cpu" else None,
                experimental_delegates=delegates if delegates else None,
            )
            self.interp.allocate_tensors()
            in_details = self.interp.get_input_details()[0]
            self.in_idx = in_details["index"]
            self.out_idx = self.interp.get_output_details()[0]["index"]
            in_shape = list(in_details["shape"])
            # Auto-detect whether model expects NHWC [B, 3072, 256, 4] or legacy NCHW [B, 4, 3072, 256]
            self.is_nhwc = len(in_shape) == 4 and in_shape[-1] == 4

            print(f"  Engine: LiteRT [{backend}] | Precision: {precision.upper()} | Threads: {self.threads} | Batch size: {self.batch_size}")

            def _tflite_infer(x: np.ndarray) -> np.ndarray:
                if self.is_nhwc:
                    x = np.transpose(x, (0, 2, 3, 1))
                B = x.shape[0]
                if B == 1:
                    self.interp.set_tensor(self.in_idx, x)
                    self.interp.invoke()
                    res = self.interp.get_tensor(self.out_idx)
                else:
                    outs = []
                    for i in range(B):
                        self.interp.set_tensor(self.in_idx, x[i : i + 1])
                        self.interp.invoke()
                        outs.append(self.interp.get_tensor(self.out_idx))
                    res = np.concatenate(outs, axis=0)
                if self.is_nhwc:
                    res = np.transpose(res, (0, 3, 1, 2))
                return res

            self.infer_fn = _tflite_infer

    def run(self, model_in: np.ndarray) -> np.ndarray:
        return self.infer_fn(model_in)


def process_file(
    input_file: str,
    runner: ModelRunner,
    output_file: str = None,
    overlap: float = 0.25,
):
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}", file=sys.stderr)
        return False

    base, ext = os.path.splitext(input_file)
    is_video = ext.lower() in VIDEO_EXTS

    if not output_file:
        output_file = f"{base} (nomelody){ext}"

    print(f"\nProcessing: {os.path.basename(input_file)} [Backend: {runner.backend}]")

    # Decode audio stream directly through stdout pipe (avoids writing temp uncompressed WAV to disk)
    decode_cmd = [
        "ffmpeg", "-y", "-nostats", "-loglevel", "error", "-i", input_file,
        "-vn", "-ac", "2", "-ar", "44100", "-f", "s16le", "pipe:1",
    ]
    res = subprocess.run(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if res.returncode != 0 or len(res.stdout) == 0:
        print(f"Failed to decode audio track via ffmpeg: {res.stderr.decode('utf-8', errors='ignore')}", file=sys.stderr)
        return False

    sr = 44100
    audio = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    audio = audio.reshape(-1, 2).T
    total_samples = audio.shape[1]
    print(f"  Duration: {total_samples/sr:.2f}s | Sample rate: {sr}Hz stereo")

    n_fft = 6144
    hop = 1024
    dim_f = 3072
    dim_t = 256
    chunk_size = hop * (dim_t - 1)  # 261,120 samples
    overlap = max(0.05, min(0.75, overlap))
    step = max(hop, int(chunk_size * (1.0 - overlap)))

    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    win_sq = window ** 2
    pad = n_fft // 2

    # Precompute overlap-add weighting and normalizer once outside the chunk loop
    total_chunk_pad = pad + chunk_size + pad
    win_weight = np.zeros(total_chunk_pad, dtype=np.float32)
    for f_i in range(dim_t):
        s_pos = f_i * hop
        win_weight[s_pos : s_pos + n_fft] += win_sq

    inv_win_weight = np.zeros_like(win_weight)
    valid_win = win_weight > 1e-6
    inv_win_weight[valid_win] = 1.0 / win_weight[valid_win]

    chunk_win = np.hanning(chunk_size).astype(np.float32)
    # Combined inverse window + chunk blending window for the valid chunk core
    norm_chunk_win = (inv_win_weight[pad : pad + chunk_size] * chunk_win).astype(np.float32)

    padded_audio = np.pad(audio, ((0, 0), (0, chunk_size)), mode="constant")
    recon = np.zeros((2, padded_audio.shape[1]), dtype=np.float32)
    weight = np.zeros(padded_audio.shape[1], dtype=np.float32)

    starts = list(range(0, total_samples, step))
    total_chunks = len(starts)
    batch_size = max(1, runner.batch_size)

    for b_idx in range(0, total_chunks, batch_size):
        batch_starts = starts[b_idx : b_idx + batch_size]
        B = len(batch_starts)

        if B == 1:
            start = batch_starts[0]
            end = start + chunk_size
            chunk = padded_audio[:, start:end]

            c_pad = np.pad(chunk, ((0, 0), (pad, pad)), mode="reflect")
            frames = np.lib.stride_tricks.sliding_window_view(c_pad, n_fft, axis=-1)[:, ::hop, :]
            spec = fft_rfft(frames * window, n=n_fft, axis=-1)
            spec = np.transpose(spec, (0, 2, 1))[:, :dim_f, :dim_t]
            spec[:, :3, :] = 0.0

            model_in = np.empty((1, 4, dim_f, dim_t), dtype=np.float32)
            model_in[0, 0] = spec.real[0]
            model_in[0, 1] = spec.imag[0]
            model_in[0, 2] = spec.real[1]
            model_in[0, 3] = spec.imag[1]

            pred = runner.run(model_in)[0]

            pred_frames = np.zeros((2, dim_t, dim_f + 1), dtype=np.complex64)
            pred_frames[0, :, :dim_f].real = pred[0].T
            pred_frames[0, :, :dim_f].imag = pred[1].T
            pred_frames[1, :, :dim_f].real = pred[2].T
            pred_frames[1, :, :dim_f].imag = pred[3].T

            iframes = (fft_irfft(pred_frames, n=n_fft, axis=-1) * window).reshape(2, dim_t, 6, hop)

            chunk_recon = np.zeros((2, total_chunk_pad), dtype=np.float32)
            for k in range(6):
                chunk_recon[:, k * hop : (dim_t + k) * hop] += iframes[:, :, k, :].reshape(2, dim_t * hop)

            recon[:, start:end] += chunk_recon[:, pad : pad + chunk_size] * norm_chunk_win
            weight[start:end] += chunk_win
        else:
            batch_chunks = np.stack([padded_audio[:, s : s + chunk_size] for s in batch_starts], axis=0)
            c_pad = np.pad(batch_chunks, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
            frames = np.lib.stride_tricks.sliding_window_view(c_pad, n_fft, axis=-1)[:, :, ::hop, :]
            spec = fft_rfft(frames * window, n=n_fft, axis=-1)
            spec = np.transpose(spec, (0, 1, 3, 2))[:, :, :dim_f, :dim_t]
            spec[:, :, :3, :] = 0.0

            model_in = np.empty((B, 4, dim_f, dim_t), dtype=np.float32)
            model_in[:, 0] = spec.real[:, 0]
            model_in[:, 1] = spec.imag[:, 0]
            model_in[:, 2] = spec.real[:, 1]
            model_in[:, 3] = spec.imag[:, 1]

            pred = runner.run(model_in)

            pred_frames = np.zeros((B, 2, dim_t, dim_f + 1), dtype=np.complex64)
            pred_frames[:, 0, :, :dim_f].real = np.transpose(pred[:, 0], (0, 2, 1))
            pred_frames[:, 0, :, :dim_f].imag = np.transpose(pred[:, 1], (0, 2, 1))
            pred_frames[:, 1, :, :dim_f].real = np.transpose(pred[:, 2], (0, 2, 1))
            pred_frames[:, 1, :, :dim_f].imag = np.transpose(pred[:, 3], (0, 2, 1))

            iframes = (fft_irfft(pred_frames, n=n_fft, axis=-1) * window).reshape(B, 2, dim_t, 6, hop)

            for b_i, start in enumerate(batch_starts):
                end = start + chunk_size
                chunk_recon = np.zeros((2, total_chunk_pad), dtype=np.float32)
                for k in range(6):
                    chunk_recon[:, k * hop : (dim_t + k) * hop] += iframes[b_i, :, :, k, :].reshape(2, dim_t * hop)

                recon[:, start:end] += chunk_recon[:, pad : pad + chunk_size] * norm_chunk_win
                weight[start:end] += chunk_win

        print(f"\r  Removing melody: [{min(b_idx + B, total_chunks)}/{total_chunks}] chunks", end="", flush=True)

    print("")
    valid = weight > 1e-6
    recon[:, valid] /= weight[valid]
    vocal = recon[:, :total_samples] * 1.035

    peak = np.max(np.abs(vocal))
    if peak > 0.98:
        vocal = vocal * (0.98 / peak)

    out_data = (vocal.T * 32767.0).astype(np.int16)
    out_bytes = out_data.tobytes()

    # Apply high-shelf filter on INT8 to remove the 8-bit quantization noise floor
    audio_filters = []
    if getattr(runner, "precision", "fp32") == "int8":
        audio_filters.extend(["-af", "highshelf=f=9500:gain=-15"])

    # Mux back directly via ffmpeg stdin pipe (avoids writing temp stem.wav to disk)
    if is_video:
        mux_cmd = [
            "ffmpeg", "-y", "-nostats", "-loglevel", "error", "-i", input_file,
            "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", "pipe:0",
            "-map", "0:v:0", "-map", "1:a:0",
            *audio_filters,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            output_file,
        ]
    else:
        mux_cmd = [
            "ffmpeg", "-y", "-nostats", "-loglevel", "error",
            "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", "pipe:0",
            *audio_filters,
            "-c:a", "flac", output_file,
        ]

    proc = subprocess.Popen(
        mux_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, err = proc.communicate(input=out_bytes)
    if proc.returncode != 0:
        print(f"FFmpeg muxing failed: {err.decode('utf-8', errors='ignore')}", file=sys.stderr)
        return False

    scan_mediastore(output_file)
    print(f"  Saved: {output_file}")
    return True


def find_free_port(start_port: int = 8080) -> int:
    for p in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


def get_stream_urls(target: str, quality: int = 720) -> dict:
    if yt_dlp is None:
        print("Error: yt-dlp is required for live YouTube playback.", file=sys.stderr)
        print("Install it with: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)

    if not (target.startswith("http://") or target.startswith("https://")):
        query = f"ytsearch1:{target}"
    else:
        query = target

    fmt = (
        f"bestvideo[height<={quality}][vcodec^=avc1]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={quality}]+bestaudio/"
        f"best[height<={quality}]/best"
    )
    opts = {
        "format": fmt,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query, download=False)
        if "entries" in info:
            info = info["entries"][0]

    title = info.get("title", "YouTube Stream")
    duration = info.get("duration", 0)
    webpage_url = info.get("webpage_url", target)

    req = info.get("requested_formats")
    if req and len(req) >= 2:
        v_url = req[0]["url"]
        a_url = req[1]["url"]
    else:
        v_url = info.get("url")
        a_url = v_url

    return {
        "title": title,
        "duration": duration,
        "url": webpage_url,
        "video_url": v_url,
        "audio_url": a_url,
    }


class LiveFilterEngine:
    def __init__(
        self,
        runner: ModelRunner,
        video_url: str,
        audio_url: str,
        stream_format: str = "mp4",
        overlap: float = 0.25,
    ):
        self.runner = runner
        self.video_url = video_url
        self.audio_url = audio_url
        self.stream_format = stream_format
        self.overlap = max(0.05, min(0.75, overlap))

        self.sr = 44100
        self.n_fft = 6144
        self.hop = 1024
        self.dim_f = 3072
        self.dim_t = 256
        self.chunk_size = self.hop * (self.dim_t - 1)  # 261,120 samples
        self.step = max(self.hop, int(self.chunk_size * (1.0 - self.overlap)))
        self.pad = self.n_fft // 2
        self.total_chunk_pad = self.pad + self.chunk_size + self.pad

        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)).astype(np.float32)
        win_sq = self.window ** 2
        win_weight = np.zeros(self.total_chunk_pad, dtype=np.float32)
        for f_i in range(self.dim_t):
            s_pos = f_i * self.hop
            win_weight[s_pos : s_pos + self.n_fft] += win_sq
        inv_win_weight = np.zeros_like(win_weight)
        valid_win = win_weight > 1e-6
        inv_win_weight[valid_win] = 1.0 / win_weight[valid_win]

        chunk_win = np.hanning(self.chunk_size).astype(np.float32)
        self.chunk_win = chunk_win
        self.norm_chunk_win = (inv_win_weight[self.pad : self.pad + self.chunk_size] * chunk_win).astype(np.float32)

        self.filtered_pcm_queue = queue.Queue(maxsize=30)
        self.subscribers = []
        self.chunk_history = []
        self.sub_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.prebuffer_ready = threading.Event()

        self.chunks_processed = 0
        self.total_audio_seconds = 0.0
        self.total_infer_seconds = 0.0
        self.bytes_muxed = 0
        self.status = "Initializing"

    def subscribe(self) -> queue.Queue:
        with self.sub_lock:
            q = queue.Queue(maxsize=max(500, len(self.chunk_history) + 200))
            for c in self.chunk_history:
                try:
                    q.put_nowait(c)
                except Exception:
                    pass
            self.subscribers.append(q)
            return q

    def unsubscribe(self, q: queue.Queue):
        with self.sub_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _broadcast(self, data: bytes):
        if data:
            self.bytes_muxed += len(data)
            with self.sub_lock:
                self.chunk_history.append(data)
                while sum(len(c) for c in self.chunk_history) > 60 * 1024 * 1024:
                    self.chunk_history.pop(0)
                for q in list(self.subscribers):
                    try:
                        q.put_nowait(data)
                    except queue.Full:
                        try:
                            q.get_nowait()
                            q.put_nowait(data)
                        except Exception:
                            pass
        else:
            with self.sub_lock:
                for q in list(self.subscribers):
                    try:
                        q.put_nowait(b"")
                    except Exception:
                        pass

    def start(self):
        self.t_infer = threading.Thread(target=self._inference_worker, daemon=True)
        self.t_mux = threading.Thread(target=self._muxer_worker, daemon=True)
        self.t_infer.start()
        self.t_mux.start()

    def stop(self):
        self.stop_event.set()

    def _inference_worker(self):
        self.status = "Buffering audio stream"
        dec_cmd = [
            "ffmpeg", "-nostats", "-loglevel", "error", "-i", self.audio_url,
            "-vn", "-ac", "2", "-ar", "44100", "-f", "s16le", "pipe:1"
        ]
        dec_proc = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        accum_recon = np.zeros((2, self.chunk_size), dtype=np.float32)
        accum_weight = np.zeros(self.chunk_size, dtype=np.float32)
        input_buffer = np.zeros((2, 0), dtype=np.float32)
        bytes_per_sample = 4  # 2ch * 2 bytes

        self.status = "Filtering melody"

        try:
            while not self.stop_event.is_set():
                needed = self.chunk_size - input_buffer.shape[1]
                if needed > 0:
                    raw = dec_proc.stdout.read(needed * bytes_per_sample)
                    if raw:
                        new_samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, 2).T / 32768.0
                        input_buffer = np.concatenate([input_buffer, new_samples], axis=1)
                    else:
                        if input_buffer.shape[1] == 0:
                            break
                        pad_len = self.chunk_size - input_buffer.shape[1]
                        input_buffer = np.pad(input_buffer, ((0, 0), (0, pad_len)))
                        is_last = True
                else:
                    is_last = False

                if input_buffer.shape[1] < self.chunk_size:
                    pad_len = self.chunk_size - input_buffer.shape[1]
                    chunk = np.pad(input_buffer, ((0, 0), (0, pad_len)))
                    is_last = True
                else:
                    chunk = input_buffer[:, :self.chunk_size]
                    is_last = False

                t0 = time.time()
                c_pad = np.pad(chunk, ((0, 0), (self.pad, self.pad)), mode="reflect")
                frames = np.lib.stride_tricks.sliding_window_view(c_pad, self.n_fft, axis=-1)[:, ::self.hop, :]
                spec = fft_rfft(frames * self.window, n=self.n_fft, axis=-1)
                spec = np.transpose(spec, (0, 2, 1))[:, :self.dim_f, :self.dim_t]
                spec[:, :3, :] = 0.0

                model_in = np.empty((1, 4, self.dim_f, self.dim_t), dtype=np.float32)
                model_in[0, 0] = spec.real[0]
                model_in[0, 1] = spec.imag[0]
                model_in[0, 2] = spec.real[1]
                model_in[0, 3] = spec.imag[1]

                pred = self.runner.run(model_in)[0]

                pred_frames = np.zeros((2, self.dim_t, self.dim_f + 1), dtype=np.complex64)
                pred_frames[0, :, :self.dim_f].real = pred[0].T
                pred_frames[0, :, :self.dim_f].imag = pred[1].T
                pred_frames[1, :, :self.dim_f].real = pred[2].T
                pred_frames[1, :, :self.dim_f].imag = pred[3].T

                iframes = (fft_irfft(pred_frames, n=self.n_fft, axis=-1) * self.window).reshape(2, self.dim_t, 6, self.hop)
                chunk_recon = np.zeros((2, self.total_chunk_pad), dtype=np.float32)
                for k in range(6):
                    chunk_recon[:, k * self.hop : (self.dim_t + k) * self.hop] += iframes[:, :, k, :].reshape(2, self.dim_t * self.hop)

                accum_recon += chunk_recon[:, self.pad : self.pad + self.chunk_size] * self.norm_chunk_win
                accum_weight += self.chunk_win

                norm = np.maximum(accum_weight[:self.step], 1e-6)
                vocal = (accum_recon[:, :self.step] / norm) * 1.035
                peak = np.max(np.abs(vocal))
                if peak > 0.98:
                    vocal = vocal * (0.98 / peak)
                out_bytes = (vocal.T * 32767.0).astype(np.int16).tobytes()

                self.filtered_pcm_queue.put(out_bytes)

                accum_recon[:, :self.chunk_size - self.step] = accum_recon[:, self.step:]
                accum_recon[:, self.chunk_size - self.step:] = 0.0
                accum_weight[:self.chunk_size - self.step] = accum_weight[self.step:]
                accum_weight[self.chunk_size - self.step:] = 0.0
                input_buffer = input_buffer[:, self.step:]

                t_infer = time.time() - t0
                self.chunks_processed += 1
                self.total_audio_seconds += self.step / self.sr
                self.total_infer_seconds += t_infer

                if self.chunks_processed >= 1 and not self.prebuffer_ready.is_set():
                    self.prebuffer_ready.set()

                if is_last:
                    break

            if np.any(accum_weight[:self.chunk_size - self.step] > 1e-6):
                rem_norm = np.maximum(accum_weight[:self.chunk_size - self.step], 1e-6)
                rem_vocal = (accum_recon[:, :self.chunk_size - self.step] / rem_norm) * 1.035
                peak = np.max(np.abs(rem_vocal))
                if peak > 0.98:
                    rem_vocal = rem_vocal * (0.98 / peak)
                rem_bytes = (rem_vocal.T * 32767.0).astype(np.int16).tobytes()
                self.filtered_pcm_queue.put(rem_bytes)

        except Exception as e:
            print(f"Inference error: {e}", file=sys.stderr)
        finally:
            self.filtered_pcm_queue.put(None)
            try:
                dec_proc.terminate()
            except Exception:
                pass

    def _muxer_worker(self):
        self.prebuffer_ready.wait(timeout=15)

        audio_filters = []
        if getattr(self.runner, "precision", "fp32") == "int8":
            audio_filters.extend(["-af", "highshelf=f=9500:gain=-15"])

        if self.stream_format == "mkv":
            fmt_flags = ["-f", "matroska", "-live", "1"]
        else:
            fmt_flags = ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof"]

        mux_cmd = [
            "ffmpeg", "-nostats", "-loglevel", "warning",
            "-i", self.video_url,
            "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", "pipe:0",
            "-map", "0:v:0", "-map", "1:a:0",
            *audio_filters,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-max_muxing_queue_size", "4096",
            *fmt_flags,
            "pipe:1"
        ]

        mux_proc = subprocess.Popen(
            mux_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        def _feeder():
            try:
                while not self.stop_event.is_set():
                    pcm = self.filtered_pcm_queue.get()
                    if pcm is None:
                        break
                    mux_proc.stdin.write(pcm)
                    mux_proc.stdin.flush()
            except Exception:
                pass
            finally:
                try:
                    mux_proc.stdin.close()
                except Exception:
                    pass

        t_feed = threading.Thread(target=_feeder, daemon=True)
        t_feed.start()

        self.status = "Streaming live"

        try:
            while not self.stop_event.is_set():
                chunk = mux_proc.stdout.read(32768)
                if not chunk:
                    break
                self._broadcast(chunk)
        except Exception:
            pass
        finally:
            self.status = "Ended"
            self._broadcast(b"")
            try:
                mux_proc.terminate()
            except Exception:
                pass


class LiveStreamHandler(http.server.BaseHTTPRequestHandler):
    engine = None
    title = "nomelody live"

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            ctype = "video/mp4" if self.engine.stream_format == "mp4" else "video/x-matroska"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "none")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = self.engine.subscribe()
            try:
                while not self.engine.stop_event.is_set():
                    try:
                        chunk = q.get(timeout=2.0)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except queue.Empty:
                        if self.engine.status == "Ended":
                            break
            except Exception:
                pass
            finally:
                self.engine.unsubscribe(q)

        elif self.path == "/stats":
            rtf = self.engine.total_infer_seconds / max(0.001, self.engine.total_audio_seconds)
            data = json.dumps({
                "status": self.engine.status,
                "chunks": self.engine.chunks_processed,
                "audio_seconds": round(self.engine.total_audio_seconds, 1),
                "rtf": round(rtf, 2),
                "speed": f"{1.0/max(0.01, rtf):.2f}x" if rtf > 0 else "--",
                "bytes": self.engine.bytes_muxed,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            stream_src = f"/stream.{self.engine.stream_format}"
            html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.title} - nomelody live</title>
<style>
  body {{ background: #0f1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 90vh; }}
  .player-box {{ width: 100%; max-width: 800px; background: #161b22; border: 1px solid #30363d; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
  video {{ width: 100%; aspect-ratio: 16/9; background: #000; display: block; }}
  .content {{ padding: 16px; }}
  h2 {{ margin: 0 0 8px 0; font-size: 1.15rem; font-weight: 600; }}
  .tags {{ display: flex; gap: 8px; margin-bottom: 12px; }}
  .tag {{ font-size: 0.75rem; padding: 3px 8px; border-radius: 6px; font-weight: 600; text-transform: uppercase; }}
  .tag-live {{ background: #238636; color: #fff; }}
  .tag-engine {{ background: #1f6feb; color: #fff; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; background: #0d1117; padding: 12px; border-radius: 8px; font-size: 0.85rem; border: 1px solid #21262d; }}
  .stats-item {{ display: flex; flex-direction: column; }}
  .stats-label {{ color: #8b949e; font-size: 0.75rem; }}
  .stats-val {{ font-weight: 600; margin-top: 2px; color: #58a6ff; }}
</style>
</head>
<body>
<div class="player-box">
  <video id="video" controls autoplay playsinline></video>
  <div class="content">
    <h2>{self.title}</h2>
    <div class="tags">
      <span class="tag tag-live" id="status-tag">Live Filter</span>
      <span class="tag tag-engine">{self.engine.runner.backend.upper()} ({self.engine.runner.precision.upper()})</span>
    </div>
    <div class="stats-grid">
      <div class="stats-item"><span class="stats-label">Real-Time Speed</span><span class="stats-val" id="val-speed">--</span></div>
      <div class="stats-item"><span class="stats-label">Filtered Audio</span><span class="stats-val" id="val-audio">0.0s</span></div>
      <div class="stats-item"><span class="stats-label">Streamed Data</span><span class="stats-val" id="val-bytes">0 KB</span></div>
    </div>
  </div>
</div>
<script>
  const video = document.getElementById('video');
  const streamSrc = '{stream_src}';

  async function startPlayback() {{
    if (window.MediaSource && streamSrc.endsWith('.mp4')) {{
      const ms = new MediaSource();
      video.src = URL.createObjectURL(ms);
      ms.addEventListener('sourceopen', async () => {{
        try {{
          const candidates = [
            'video/mp4; codecs="avc1.4D401F, mp4a.40.2"',
            'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
            'video/mp4; codecs="avc1, mp4a.40.2"',
            'video/mp4'
          ];
          const chosen = candidates.find(c => MediaSource.isTypeSupported(c)) || 'video/mp4';
          const sb = ms.addSourceBuffer(chosen);
          sb.mode = 'sequence';

          const res = await fetch(streamSrc);
          const reader = res.body.getReader();
          const queue = [];
          let updating = false;

          function pushNext() {{
            if (!updating && queue.length > 0 && !sb.updating) {{
              updating = true;
              sb.appendBuffer(queue.shift());
            }}
          }}

          sb.addEventListener('updateend', () => {{
            updating = false;
            pushNext();
            if (video.paused && video.buffered.length > 0) {{
              video.play().catch(() => {{
                video.muted = true;
                video.play().catch(() => {{}});
              }});
            }}
          }});

          while (true) {{
            const {{ done, value }} = await reader.read();
            if (done) {{
              if (ms.readyState === 'open') ms.endOfStream();
              break;
            }}
            queue.push(value);
            pushNext();
          }}
        }} catch(err) {{
          console.warn('MSE streaming error, falling back to direct video.src:', err);
          video.src = streamSrc;
          video.play().catch(() => {{ video.muted = true; video.play().catch(() => {{}}); }});
        }}
      }});
    }} else {{
      video.src = streamSrc;
      video.play().catch(() => {{ video.muted = true; video.play().catch(() => {{}}); }});
    }}
  }}

  startPlayback();

  setInterval(async () => {{
    try {{
      const res = await fetch('/stats');
      const data = await res.json();
      document.getElementById('status-tag').textContent = data.status;
      document.getElementById('val-speed').textContent = data.speed;
      document.getElementById('val-audio').textContent = data.audio_seconds + 's (' + data.chunks + ' chunks)';
      document.getElementById('val-bytes').textContent = Math.round(data.bytes / 1024) + ' KB';
    }} catch(e) {{}}
  }}, 1000);
</script>
</body>
</html>""".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)


def play_live(
    target: str,
    runner: ModelRunner,
    port: int = 8080,
    player: str = "intent",
    stream_format: str = "mp4",
    quality: int = 720,
    overlap: float = 0.25,
):
    print(f"\nResolving stream for: {target}...")
    stream_info = get_stream_urls(target, quality=quality)
    title = stream_info["title"]
    duration = stream_info["duration"]
    dur_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}" if duration else "Live / Unknown"

    port = find_free_port(port)
    stream_file = f"stream.{stream_format}"
    stream_url = f"http://127.0.0.1:{port}/{stream_file}"
    web_url = f"http://127.0.0.1:{port}/"
    mime_type = "video/mp4" if stream_format == "mp4" else "video/x-matroska"

    engine = LiveFilterEngine(
        runner=runner,
        video_url=stream_info["video_url"],
        audio_url=stream_info["audio_url"],
        stream_format=stream_format,
        overlap=overlap,
    )
    LiveStreamHandler.engine = engine
    LiveStreamHandler.title = title

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), LiveStreamHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    engine.start()

    print("\n" + "=" * 60)
    print(f"  Live Video:  {title}")
    print(f"  Duration:    {dur_str}")
    print(f"  Engine:      {runner.backend} [{runner.precision.upper()}]")
    print(f"  Stream URL:  {stream_url}")
    print(f"  Web Player:  {web_url}")
    print("=" * 60)

    # Pre-buffer first chunk before launching player
    print("Pre-buffering live audio filter...", end="", flush=True)
    engine.prebuffer_ready.wait(timeout=15)
    print(" Ready!")

    # Launch video player on device
    if player == "browser":
        print("Launching browser web player...")
        if shutil.which("termux-open-url"):
            subprocess.run(["termux-open-url", web_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import webbrowser
            webbrowser.open(web_url)
    elif player == "intent":
        if shutil.which("am"):
            print("Launching default video player...")
            subprocess.run(
                ["am", "start", "-a", "android.intent.action.VIEW", "-d", stream_url, "-t", mime_type],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print("Android 'am' command not found, falling back to browser...")
            import webbrowser
            webbrowser.open(web_url)
    elif player == "mx":
        if shutil.which("am"):
            print("Launching MX Player...")
            subprocess.run(
                ["am", "start", "-n", "com.mxtech.videoplayer.ad/.ActivityScreen", "-d", stream_url, "-t", mime_type],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print("Android 'am' command not found.")

    print("\nPress Ctrl+C to stop streaming.\n")
    try:
        while not engine.stop_event.is_set():
            time.sleep(1.0)
            rtf = engine.total_infer_seconds / max(0.001, engine.total_audio_seconds)
            speed_str = f"{1.0/max(0.01, rtf):.2f}x" if rtf > 0 else "--"
            print(
                f"\r[Live] Chunks: {engine.chunks_processed} | "
                f"Audio: {engine.total_audio_seconds:.1f}s | "
                f"Speed: {speed_str} ({rtf:.2f} RTF) | "
                f"Streamed: {engine.bytes_muxed/1024:.1f} KB | "
                f"Status: {engine.status}   ",
                end="",
                flush=True,
            )
            if engine.status == "Ended":
                break
    except KeyboardInterrupt:
        print("\nStopping live playback...")
    finally:
        engine.stop()
        server.shutdown()
        print("\nLive playback closed.")


def main():
    parser = argparse.ArgumentParser(
        description="nomelody: Local AI vocal isolation for videos and audio (removes background music/melody)"
    )
    parser.add_argument("inputs", nargs="*", help="Input video or audio file(s), or 'live <url/query>'")
    parser.add_argument("-o", "--output", help="Explicit output file path (single file only)")
    parser.add_argument(
        "--live",
        metavar="TARGET",
        default=None,
        help="Stream and filter YouTube video or search query live on-the-fly",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP streaming port for live mode (default: 8080)",
    )
    parser.add_argument(
        "--player",
        choices=["browser", "intent", "mx", "none"],
        default="browser",
        help="Player launcher for live mode: 'browser' (default web player), 'intent' (Android system player), 'mx' (MX Player), or 'none'",
    )
    parser.add_argument(
        "--format",
        dest="stream_format",
        choices=["mp4", "mkv"],
        default="mp4",
        help="Container format for live stream: 'mp4' (default, widest compatibility) or 'mkv'",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=720,
        help="Maximum video resolution for live streaming (default: 720)",
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "onnx-cpu", "openvino", "litert-gpu", "litert-cpu", "qnn-dsp", "qti-dsp"],
        default=None,
        help="Inference engine: 'onnx' (default, multi-provider GPU/CPU), 'onnx-cpu' (forced CPU ONNX Runtime), 'qnn-dsp' (Qualcomm Hexagon DSP/NPU via QNN NNAPI), 'openvino' (Intel OpenVINO runtime), 'litert-gpu' (default for live mode, OpenCL GPU), 'litert-cpu' (LiteRT XNNPACK CPU)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "dsp", "npu"],
        default="auto",
        help="Execution target device for ONNX/OpenVINO: 'auto' (default), 'cpu' (forces CPU inference), 'gpu', or 'dsp'/'npu' (Qualcomm Hexagon DSP)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default=None,
        help="Inference precision: 'fp32' (full precision), 'fp16' (fast static FP16, default for litert-gpu), or 'int8' (static INT8, fastest CPU throughput)",
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
        default=None,
        help="Inference worker threads (default: CPU core count up to 8)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: 1)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Chunk overlap ratio between 0.05 and 0.75 (default: 0.25, UVR standard)",
    )
    args = parser.parse_args()

    # Determine if live mode is requested
    live_target = args.live
    inputs = list(args.inputs)
    if not live_target and inputs:
        if inputs[0].lower() == "live":
            if len(inputs) > 1:
                live_target = " ".join(inputs[1:])
            else:
                print("Error: 'live' requires a YouTube URL or query (e.g. 'nomelody live <url>').", file=sys.stderr)
                sys.exit(1)
        elif inputs[0].startswith(("http://", "https://")) or "youtu.be" in inputs[0] or "youtube.com" in inputs[0]:
            live_target = inputs[0]

    # Select backend defaults
    backend = args.backend
    if backend is None:
        is_android = os.path.exists("/system") or os.environ.get("TERMUX_VERSION") or shutil.which("termux-info")
        backend = "litert-gpu" if (live_target and is_android) else "onnx"

    runner = ModelRunner(
        backend=backend,
        custom_model=args.model,
        threads=args.threads,
        batch_size=args.batch_size,
        precision=args.precision,
        device=args.device,
    )

    if live_target:
        play_live(
            target=live_target,
            runner=runner,
            port=args.port,
            player=args.player,
            stream_format=args.stream_format,
            quality=args.quality,
            overlap=args.overlap,
        )
    elif inputs:
        for inp in inputs:
            process_file(
                inp,
                runner=runner,
                output_file=args.output,
                overlap=args.overlap,
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
