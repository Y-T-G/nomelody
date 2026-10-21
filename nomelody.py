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
import warnings
import argparse
import subprocess
import tempfile
import ctypes
import numpy as np

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
        precision: str = "fp32",
    ):
        self.backend = backend
        self.threads = get_optimal_threads(threads)
        self.batch_size = max(1, batch_size)
        self.precision = precision

        if backend == "onnx":
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
                model_path = ONNX_MODEL_PATH
                ensure_file(model_path, ONNX_MODEL_URL, "ONNX model (Voc_FT)")

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.intra_op_num_threads = self.threads
            opts.inter_op_num_threads = 1

            available_providers = ort.get_available_providers()
            preferred = [
                "CUDAExecutionProvider",
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
            elif precision == "fp32" or backend == "litert-cpu":
                model_path = TFLITE_FP32_PATH
                ensure_file(model_path, TFLITE_FP32_URL, "LiteRT FP32 model (Voc_FT)")
            else:
                model_path = TFLITE_MODEL_PATH
                ensure_file(model_path, TFLITE_MODEL_URL, "LiteRT FP16 model (Voc_FT)")

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
        help="Inference engine: 'onnx' (default, multi-provider CPU/GPU), 'litert-gpu' (LiteRT OpenCL GPU delegate), 'litert-cpu' (LiteRT XNNPACK CPU)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "int8"],
        default="fp32",
        help="Inference precision: 'fp32' (default, full precision) or 'int8' (calibrated static INT8, ~2.8x faster on CPU, 75%% smaller model)",
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

    runner = ModelRunner(
        backend=args.backend,
        custom_model=args.model,
        threads=args.threads,
        batch_size=args.batch_size,
        precision=args.precision,
    )

    for inp in args.inputs:
        process_file(
            inp,
            runner=runner,
            output_file=args.output,
            overlap=args.overlap,
        )


if __name__ == "__main__":
    main()
