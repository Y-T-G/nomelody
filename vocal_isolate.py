#!/usr/bin/env python3
"""
vocal-isolate / nomelody: On-device neural vocal and instrumental separation for video and audio.
Powered by ONNX Runtime and the UVR MDX-Net architecture.

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
import numpy as np

# Suppress onnxruntime platform warning on Android/Termux
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime.*")
try:
    import onnxruntime as ort
except ImportError:
    print(
        "Error: onnxruntime is required. Install via 'pkg install python-onnxruntime' "
        "(Termux) or 'pip install onnxruntime' (Linux/macOS/Windows).",
        file=sys.stderr,
    )
    sys.exit(1)

MODEL_DIR = os.path.expanduser("~/.local/share/models")
MODEL_NAME = "UVR-MDX-NET-Voc_FT.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
MODEL_URL = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{MODEL_NAME}"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".ts", ".m4v"}


def ensure_model_exists(custom_path: str = None) -> str:
    path = custom_path or MODEL_PATH
    if os.path.isfile(path) and os.path.getsize(path) > 10_000_000:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"Downloading MDX-Net vocal model ({MODEL_NAME}) to {path}...")
    subprocess.run(["curl", "-L", "-o", path, MODEL_URL], check=True)
    return path


def scan_mediastore(path: str):
    abs_p = os.path.abspath(path)
    if abs_p.startswith(("/storage/emulated/0", "/sdcard", "/data/media/0")):
        cmd = f'content call --uri content://media --method scan_file --arg "{abs_p}"'
        subprocess.run(["su", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_file(
    input_file: str,
    mode: str = "vocals",
    output_file: str = None,
    session: ort.InferenceSession = None,
):
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}", file=sys.stderr)
        return False

    base, ext = os.path.splitext(input_file)
    is_video = ext.lower() in VIDEO_EXTS

    suffix = " (vocals)" if mode == "vocals" else " (instrumental)"
    if not output_file:
        output_file = f"{base}{suffix}{ext}"

    print(f"\nProcessing: {os.path.basename(input_file)}")
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

        # MDX-Net STFT specifications
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

            # Reflection pad matching STFT center=True
            c_pad = np.pad(chunk, ((0, 0), (pad, pad)), mode="reflect")
            frames = np.lib.stride_tricks.sliding_window_view(c_pad, n_fft, axis=-1)[:, ::hop, :]
            spec = np.fft.rfft(frames * window, n=n_fft, axis=-1)
            spec = np.transpose(spec, (0, 2, 1))

            spec = spec[:, :dim_f, :dim_t]
            spec[:, :3, :] = 0.0

            model_in = np.stack(
                [spec.real[0], spec.imag[0], spec.real[1], spec.imag[1]], axis=0
            )[None, ...].astype(np.float32)
            pred = session.run(None, {"input": model_in})[0][0]

            pred_c0 = pred[0] + 1j * pred[1]
            pred_c1 = pred[2] + 1j * pred[3]
            pred_spec = np.stack([pred_c0, pred_c1], axis=0)

            # Pad back to n_fft // 2 + 1 frequency bins
            pred_spec_padded = np.pad(pred_spec, ((0, 0), (0, 1), (0, 0)), mode="constant")
            pred_frames = np.transpose(pred_spec_padded, (0, 2, 1))

            # Inverse STFT
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
            print(f"\r  Separating stems: [{idx + 1}/{total_chunks}] chunks", end="", flush=True)

        print("")
        valid = weight > 1e-6
        recon[:, valid] /= weight[valid]
        vocal = recon[:, :total_samples] * 1.035

        if mode == "instrumental":
            final_audio = audio - vocal
        else:
            final_audio = vocal

        peak = np.max(np.abs(final_audio))
        if peak > 0.98:
            final_audio = final_audio * (0.98 / peak)

        out_data = (final_audio.T * 32767.0).astype(np.int16)
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
        description="Local AI stem separator (vocal and instrumental isolation for videos and audio)"
    )
    parser.add_argument("inputs", nargs="+", help="Input video or audio file(s)")
    parser.add_argument("-o", "--output", help="Explicit output file path (single file only)")
    parser.add_argument(
        "--inst",
        "--instrumental",
        dest="mode",
        action="store_const",
        const="instrumental",
        default="vocals",
        help="Keep instrumental instead of vocals",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both vocal and instrumental versions",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to custom MDX-Net ONNX model file",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Inference worker threads (default: 4)",
    )
    args = parser.parse_args()

    model_file = ensure_model_exists(args.model)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = args.threads
    opts.intra_op_num_threads = args.threads
    session = ort.InferenceSession(model_file, sess_options=opts)

    for inp in args.inputs:
        if args.both:
            process_file(inp, mode="vocals", session=session)
            process_file(inp, mode="instrumental", session=session)
        else:
            process_file(inp, mode=args.mode, output_file=args.output, session=session)


if __name__ == "__main__":
    main()
