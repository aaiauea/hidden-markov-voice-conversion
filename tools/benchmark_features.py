#!/usr/bin/env python3
"""Performance benchmark for HMVC feature extraction (NumPy only).

Measures, on deterministic synthetic speech-like signals:

* full-extraction wall time and real-time factor per sample rate;
* per-stage cost (F0, CheapTrick + MCEP, band aperiodicity, STFT descriptors);
* scaling with duration (time should grow linearly) and peak traced memory
  (bounded by ``block_frames``, not by the utterance length).

Usage::

    python tools/benchmark_features.py                 # print JSON
    python tools/benchmark_features.py --write         # docs/feature-extraction/benchmark.json
    python tools/benchmark_features.py --quick         # smaller run (CI smoke)

Timings are single-process best-of-N; set ``OPENBLAS_NUM_THREADS=1`` /
``OMP_NUM_THREADS=1`` for reproducible single-threaded numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

from hmvc.features import AnalysisConfig, FeatureExtractor  # noqa: E402
from hmvc.features import aperiodicity as A  # noqa: E402
from hmvc.features import envelope as E  # noqa: E402
from hmvc.features.cepstrum import sp2mc  # noqa: E402
from hmvc.features.descriptors import spectral_descriptors  # noqa: E402
from hmvc.features.f0 import estimate_f0  # noqa: E402
from hmvc.features.framing import blocks, frame_signal, get_window, power_spectrum  # noqa: E402
from tests._support import feature_signals as S  # noqa: E402

OUT = PROJECT / "docs" / "feature-extraction" / "benchmark.json"


def utterance(fs: int, seconds: float) -> np.ndarray:
    """Concatenated mixed utterances (silence, vowels, fricative) of ``seconds``."""
    parts, total, seed = [], 0, 0
    while total < seconds * fs:
        x = S.mixed_utterance(fs, seed=seed)["x"]
        parts.append(x)
        total += x.size
        seed += 1
    return np.concatenate(parts)[:int(seconds * fs)]


def best_of(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def stages(x: np.ndarray, fs: int, repeat: int) -> dict:
    cfg = AnalysisConfig()
    g = cfg.resolve(fs)
    hop = g.hop_length
    pitch = estimate_f0(x, fs, hop)
    edges = A.band_edges(cfg.n_aperiodicity_bands, fs)
    window = get_window("hann", g.frame_length)
    n_frames = pitch.f0.size

    def envelope_mcep():
        for start, stop in blocks(n_frames, cfg.block_frames):
            env = E.cheaptrick_frames(x, fs, pitch.f0[start:stop], start, hop, g.fft_size)
            sp2mc(env, g.mcep_order, g.mcep_alpha)

    def descriptors():
        prev = None
        for start, stop in blocks(n_frames, cfg.block_frames):
            frames = frame_signal(x, g.frame_length, hop, stop - start, start)
            _, prev = spectral_descriptors(power_spectrum(frames, window, g.fft_size), fs,
                                           previous_magnitude=prev)

    return {
        "f0_voicing_s": round(best_of(lambda: estimate_f0(x, fs, hop), repeat), 4),
        "cheaptrick_mcep_s": round(best_of(envelope_mcep, repeat), 4),
        "band_aperiodicity_s": round(best_of(
            lambda: A.band_aperiodicity(x, fs, pitch.f0, hop, edges), repeat), 4),
        "stft_descriptors_s": round(best_of(descriptors, repeat), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repeat = 1 if args.quick else 3
    seconds = 2.0 if args.quick else 10.0
    rates = (16000, 48000) if args.quick else (8000, 16000, 22050, 44100, 48000)
    durations = (1.0, 4.0) if args.quick else (1.0, 10.0, 60.0)

    result = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/benchmark_features.py" + (" --quick" if args.quick else ""),
        "python": platform.python_version(), "numpy": np.__version__,
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "signal": "concatenated synthetic mixed utterances (silence, vowels, fricative)",
        "by_sample_rate": {}, "duration_scaling_16k": {},
    }
    for fs in rates:
        x = utterance(fs, seconds)
        ext = FeatureExtractor()
        ext.extract(x[:fs], fs)                               # warm caches
        total = best_of(lambda: ext.extract(x, fs), repeat)
        g = AnalysisConfig().resolve(fs)
        result["by_sample_rate"][str(fs)] = {
            "audio_s": seconds, "fft_size": g.fft_size, "mcep_order": g.mcep_order,
            "frames": 1 + x.size // g.hop_length,
            "extract_s": round(total, 4), "x_realtime": round(seconds / total, 1),
            "ms_per_frame": round(1000 * total / (1 + x.size // g.hop_length), 4),
            "stages": stages(x, fs, repeat),
        }
        print(f"{fs} Hz: {total:.3f} s for {seconds:g} s audio "
              f"({seconds / total:.1f}x real time)", file=sys.stderr)

    for seconds_i in durations:
        x = utterance(16000, seconds_i)
        ext = FeatureExtractor()
        total = best_of(lambda: ext.extract(x, 16000), 1 if seconds_i > 10 else repeat)
        tracemalloc.start()
        ext.extract(x, 16000)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["duration_scaling_16k"][str(seconds_i)] = {
            "extract_s": round(total, 4), "x_realtime": round(seconds_i / total, 1),
            "peak_traced_mib": round(peak / 2 ** 20, 1),
            "input_mib": round(x.nbytes / 2 ** 20, 2)}
        print(f"16 kHz {seconds_i:g} s: {total:.3f} s, peak {peak / 2 ** 20:.1f} MiB",
              file=sys.stderr)

    text = json.dumps(result, indent=1)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print("wrote", OUT, file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
