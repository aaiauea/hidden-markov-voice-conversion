#!/usr/bin/env python3
"""Generate reference outputs from ESTABLISHED implementations for the feature tests.

Maintainer-only and opt-in.  This script must run in a *separate* environment
that has ``pyworld`` (WORLD), ``pysptk`` (SPTK) and ``librosa`` installed; HMVC
itself never depends on them, and the ordinary test-suite blocks their import.
It never imports ``hmvc`` (asserted), so the stored oracles are independent of
the code they check.  Inputs come from ``tests/_support/feature_signals.py``
and are stored verbatim (float32-exact) in the fixture.

    python tools/generate_feature_reference.py --write

Outputs ``tests/data/feature-reference/reference.npz`` and ``reference.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tests" / "_support"))
import feature_signals as signals  # noqa: E402

OUT = PROJECT / "tests" / "data" / "feature-reference"
HOP_MS = 5.0
F0_FLOOR, F0_CEIL = 60.0, 800.0
MCEP_ORDER = {16000: 24, 22050: 34}
ALPHA_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)


def fft_size_for(fs: int) -> int:
    # WORLD's GetFFTSizeForCheapTrick(fs, f0_floor)
    return int(2 ** (1 + int(np.log2(3.0 * fs / F0_FLOOR + 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="overwrite the fixture")
    args = parser.parse_args()

    import librosa
    import pysptk
    import pyworld

    arrays = {}
    cases = {}
    for name, (x, f0_samples, fs) in signals.reference_signals().items():
        hop = int(round(fs * HOP_MS / 1000.0))
        n_fft = fft_size_for(fs)
        frame_length = int(round(fs * 0.025))
        order = MCEP_ORDER[fs]
        alpha = float(pysptk.util.mcepalpha(fs))
        true_f0 = signals.frame_f0(f0_samples, hop)
        f0_h, tpos = pyworld.harvest(x, fs, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL,
                                     frame_period=HOP_MS)
        f0_d, tpos_d = pyworld.dio(x, fs, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL,
                                   frame_period=HOP_MS)
        f0_d = pyworld.stonemask(x, f0_d, tpos_d, fs)
        # Envelope oracle: WORLD CheapTrick driven by the *true* F0, which
        # isolates the envelope algorithm from F0-estimator differences.
        sp = pyworld.cheaptrick(x, true_f0, tpos, fs, q1=-0.15, f0_floor=F0_FLOOR,
                                fft_size=n_fft)
        mc = pysptk.sp2mc(sp, order, alpha)
        sp_back = pysptk.mc2sp(mc, alpha, n_fft)
        assert len(true_f0) == len(f0_h) == sp.shape[0]

        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=frame_length,
                                   window="hann", center=True, pad_mode="constant"))
        rms = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop,
                                  center=True, pad_mode="constant")[0]
        centroid = librosa.feature.spectral_centroid(S=stft, sr=fs, n_fft=n_fft)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=fs, n_fft=n_fft)[0]
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=fs, n_fft=n_fft,
                                                   roll_percent=0.85)[0]
        flatness = librosa.feature.spectral_flatness(S=stft)[0]
        yin = librosa.yin(x, fmin=F0_FLOOR, fmax=F0_CEIL, sr=fs, frame_length=n_fft,
                          hop_length=hop, center=True, trough_threshold=0.1)
        zcr = librosa.feature.zero_crossing_rate(x, frame_length=frame_length,
                                                 hop_length=hop, center=True)[0]

        arrays.update({
            f"{name}/x": x.astype(np.float32), f"{name}/true_f0": true_f0,
            f"{name}/harvest_f0": f0_h, f"{name}/dio_f0": f0_d,
            f"{name}/cheaptrick_log_sp": np.log(sp).astype(np.float32),
            f"{name}/sptk_mcep": mc,
            f"{name}/librosa_rms": rms, f"{name}/librosa_centroid": centroid,
            f"{name}/librosa_bandwidth": bandwidth, f"{name}/librosa_rolloff": rolloff,
            f"{name}/librosa_flatness": flatness, f"{name}/librosa_yin": yin,
            f"{name}/librosa_zcr": zcr,
        })
        if name == "vowel_low":
            arrays[f"{name}/sptk_mc2sp_log"] = np.log(sp_back).astype(np.float32)
            arrays[f"{name}/librosa_log_stft"] = np.log(np.maximum(stft.T, 1e-12)).astype(np.float32)
        cases[name] = {"sample_rate": fs, "hop_length": hop, "fft_size": n_fft,
                       "frame_length": frame_length, "mcep_order": order, "mcep_alpha": alpha,
                       "frames": int(len(true_f0)), "samples": int(x.size)}

    # Frequency warping and all-pass constants.
    rng = np.random.default_rng(2026)
    cep = rng.standard_normal((6, 257)) * (0.8 ** np.arange(257))
    arrays["freqt/input"] = cep
    for a in (0.0, 0.31, 0.42, 0.55, -0.42):
        arrays[f"freqt/alpha_{a:+.2f}"] = np.stack([pysptk.freqt(c, 30, a) for c in cep])
    arrays["mcepalpha/rates"] = np.array(ALPHA_RATES, dtype=np.int64)
    arrays["mcepalpha/values"] = np.array([pysptk.util.mcepalpha(r) for r in ALPHA_RATES])

    loaded = sorted(m for m in sys.modules if m == "hmvc" or m.startswith("hmvc."))
    assert not loaded, f"reference generation must not import HMVC: {loaded}"
    support = PROJECT / "tests" / "_support" / "feature_signals.py"
    provenance = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/generate_feature_reference.py",
        "signal_recipe": {"path": "tests/_support/feature_signals.py",
                          "sha256": hashlib.sha256(support.read_bytes()).hexdigest()},
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("numpy", "pyworld", "pysptk", "librosa", "scipy")},
        "settings": {"hop_ms": HOP_MS, "f0_floor": F0_FLOOR, "f0_ceil": F0_CEIL,
                     "cheaptrick_q1": -0.15, "stft_window": "hann (periodic)",
                     "stft_center": True, "stft_pad_mode": "constant"},
        "cases": cases,
        "hmvc_imported": False,
        "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in sorted(arrays.items())},
    }
    if not args.write:
        print(json.dumps(provenance["cases"], indent=1))
        print("dry run; pass --write to update", OUT)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "reference.npz", **arrays)
    provenance["npz_sha256"] = hashlib.sha256((OUT / "reference.npz").read_bytes()).hexdigest()
    (OUT / "reference.json").write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")
    print("wrote", OUT, f"({(OUT / 'reference.npz').stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
