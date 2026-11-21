"""Synthetic source/target signals for the conversion-port reference fixtures.

The recipe is NumPy-only, independent of hmvc, and deterministic (seeded).

The source and target are speech-like mixed utterances: silence / voiced
vowels with gliding F0+vibrato / fricative noise / silence.  They differ
deliberately (different F0, formants and structure) so the retrieval
database, the grain synth and the GMM vocoder all have real work to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling recipe module

import feature_signals as S  # noqa: E402

SR = 22050


def _mixed_utterance(seed, f0_lo, f0_hi, f0_high, fs=SR, duration_total=1.2,
                     quiet_intro_frac=0.18, fricative_hz=3200.0):
    """silence | /a/ glide | fricative | silence | /i/ glide | silence."""
    parts, truth = [], []

    def add(x, f0):
        parts.append(x)
        truth.append(f0)

    def gap(d):
        return np.zeros(int(d * fs)), np.zeros(int(d * fs))

    add(*gap(0.10 * quiet_intro_frac))
    v1 = S.vowel(fs, duration_total * 0.32, f0_lo, f0_hi, 5.0, 22.0,
                 S.FORMANTS_A, -42.0, 0.22, seed)
    add(v1["x"], v1["f0"])
    n_fric = int(duration_total * 0.13 * fs)
    add(S.highpass_noise(n_fric, fs, fricative_hz, 0.05, seed + 1), np.zeros(n_fric))
    add(*gap(0.05 * duration_total))
    v2 = S.vowel(fs, duration_total * 0.30, f0_high, f0_high * 0.82, 6.0, 28.0,
                 S.FORMANTS_I, -42.0, 0.18, seed + 2)
    add(v2["x"], v2["f0"])
    add(*gap(0.07 * duration_total))
    return np.concatenate(parts), np.concatenate(truth)


def source():
    """Source voice (expected to be converted)."""
    return _mixed_utterance(seed=101, f0_lo=115.0, f0_hi=150.0, f0_high=235.0, fs=SR)


def target():
    """Target voice (what the source is converted towards)."""
    return _mixed_utterance(seed=202, f0_lo=140.0, f0_hi=175.0, f0_high=295.0,
                            fs=SR, duration_total=1.5)


def reference_signals():
    """``{name: (x float64, f0 float64, fs)}`` — the fixed comparison set."""
    src, src_f0 = source()
    tgt, tgt_f0 = target()
    return {
        "source": (np.ascontiguousarray(src.astype(np.float32).astype(np.float64)), src_f0, SR),
        "target": (np.ascontiguousarray(tgt.astype(np.float32).astype(np.float64)), tgt_f0, SR),
    }
