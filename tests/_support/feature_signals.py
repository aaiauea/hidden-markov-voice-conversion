"""Deterministic synthetic signals with known ground truth (NumPy only).

Shared by the feature tests, ``tools/generate_feature_reference.py`` (which
must not import ``hmvc``) and ``tools/evaluate_features.py``.  Voiced sounds
are built by additive synthesis, so both the F0 track and the spectral
envelope are known exactly:

    x(n) = sum_h  A(h * f0(n)) * sin(h * phi(n)),   phi = 2 pi cumsum(f0) / fs

where ``A(f)`` is the amplitude response of a cascade of two-pole formant
resonators with a spectral tilt.  Harmonics fade out smoothly between 85 % and
95 % of Nyquist so a gliding F0 never aliases.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

#: (centre Hz, bandwidth Hz) for an /a/-like and an /i/-like vowel.
FORMANTS_A = ((730.0, 90.0), (1090.0, 110.0), (2440.0, 170.0), (3400.0, 250.0))
FORMANTS_I = ((300.0, 60.0), (2300.0, 120.0), (3000.0, 200.0), (3700.0, 250.0))


def formant_amplitude(freqs: np.ndarray, fs: int, formants: Sequence[Tuple[float, float]],
                      tilt_db_per_octave: float = -6.0) -> np.ndarray:
    """Amplitude response (linear) of the formant cascade at ``freqs`` Hz."""
    freqs = np.asarray(freqs, dtype=np.float64)
    z = np.exp(-1j * 2.0 * np.pi * freqs / fs)
    response = np.ones_like(z)
    for centre, bandwidth in formants:
        r = np.exp(-np.pi * bandwidth / fs)
        theta = 2.0 * np.pi * centre / fs
        gain = 1.0 - 2.0 * r * np.cos(theta) + r * r          # unit gain at DC
        response *= gain / (1.0 - 2.0 * r * np.cos(theta) * z + r * r * z * z)
    tilt = (np.maximum(freqs, 50.0) / 50.0) ** (tilt_db_per_octave / (20.0 * np.log10(2.0)))
    return np.abs(response) * tilt


def harmonic_signal(f0: np.ndarray, fs: int,
                    amplitude: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Additive harmonic synthesis for a per-sample F0 track (0 = silence)."""
    f0 = np.asarray(f0, dtype=np.float64)
    phase = 2.0 * np.pi * np.cumsum(f0) / fs
    voiced = f0 > 0
    out = np.zeros_like(f0)
    if not voiced.any():
        return out
    n_harm = int(0.95 * fs / 2.0 / f0[voiced].min()) + 1
    for h in range(1, n_harm + 1):
        fh = h * f0
        fade = np.clip((0.95 * fs / 2.0 - fh) / (0.10 * fs / 2.0), 0.0, 1.0)
        if not np.any(fade[voiced] > 0):
            break
        out += np.where(voiced, amplitude(fh) * fade * np.sin(h * phase), 0.0)
    return out


def f0_contour(n: int, fs: int, start_hz: float, end_hz: float,
               vibrato_hz: float = 0.0, vibrato_cents: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fs
    base = start_hz * (end_hz / start_hz) ** (t / max(t[-1], 1e-9)) if n > 1 else \
        np.full(n, start_hz)
    return base * 2.0 ** (vibrato_cents / 1200.0 * np.sin(2.0 * np.pi * vibrato_hz * t))


def vowel(fs: int = 16000, duration: float = 0.5, start_hz: float = 120.0,
          end_hz: float = 160.0, vibrato_hz: float = 5.5, vibrato_cents: float = 30.0,
          formants=FORMANTS_A, noise_db: float = -40.0, level: float = 0.3,
          seed: int = 0) -> Dict[str, np.ndarray]:
    """A gliding, vibrato vowel with a little additive noise; returns signal + truth."""
    n = int(round(duration * fs))
    f0 = f0_contour(n, fs, start_hz, end_hz, vibrato_hz, vibrato_cents)
    amp = lambda f: formant_amplitude(f, fs, formants)  # noqa: E731
    x = harmonic_signal(f0, fs, amp)
    x *= level / max(np.sqrt(np.mean(x ** 2)), 1e-12)
    rng = np.random.default_rng(seed)
    if noise_db is not None:
        x += rng.standard_normal(n) * level * 10.0 ** (noise_db / 20.0)
    return {"x": x, "f0": f0}


def highpass_noise(n: int, fs: int, cutoff: float, level: float, seed: int) -> np.ndarray:
    """White noise high-passed in the frequency domain (fricative-like)."""
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec *= np.clip((freqs - cutoff) / 500.0, 0.0, 1.0)
    noise = np.fft.irfft(spec, n)
    return noise * level / max(np.sqrt(np.mean(noise ** 2)), 1e-12)


def mixed_utterance(fs: int = 16000, seed: int = 3) -> Dict[str, np.ndarray]:
    """silence | vowel /a/ | fricative noise | silence | vowel /i/ | silence."""
    parts, truth = [], []

    def add(x, f0):
        parts.append(x)
        truth.append(f0)

    gap = lambda d: (np.zeros(int(d * fs)), np.zeros(int(d * fs)))  # noqa: E731
    add(*gap(0.06))
    v1 = vowel(fs, 0.16, 110.0, 135.0, 5.0, 20.0, FORMANTS_A, -45.0, 0.25, seed)
    add(v1["x"], v1["f0"])
    n_fric = int(0.10 * fs)
    add(highpass_noise(n_fric, fs, 3000.0, 0.05, seed + 1), np.zeros(n_fric))
    add(*gap(0.04))
    v2 = vowel(fs, 0.16, 210.0, 190.0, 6.0, 25.0, FORMANTS_I, -45.0, 0.2, seed + 2)
    add(v2["x"], v2["f0"])
    add(*gap(0.04))
    return {"x": np.concatenate(parts), "f0": np.concatenate(truth)}


def frame_f0(f0_samples: np.ndarray, hop: int) -> np.ndarray:
    """Ground-truth F0 on the shared frame grid (sample at each frame centre)."""
    f0_samples = np.asarray(f0_samples, dtype=np.float64)
    n_frames = 1 + f0_samples.size // hop
    idx = np.minimum(np.arange(n_frames) * hop, f0_samples.size - 1)
    return f0_samples[idx]


def reference_signals() -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """The fixed signal set used by the stored reference fixtures.

    Returns ``{name: (signal float64 (exactly float32-representable), f0, fs)}``.
    """
    out = {}
    low = vowel(16000, 0.5, 115.0, 150.0, 5.5, 30.0, FORMANTS_A, -40.0, 0.3, seed=11)
    high = vowel(16000, 0.5, 300.0, 350.0, 6.0, 40.0, FORMANTS_I, -40.0, 0.3, seed=12)
    mixed = mixed_utterance(16000, seed=13)
    wide = vowel(22050, 0.4, 140.0, 180.0, 5.0, 25.0, FORMANTS_A, -40.0, 0.3, seed=14)
    for name, sig, fs in (("vowel_low", low, 16000), ("vowel_high", high, 16000),
                          ("mixed", mixed, 16000), ("vowel_22k", wide, 22050)):
        out[name] = (sig["x"].astype(np.float32).astype(np.float64), sig["f0"], fs)
    return out
