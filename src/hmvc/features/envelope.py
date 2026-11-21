"""Spectral-envelope estimation.

Three classical estimators are provided.  All return a linear **power**
envelope ``(T, fft_size//2 + 1)`` on the shared frame grid.

``cheaptrick`` (default)
    An independent NumPy implementation of WORLD's CheapTrick algorithm
    (M. Morise, "CheapTrick, a spectral envelope estimator for high-quality
    speech synthesis", Speech Communication 67, 2015):

    1. F0-adaptive Hann window spanning three pitch periods, normalised to unit
       energy, with the window-weighted DC removed;
    2. power spectrum, with the region below F0 "folded" (DC correction);
    3. rectangular smoothing over 2*F0/3 on the linear-frequency axis;
    4. log, then cepstral liftering with ``sinc(F0 q)`` (smoothing) times
       ``(1 - 2 q1) + 2 q1 cos(2 pi F0 q)`` (spectral recovery).

    The window is matched to the period, so the harmonic structure is removed
    by construction and the estimate is (to first order) independent of F0 --
    the property voice conversion needs most.  Frames without F0 use
    ``default_f0`` = 500 Hz, as WORLD does.  Differences from WORLD's C++:
    edge samples are replicated identically, but WORLD's random 1e-12
    safeguards are replaced by a deterministic power floor (bit-for-bit
    repeatable features), and a voiced F0 below the window limit is raised to
    that limit rather than replaced by ``default_f0``.

``true_envelope``
    Iterative cepstral smoothing that keeps raising the log spectrum to the
    envelope until it rides on the harmonic peaks (A. Röbel and X. Rodet,
    "Efficient spectral envelope estimation and its application to pitch
    shifting and envelope preservation", DAFx 2005), with the F0-adaptive
    order ``fs / (2 F0)``.

``lifter``
    Plain low-quefrency liftering of the log-magnitude spectrum with the same
    F0-adaptive order.  Cheapest, but biased: it averages harmonic peaks and
    inter-harmonic valleys, so its level and shape depend on F0.
"""

from __future__ import annotations

import numpy as np

from hmvc.features.framing import blocks

#: F0 (Hz) used for frames without a pitch estimate (WORLD's kDefaultF0).
DEFAULT_F0 = 500.0


def cheaptrick_f0_limit(sample_rate: int, fft_size: int) -> float:
    """Lowest F0 whose three-period window fits in ``fft_size`` (WORLD convention)."""
    return 3.0 * sample_rate / (fft_size - 3.0)


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int64)


def _interp_rows(y: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation of ``y`` (B, K) at fractional ``pos`` (B, J).

    Mirrors WORLD's ``interp1Q``: base = trunc(pos), slope taken from the next
    sample, zero slope at the last sample.
    """
    k = y.shape[1]
    base = np.clip(np.floor(pos).astype(np.int64), 0, k - 1)
    frac = pos - base
    nxt = np.minimum(base + 1, k - 1)
    y0 = np.take_along_axis(y, base, axis=1)
    y1 = np.take_along_axis(y, nxt, axis=1)
    return y0 + (y1 - y0) * frac


def _cheaptrick_block(x: np.ndarray, fs: int, f0: np.ndarray, origins: np.ndarray,
                      fft_size: int, q1: float, power_floor: float) -> np.ndarray:
    n_frames = f0.shape[0]
    n = x.size
    half_len = _round_half_up(1.5 * fs / f0)                       # (B,)
    h_max = int(half_len.max())
    base = np.arange(-h_max, h_max + 1)[None, :]                    # (1, L)
    mask = np.abs(base) <= half_len[:, None]
    idx = np.clip(origins[:, None] + base, 0, n - 1)
    window = np.where(mask, 0.5 * np.cos(np.pi * base / (1.5 * fs) * f0[:, None]) + 0.5, 0.0)
    window /= np.sqrt(np.sum(window ** 2, axis=1, keepdims=True))
    segment = x[idx] * window
    segment -= window * (segment.sum(axis=1, keepdims=True) / window.sum(axis=1, keepdims=True))

    spec = np.fft.rfft(segment, fft_size, axis=1)
    power = spec.real ** 2 + spec.imag ** 2                          # (B, F+1)
    n_bins = power.shape[1]
    delta = fs / fft_size

    # DC correction: P(f) += P(f0 - f) for f below F0.
    limits = (f0 * fft_size / fs).astype(np.int64)                  # bins 0..limit
    k_max = int(limits.max()) + 1
    i = np.arange(k_max)[None, :]
    pos = np.clip(f0[:, None] / delta - i, 0.0, n_bins - 1)
    replica = _interp_rows(power, pos)
    power[:, :k_max] += np.where(i <= limits[:, None], replica, 0.0)

    # Linear smoothing over width 2*F0/3 (cumulative-sum box filter on a
    # spectrum mirrored about DC and Nyquist).
    width = 2.0 * f0 / 3.0
    boundary = int((width.max() * fft_size / fs)) + 1
    half = fft_size // 2
    j = np.arange(half + 2 * boundary + 1)
    mirror_idx = np.where(j < boundary, boundary - j,
                          np.where(j < half + boundary, j - boundary,
                                   half - (j - (half + boundary))))
    segment_sum = np.cumsum(power[:, mirror_idx] * delta, axis=1)
    k = np.arange(n_bins)[None, :]
    offset = boundary - 0.5
    low = _interp_rows(segment_sum, k - width[:, None] / (2.0 * delta) + offset)
    high = _interp_rows(segment_sum, k + width[:, None] / (2.0 * delta) + offset)
    smoothed = (high - low) / width[:, None]

    # Smoothing and spectral recovery in the cepstral domain.
    log_power = np.log(np.maximum(smoothed, power_floor))
    cep = np.fft.irfft(log_power, fft_size, axis=1)
    quef = np.arange(fft_size)
    quef = np.minimum(quef, fft_size - quef)[None, :] / float(fs)   # symmetric
    lift = np.sinc(f0[:, None] * quef) * ((1.0 - 2.0 * q1)
                                          + 2.0 * q1 * np.cos(2.0 * np.pi * f0[:, None] * quef))
    return np.exp(np.fft.rfft(cep * lift, axis=1).real)


def cheaptrick_frames(x: np.ndarray, sample_rate: int, f0: np.ndarray, start: int,
                      hop: int, fft_size: int, q1: float = -0.15,
                      power_floor: float = 1e-12, default_f0: float = DEFAULT_F0
                      ) -> np.ndarray:
    """CheapTrick envelopes for frames ``start .. start + len(f0) - 1`` (one block)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    n_bins = int(fft_size) // 2 + 1
    if f0.size == 0:
        return np.empty((0, n_bins))
    if x.size == 0:
        return np.full((f0.size, n_bins), float(power_floor))
    limit = cheaptrick_f0_limit(sample_rate, fft_size)
    if default_f0 < limit:
        raise ValueError(f"default_f0={default_f0} is below the window limit {limit:.1f} Hz")
    voiced = np.isfinite(f0) & (f0 > 0)
    current = np.where(voiced, np.maximum(f0, limit), default_f0)
    origins = (int(start) + np.arange(f0.size, dtype=np.int64)) * int(hop)
    return _cheaptrick_block(x, int(sample_rate), current, origins, int(fft_size),
                             float(q1), float(power_floor))


def cheaptrick(x: np.ndarray, sample_rate: int, f0: np.ndarray, hop: int, fft_size: int,
               q1: float = -0.15, default_f0: float = DEFAULT_F0,
               power_floor: float = 1e-12, block: int = 512) -> np.ndarray:
    """CheapTrick power envelope ``(T, fft_size//2 + 1)`` for frames centred at ``t*hop``.

    ``f0`` is ``(T,)`` in Hz with 0 for unvoiced frames.  Processed in blocks
    of ``block`` frames to bound memory.
    """
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    out = np.empty((f0.size, int(fft_size) // 2 + 1), dtype=np.float64)
    for lo, hi in blocks(f0.size, block):
        out[lo:hi] = cheaptrick_frames(x, sample_rate, f0[lo:hi], lo, hop, fft_size, q1,
                                       power_floor, default_f0)
    return out


# --------------------------------------------------------------------------
# Cepstral smoothing alternatives (fixed-window spectra)
# --------------------------------------------------------------------------


def adaptive_order(sample_rate: int, f0: np.ndarray, n_bins: int,
                   default_f0: float = DEFAULT_F0) -> np.ndarray:
    """F0-adaptive cepstral order ``fs / (2 F0)`` (unvoiced: ``default_f0``)."""
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    current = np.where(np.isfinite(f0) & (f0 > 0), f0, default_f0)
    return np.clip(np.floor(sample_rate / (2.0 * current)), 2, n_bins - 2)


def _smooth_log(log_amp: np.ndarray, order: np.ndarray) -> np.ndarray:
    n_fft = 2 * (log_amp.shape[1] - 1)
    cep = np.fft.irfft(log_amp, n_fft, axis=1)
    quef = np.arange(n_fft)
    quef = np.minimum(quef, n_fft - quef)[None, :]
    return np.fft.rfft(cep * (quef <= order[:, None]), axis=1).real


def lifter_envelope(power: np.ndarray, sample_rate: int, f0: np.ndarray,
                    power_floor: float = 1e-12) -> np.ndarray:
    """Plain cepstral-lifter power envelope from fixed-window power spectra."""
    power = np.atleast_2d(np.asarray(power, dtype=np.float64))
    order = adaptive_order(sample_rate, f0, power.shape[1])
    log_amp = 0.5 * np.log(np.maximum(power, power_floor))
    return np.exp(2.0 * _smooth_log(log_amp, order))


def true_envelope(power: np.ndarray, sample_rate: int, f0: np.ndarray,
                  power_floor: float = 1e-12, threshold_db: float = 1.0,
                  max_iter: int = 200) -> np.ndarray:
    """True-envelope power estimate (iterative cepstral smoothing, Röbel & Rodet)."""
    power = np.atleast_2d(np.asarray(power, dtype=np.float64))
    order = adaptive_order(sample_rate, f0, power.shape[1])
    target = 0.5 * np.log(np.maximum(power, power_floor))
    current = target.copy()
    envelope = _smooth_log(current, order)
    thresh = threshold_db * np.log(10.0) / 20.0
    active = np.arange(power.shape[0])
    for _ in range(max_iter):
        if active.size == 0:
            break
        excess = np.max(target[active] - envelope[active], axis=1)
        active = active[excess > thresh]
        if active.size == 0:
            break
        current[active] = np.maximum(target[active], envelope[active])
        envelope[active] = _smooth_log(current[active], order[active])
    return np.exp(2.0 * envelope)
