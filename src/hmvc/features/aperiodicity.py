"""Band aperiodicity (noise-to-total power ratio per frequency band).

For a voiced frame with period ``T0`` the band-limited, window-corrected
normalised autocorrelation at lag ``T0`` is the band's *harmonicity*
(P. Boersma, "Accurate short-term analysis of the fundamental frequency and the
harmonics-to-noise ratio of a sampled sound", IFA Proceedings 17, 1993):

    h_b = [r_b(T0) / r_b(0)] / [r_w(T0) / r_w(0)]

where ``r_b`` is the autocorrelation of the windowed frame restricted to band
``b`` (evaluated directly from the power spectrum, so ``T0`` may be
fractional) and ``r_w`` the autocorrelation of the window itself.  For a
periodic component plus uncorrelated noise ``h_b`` is the periodic power
fraction, so ``ap_b = 1 - h_b`` (clipped to [0, 1]) is the noise power
fraction -- the same quantity WORLD's aperiodicity denotes, here at band
resolution.

Windows are F0-adaptive Hann windows spanning ``periods`` pitch periods.
Unvoiced frames, and bands without measurable energy, are fully aperiodic
(1.0).
"""

from __future__ import annotations

import numpy as np

from hmvc.features.config import next_power_of_two
from hmvc.features.framing import blocks


def hz_to_mel(f):
    return 1127.01048 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * np.expm1(np.asarray(m, dtype=np.float64) / 1127.01048)


def band_edges(n_bands: int, sample_rate: int) -> np.ndarray:
    """``n_bands + 1`` mel-spaced (HTK mel) band edges from 0 Hz to Nyquist."""
    return mel_to_hz(np.linspace(0.0, float(hz_to_mel(sample_rate / 2.0)), int(n_bands) + 1))


def _half_lengths(fs, f0, periods):
    """Per-frame half window length (samples) of a ``periods``-period window."""
    return np.floor(0.5 * periods * fs / f0 + 0.5).astype(np.int64)


def _fft_sizes(half):
    """Per-frame FFT size; depends only on the frame's own window length."""
    return np.array([next_power_of_two(2 * (2 * int(h) + 1)) for h in np.atleast_1d(half)],
                    dtype=np.int64)


def _band_block(x, fs, f0, origins, edges, periods, energy_floor, n_fft):
    n = x.size
    half = _half_lengths(fs, f0, periods)
    h_max = int(half.max())
    base = np.arange(-h_max, h_max + 1)[None, :]
    mask = np.abs(base) <= half[:, None]
    window = np.where(mask, 0.5 + 0.5 * np.cos(np.pi * base / half[:, None]), 0.0)
    idx = origins[:, None] + base
    inside = (idx >= 0) & (idx < n)                        # zeros outside the signal
    segment = np.where(inside, x[np.clip(idx, 0, max(n - 1, 0))], 0.0) * window
    segment -= window * (segment.sum(axis=1, keepdims=True) / window.sum(axis=1, keepdims=True))
    spec = np.fft.rfft(segment, n_fft, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    wspec = np.fft.rfft(window, n_fft, axis=1)
    wpower = wspec.real ** 2 + wspec.imag ** 2
    n_bins = power.shape[1]
    k = np.arange(n_bins)
    weight = np.full(n_bins, 2.0)
    weight[0] = 1.0
    weight[-1] = 1.0
    lag = fs / f0                                           # fractional period
    cos = np.cos(2.0 * np.pi * k[None, :] * lag[:, None] / n_fft) * weight[None, :]
    rw_ratio = np.sum(wpower * cos, axis=1) / np.sum(wpower * weight[None, :], axis=1)
    # Per-frame band edges snapped to the nearest inter-harmonic midpoint
    # ((h + 1/2) F0): with a Hann window of 4 periods those are exactly the
    # first spectral nulls, so every band holds whole harmonic lobes and the
    # window correction r_w(T0)/r_w(0) applies to each band unchanged.
    interior = np.asarray(edges[1:-1], dtype=np.float64)
    snapped = (np.floor(interior[None, :] / f0[:, None]) + 0.5) * f0[:, None]
    nyq = fs / 2.0
    lo_hz = np.concatenate([np.full((f0.size, 1), edges[0]), snapped], axis=1)
    hi_hz = np.concatenate([snapped, np.full((f0.size, 1), min(edges[-1], nyq))], axis=1)
    bin_hz = fs / n_fft
    lo_bin = np.clip(np.ceil(lo_hz / bin_hz - 1e-9), 0, n_bins).astype(np.int64)
    hi_bin = np.clip(np.ceil(hi_hz / bin_hz - 1e-9), 0, n_bins).astype(np.int64)
    hi_bin[:, -1] = np.where(edges[-1] >= nyq - 1e-9, n_bins,
                             hi_bin[:, -1])            # include Nyquist in last band
    zeros = np.zeros((f0.size, 1))
    cum_r0 = np.concatenate([zeros, np.cumsum(power * weight[None, :], axis=1)], axis=1)
    cum_rt = np.concatenate([zeros, np.cumsum(power * cos, axis=1)], axis=1)
    r0 = np.take_along_axis(cum_r0, hi_bin, 1) - np.take_along_axis(cum_r0, lo_bin, 1)
    rt = np.take_along_axis(cum_rt, hi_bin, 1) - np.take_along_axis(cum_rt, lo_bin, 1)
    total = cum_r0[:, -1:]
    valid = (hi_bin > lo_bin) & (r0 > energy_floor * np.maximum(total, 1e-300)) & (r0 > 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonicity = (rt / r0) / rw_ratio[:, None]
    harmonicity = np.where(np.isfinite(harmonicity), harmonicity, 0.0)
    return np.where(valid, 1.0 - np.clip(harmonicity, 0.0, 1.0), 1.0)


def band_aperiodicity(x: np.ndarray, sample_rate: int, f0: np.ndarray, hop: int,
                      edges: np.ndarray, periods: float = 4.0,
                      energy_floor: float = 1e-10, block: int = 512) -> np.ndarray:
    """Band aperiodicity ``(T, len(edges) - 1)`` in [0, 1] on the shared grid."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.float64)
    out = np.ones((f0.size, len(edges) - 1))
    voiced = np.flatnonzero(np.isfinite(f0) & (f0 > 0))
    if voiced.size == 0 or x.size == 0:
        return out
    # Frames are grouped by FFT size so that each frame's value depends only
    # on its own F0 and samples (block-size and neighbour invariant).
    sizes = _fft_sizes(_half_lengths(int(sample_rate), f0[voiced], float(periods)))
    for n_fft in np.unique(sizes):
        group = voiced[sizes == n_fft]
        for start, stop in blocks(group.size, block):
            frames = group[start:stop]
            out[frames] = _band_block(x, int(sample_rate), f0[frames],
                                      frames.astype(np.int64) * int(hop), edges,
                                      float(periods), float(energy_floor), int(n_fft))
    return out
