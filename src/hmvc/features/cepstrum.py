"""Cepstral representations: real cepstrum, liftering and mel-cepstrum.

Conventions (SPTK / pysptk compatible)
--------------------------------------
For a power spectrum ``P(w)`` the *amplitude* real cepstrum ``c`` satisfies

    log |X(w)| = 0.5 log P(w) = c_0 + sum_{n>=1} c_n cos(n w)

i.e. ``c = irfft(log P)`` with ``c_0`` halved -- exactly pysptk's ``sp2mc``.

The mel-cepstrum ``c~`` is the same function expanded on a warped frequency
axis.  With the first-order all-pass

    z~^-1 = (z^-1 - alpha) / (1 - alpha z^-1),
    beta(w) = w + 2 atan(alpha sin w / (1 - alpha cos w))

we have ``sum_n c_n z^-n = sum_m c~_m z~^-m`` and therefore

    log |X(w)| = sum_{m=0}^{M} c~_m cos(m beta(w)).            (1)

Because ``cos(m beta)`` is orthogonal on ``beta in [0, pi]``, truncating at
order M is the least-squares approximation of the log spectrum on the warped
(mel-like) axis.  (1) is also exactly the magnitude response of the MLSA
filter, so the coefficients are directly usable for MLSA synthesis.

Finite-order MCEP is therefore a reconstructible, compact approximation of the
envelope, not a lossless encoding: ``mc2sp(sp2mc(P, M, alpha), alpha, n)``
returns the order-M (smoothed) approximation of ``P``.  It reproduces ``P``
(to numerical precision) only when the warped cepstrum of ``P`` vanishes above
order M; otherwise the error falls as M grows (about 2.2 dB RMS log-spectral
distance at M = 24 on 16 kHz speech envelopes, see docs/feature-extraction.md
section 4.3).

Implementation
--------------
SPTK's ``freqt`` recursion is linear in its input, so the transform is a fixed
``(M+1, N+1)`` matrix for given (N, M, alpha).  It is built column by column:
``z^-n = (z^-1)^n`` and ``z^-1 = (z~^-1 + alpha) / (1 + alpha z~^-1)``, whose
power series is ``g = [alpha, (1-alpha^2), (1-alpha^2)(-alpha), ...]``, so
column ``n`` is the (truncated) n-fold convolution power of ``g``.  This is
exact (identical to the recursion up to rounding), numerically stable (the
all-pass has unit gain, so no column grows), cached, and turns sp -> mcep into
one matrix product per block of frames.  :func:`freqt_recursive` is a direct
port of SPTK's recursion, kept as the reference the tests compare against.
"""

from __future__ import annotations

import functools
import math

import numpy as np

#: Natural log -> dB conversion for amplitudes (20 / ln 10).
_DB = 20.0 / math.log(10.0)


# --------------------------------------------------------------------------
# All-pass warping coefficient
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def mcep_alpha(sample_rate: int, start: float = 0.0, stop: float = 1.0,
               step: float = 0.001, num_points: int = 1000) -> float:
    """All-pass constant that best approximates the mel scale at ``sample_rate``.

    Re-implementation of ``pysptk.util.mcepalpha`` (grid search of the
    normalised warped-frequency curve against ``1000/ln2 * ln(1 + f/1000)``),
    giving the customary values 0.41 (16 kHz), 0.455 (22.05 kHz), 0.544
    (44.1 kHz) and 0.554 (48 kHz).
    """
    fs = float(sample_rate)
    freq_step = (fs / 2.0) / num_points
    mel = 1000.0 / math.log(2.0) * np.log1p(freq_step * np.arange(num_points) / 1000.0)
    mel = mel / mel[-1]
    omega = (math.pi / num_points) * np.arange(num_points)
    best_alpha, best_dist = start, math.inf
    for alpha in np.arange(start, stop, step):
        num = (1.0 - alpha * alpha) * np.sin(omega)
        den = (1.0 + alpha * alpha) * np.cos(omega) - 2.0 * alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            warp = np.arctan(num / den)
        warp = np.where(warp < 0.0, warp + math.pi, warp)
        warp = warp / warp[-1]
        dist = float(np.sum((mel - warp) ** 2)) / num_points
        if dist < best_dist:
            best_alpha, best_dist = float(alpha), dist
    return best_alpha


def warped_frequency(omega: np.ndarray, alpha: float) -> np.ndarray:
    """``beta(w)``: phase response of the all-pass (linear -> warped axis)."""
    omega = np.asarray(omega, dtype=np.float64)
    return omega + 2.0 * np.arctan2(alpha * np.sin(omega), 1.0 - alpha * np.cos(omega))


# --------------------------------------------------------------------------
# Real cepstrum and liftering
# --------------------------------------------------------------------------


def real_cepstrum(power: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Amplitude real cepstrum ``(T, N//2 + 1)`` of power spectra ``(T, N//2 + 1)``.

    ``c[0]`` is halved (and the Nyquist term halved too), so that
    ``log|X(w)| = sum_n c_n cos(n w)`` -- the SPTK convention.
    """
    power = np.atleast_2d(np.asarray(power, dtype=np.float64))
    n_fft = 2 * (power.shape[1] - 1)
    full = np.fft.irfft(np.log(np.maximum(power, floor)), n_fft, axis=1)
    cep = full[:, :n_fft // 2 + 1].copy()
    cep[:, 0] *= 0.5
    cep[:, -1] *= 0.5
    return cep


def cepstrum_to_log_amplitude(cep: np.ndarray, n_bins: int) -> np.ndarray:
    """Inverse of :func:`real_cepstrum`: ``(T, Q)`` cepstrum -> ``(T, n_bins)`` log|X|."""
    # log|X(w_k)| = sum_n c_n cos(n w_k): build the even sequence s with
    # s_0 = c_0, s_n = s_{N-n} = c_n / 2 (0 < n < N/2), s_{N/2} = c_{N/2};
    # then rfft(s)_k is exactly that cosine sum.
    cep = np.atleast_2d(np.asarray(cep, dtype=np.float64))
    n_fft = 2 * (int(n_bins) - 1)
    half = n_fft // 2
    q = min(cep.shape[1], half + 1)
    sym = np.zeros((cep.shape[0], n_fft), dtype=np.float64)
    sym[:, 0] = cep[:, 0]
    inner = min(q, half)
    if inner > 1:
        sym[:, 1:inner] = 0.5 * cep[:, 1:inner]
        sym[:, n_fft - inner + 1:] = 0.5 * cep[:, 1:inner][:, ::-1]
    if q == half + 1:
        sym[:, half] = cep[:, half]
    return np.fft.rfft(sym, axis=1).real


def lifter(cep: np.ndarray, order, kind: str = "rectangular") -> np.ndarray:
    """Low-quefrency lifter: keep quefrencies ``0..order`` of ``cep``.

    ``order`` may be a scalar or a per-frame ``(T,)`` array.  ``kind`` is
    ``"rectangular"`` (hard cut) or ``"hann"`` (half-Hann taper reaching zero
    just after ``order``, which avoids Gibbs ripple in the envelope).
    """
    cep = np.atleast_2d(np.asarray(cep, dtype=np.float64))
    q = np.arange(cep.shape[1], dtype=np.float64)[None, :]
    order = np.asarray(order, dtype=np.float64).reshape(-1, 1)
    if kind == "rectangular":
        weights = (q <= order).astype(np.float64)
    elif kind == "hann":
        weights = np.where(q <= order, 0.5 + 0.5 * np.cos(np.pi * q / (order + 1.0)), 0.0)
    else:
        raise ValueError("kind must be 'rectangular' or 'hann'")
    return cep * weights


# --------------------------------------------------------------------------
# Frequency warping (freqt) and mel-cepstrum
# --------------------------------------------------------------------------


def freqt_recursive(c: np.ndarray, order: int, alpha: float) -> np.ndarray:
    """Direct port of SPTK's ``freqt`` (reference implementation, slow).

    ``c`` is ``(..., N+1)`` (any leading frame axes); returns ``(..., order+1)``.
    """
    c = np.asarray(c, dtype=np.float64)
    lead = c.shape[:-1]
    c1 = c.reshape(-1, c.shape[-1])
    m1 = c1.shape[1] - 1
    b = 1.0 - alpha * alpha
    c2 = np.zeros((c1.shape[0], order + 1))
    for i in range(-m1, 1):
        d = c2.copy()
        c2[:, 0] = c1[:, -i] + alpha * d[:, 0]
        if order >= 1:
            c2[:, 1] = b * d[:, 0] + alpha * d[:, 1]
        for j in range(2, order + 1):
            c2[:, j] = d[:, j - 1] + alpha * (d[:, j] - c2[:, j - 1])
    return c2.reshape(lead + (order + 1,))


@functools.lru_cache(maxsize=32)
def _freqt_matrix_cached(n_in: int, order: int, alpha_key: float) -> np.ndarray:
    alpha = float(alpha_key)
    m = order + 1
    g = np.empty(m, dtype=np.float64)
    g[0] = alpha
    if m > 1:
        g[1:] = (1.0 - alpha * alpha) * (-alpha) ** np.arange(m - 1)
    matrix = np.zeros((m, n_in), dtype=np.float64)
    column = np.zeros(m, dtype=np.float64)
    column[0] = 1.0
    matrix[:, 0] = column
    for n in range(1, n_in):
        column = np.convolve(column, g)[:m]
        matrix[:, n] = column
    matrix.setflags(write=False)
    return matrix


def freqt_matrix(n_in: int, order: int, alpha: float) -> np.ndarray:
    """``(order+1, n_in)`` matrix ``A`` with ``freqt(c) == A @ c`` (read-only, cached)."""
    if n_in < 1 or order < 0:
        raise ValueError("n_in must be >= 1 and order >= 0")
    if not (-1.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly inside (-1, 1)")
    return _freqt_matrix_cached(int(n_in), int(order), round(float(alpha), 12))


def freqt(c: np.ndarray, order: int, alpha: float) -> np.ndarray:
    """Frequency-warp cepstra ``(..., N+1)`` -> ``(..., order+1)`` (SPTK ``freqt``)."""
    c = np.asarray(c, dtype=np.float64)
    return c @ freqt_matrix(c.shape[-1], order, alpha).T


def sp2mc(power: np.ndarray, order: int, alpha: float, floor: float = 1e-12) -> np.ndarray:
    """Power spectral envelope ``(T, N//2+1)`` -> mel-cepstrum ``(T, order+1)``.

    Numerically equivalent to ``pysptk.sp2mc`` (see the module docstring).
    The result is a truncated, order-``order`` approximation of the envelope;
    :func:`mc2sp` rebuilds that approximation, not the input itself.
    """
    return freqt(real_cepstrum(power, floor=floor), order, alpha)


@functools.lru_cache(maxsize=32)
def _mc2sp_basis(order: int, n_bins: int, alpha_key: float) -> np.ndarray:
    omega = np.linspace(0.0, np.pi, int(n_bins))
    beta = warped_frequency(omega, float(alpha_key))
    basis = np.cos(np.arange(order + 1)[:, None] * beta[None, :])
    basis.setflags(write=False)
    return basis


def mc2sp(mcep: np.ndarray, alpha: float, fft_size: int,
          log_clip: float = 80.0) -> np.ndarray:
    """Mel-cepstrum ``(T, M+1)`` -> power envelope ``(T, fft_size//2+1)`` via (1).

    Evaluates the truncated series exactly (pysptk's ``mc2sp`` first warps back
    to ``fft_size/2`` linear coefficients, which differs only by that
    truncation).  For coefficients from :func:`sp2mc` the result is the
    order-M approximation of the analysed envelope, not the envelope itself.
    ``log_clip`` bounds the log power so malformed input cannot overflow.
    """
    mcep = np.atleast_2d(np.asarray(mcep, dtype=np.float64))
    basis = _mc2sp_basis(mcep.shape[1] - 1, int(fft_size) // 2 + 1, round(float(alpha), 12))
    return np.exp(np.clip(2.0 * (mcep @ basis), -log_clip, log_clip))


def mc2c(mcep: np.ndarray, alpha: float, n_out: int) -> np.ndarray:
    """Mel-cepstrum -> linear cepstrum with ``n_out`` coefficients (``freqt`` with -alpha)."""
    return freqt(mcep, int(n_out) - 1, -float(alpha))


# --------------------------------------------------------------------------
# Distortion measures
# --------------------------------------------------------------------------


def mel_cepstral_distortion(a: np.ndarray, b: np.ndarray, include_c0: bool = False
                            ) -> np.ndarray:
    """Per-frame mel-cepstral distortion in dB (Kubichek / SPTK convention).

    ``MCD = (10 / ln 10) * sqrt(2 * sum_{m>=1} (a_m - b_m)^2)``.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    start = 0 if include_c0 else 1
    diff = a[:, start:] - b[:, start:]
    return (10.0 / math.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=1))


def log_spectral_distance(p: np.ndarray, q: np.ndarray, floor: float = 1e-12,
                          remove_gain: bool = False, weights: np.ndarray | None = None
                          ) -> np.ndarray:
    """Per-frame RMS log-spectral distance in dB between power spectra.

    ``remove_gain`` subtracts each frame's mean log difference first (a shape-
    only comparison).  ``weights`` (per bin) gives a weighted RMS.
    """
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    diff = 10.0 * np.log10(np.maximum(p, floor) / np.maximum(q, floor))
    if weights is None:
        weights = np.ones(diff.shape[1])
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    if remove_gain:
        diff = diff - np.sum(diff * weights, axis=1, keepdims=True)
    return np.sqrt(np.sum(diff ** 2 * weights, axis=1))


def mel_weights(n_bins: int, sample_rate: int) -> np.ndarray:
    """Per-bin weights proportional to d(mel)/df (HTK mel), for mel-weighted LSD."""
    f = np.linspace(0.0, sample_rate / 2.0, int(n_bins))
    return 1.0 / (700.0 + f)
