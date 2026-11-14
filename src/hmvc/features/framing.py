"""Frame grid, windows and short-time spectra.

One frame grid is shared by every feature stream:

    frame t is centred on sample  t * hop
    an n-sample signal has        1 + n // hop  frames  (0 frames if n == 0)

which is WORLD's convention (``f0_length = 1 + floor(duration / frame_period)``)
expressed in integer samples.  Fixed-length frames are cut from a zero-padded
copy of the signal, so the first and last frames are always defined, even for
signals shorter than one frame.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from hmvc.features.config import WINDOWS


def frame_count(n_samples: int, hop: int) -> int:
    """Number of frames on the shared grid for an ``n_samples`` signal."""
    if hop < 1:
        raise ValueError("hop must be >= 1")
    return 0 if n_samples <= 0 else 1 + int(n_samples) // int(hop)


def frame_times(n_frames: int, hop: int, sample_rate: int) -> np.ndarray:
    """Frame centre times in seconds."""
    return np.arange(int(n_frames)) * (float(hop) / float(sample_rate))


def get_window(name: str, length: int) -> np.ndarray:
    """Periodic ("DFT-even") analysis window of ``length`` samples.

    Periodic windows are the STFT convention (they match ``librosa`` /
    ``scipy.signal.get_window(..., fftbins=True)``).
    """
    if name not in WINDOWS:
        raise ValueError(f"unknown window {name!r}; expected one of {WINDOWS}")
    n = np.arange(int(length), dtype=np.float64)
    phase = 2.0 * np.pi * n / max(int(length), 1)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(phase)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(phase)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    return np.ones(int(length), dtype=np.float64)


def frame_signal(x: np.ndarray, frame_length: int, hop: int,
                 n_frames: int | None = None, start: int = 0,
                 left: int | None = None) -> np.ndarray:
    """Frames ``(n, frame_length)`` for frames ``start .. start+n-1``.

    Frame ``t`` covers samples ``[t*hop - left, t*hop - left + frame_length)``
    with ``left = ceil(frame_length / 2)`` by default (centred frames: the
    librosa/``numpy.fft`` convention of centring a window inside an even-length
    FFT frame; it only matters for odd lengths).  Samples outside the signal
    are zeros.  Returns a copy (safe to modify).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    frame_length, hop = int(frame_length), int(hop)
    total = frame_count(x.size, hop)
    if n_frames is None:
        n_frames = total - start
    n_frames = max(0, int(n_frames))
    if n_frames == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    left = (frame_length + 1) // 2 if left is None else int(left)
    first = start * hop - left
    last = (start + n_frames - 1) * hop - left + frame_length
    lo, hi = max(first, 0), min(last, x.size)
    segment = np.zeros(last - first, dtype=np.float64)
    if hi > lo:
        segment[lo - first:hi - first] = x[lo:hi]
    view = np.lib.stride_tricks.sliding_window_view(segment, frame_length)[::hop]
    return np.array(view[:n_frames])


def blocks(n_frames: int, block: int) -> Iterator[Tuple[int, int]]:
    """``(start, stop)`` frame ranges of at most ``block`` frames."""
    for start in range(0, int(n_frames), max(1, int(block))):
        yield start, min(start + block, int(n_frames))


def power_spectrum(frames: np.ndarray, window: np.ndarray, fft_size: int) -> np.ndarray:
    """Windowed power spectrum ``(n, fft_size//2 + 1)``, normalised by ``sum(w**2)``.

    The normalisation makes the value a power *density* independent of the
    frame length: white noise of variance s**2 has an expected value of s**2
    in every bin whatever ``frame_length``/``window`` are.
    """
    frames = np.atleast_2d(frames)
    spec = np.fft.rfft(frames * window[None, :], int(fft_size), axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    return power / float(np.sum(window ** 2))


def log_magnitude(power: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Natural-log magnitude ``0.5 * log(max(power, floor))``."""
    return 0.5 * np.log(np.maximum(power, floor))
