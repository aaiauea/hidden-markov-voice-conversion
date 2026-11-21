"""Frame energy and spectral descriptors on fixed analysis frames.

Energy is measured on the *unwindowed* frame (``sqrt(mean(x^2))``, the
``librosa.feature.rms`` convention).  Spectral descriptors are computed from
the windowed STFT of the same frames; conventions follow librosa where an
established definition exists so that values are comparable:

=============  ==============================================================
centroid       magnitude-weighted mean frequency (Hz)
bandwidth      magnitude-weighted standard deviation around the centroid (Hz)
rolloff        lowest frequency below which ``rolloff_percent`` of the
               magnitude sum lies (Hz)
flatness       geometric / arithmetic mean of the power spectrum, in [0, 1]
tilt           least-squares slope of the power spectrum in dB per octave
               over ``tilt_range_hz``
zcr            zero-crossing rate: sign changes per sample in the frame
flux           L2 norm of the positive change of the (sum-normalised)
               magnitude spectrum relative to the previous frame
=============  ==============================================================

Silent frames yield finite values: centroid/bandwidth/rolloff/tilt 0,
flatness 1 (a flat floor), zcr 0, flux 0.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

DESCRIPTORS = ("centroid", "bandwidth", "rolloff", "flatness", "tilt", "zcr", "flux")


def frame_rms(frames: np.ndarray) -> np.ndarray:
    frames = np.atleast_2d(frames)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def rms_to_db(rms: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
    """``20 log10(rms)`` relative to full scale, floored at ``floor_db``."""
    floor = 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(rms, floor))


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    frames = np.atleast_2d(frames)
    signs = np.signbit(frames)
    crossings = np.sum(signs[:, 1:] != signs[:, :-1], axis=1)
    return crossings / float(frames.shape[1])


def spectral_descriptors(power: np.ndarray, sample_rate: int,
                         rolloff_percent: float = 0.85,
                         tilt_range_hz: Tuple[float, float] = (100.0, 5000.0),
                         power_floor: float = 1e-12,
                         previous_magnitude: np.ndarray | None = None
                         ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Descriptors for power spectra ``(T, bins)``.

    Returns ``(descriptors, last_normalised_magnitude)``; pass the latter as
    ``previous_magnitude`` when processing the next block so spectral flux is
    continuous across block boundaries.
    """
    power = np.atleast_2d(np.asarray(power, dtype=np.float64))
    n_bins = power.shape[1]
    freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    mag = np.sqrt(np.maximum(power, 0.0))
    total = mag.sum(axis=1)
    alive = total > np.sqrt(power_floor) * n_bins
    safe_total = np.where(alive, total, 1.0)
    norm = mag / safe_total[:, None]
    centroid = np.where(alive, norm @ freqs, 0.0)
    spread = np.sum(norm * (freqs[None, :] - centroid[:, None]) ** 2, axis=1)
    bandwidth = np.where(alive, np.sqrt(np.maximum(spread, 0.0)), 0.0)
    cum = np.cumsum(mag, axis=1)
    reached = cum >= rolloff_percent * cum[:, -1:]
    rolloff = np.where(alive, freqs[np.argmax(reached, axis=1)], 0.0)
    floored = np.maximum(power, power_floor)
    flatness = np.exp(np.mean(np.log(floored), axis=1)) / np.mean(floored, axis=1)
    lo, hi = tilt_range_hz
    hi = min(hi, 0.95 * sample_rate / 2.0)
    band = (freqs >= lo) & (freqs <= hi)
    if band.sum() >= 2:
        octave = np.log2(freqs[band])
        octave = octave - octave.mean()
        level = 10.0 * np.log10(floored[:, band])
        tilt = (level - level.mean(axis=1, keepdims=True)) @ octave / float(octave @ octave)
        tilt = np.where(alive, tilt, 0.0)
    else:
        tilt = np.zeros(power.shape[0])
    prev = np.concatenate([norm[:1] if previous_magnitude is None
                           else np.atleast_2d(previous_magnitude), norm[:-1]], axis=0)
    flux = np.sqrt(np.sum(np.maximum(norm - prev, 0.0) ** 2, axis=1))
    flux = np.where(alive, flux, 0.0)
    if previous_magnitude is None and flux.size:
        flux[0] = 0.0
    out = {"centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff,
           "flatness": flatness, "tilt": tilt, "flux": flux}
    return out, norm[-1:].copy() if norm.shape[0] else previous_magnitude
