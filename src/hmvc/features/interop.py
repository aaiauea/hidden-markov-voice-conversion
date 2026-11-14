"""Optional bridge from HMVC features to the extracted HMS engine.

The feature package does not import :mod:`hmvc.engine` anywhere else.  This
adapter converts :class:`~hmvc.features.extractor.AcousticFeatures` into the
engine's WORLD-style :class:`~hmvc.engine.core.features.AcousticFrameSequence`
(``f0``, linear power ``sp``, per-bin ``ap``) so the engine's pure-NumPy
vocoders (``builtin``/``mlsa``) can render them -- used for analysis/synthesis
loop-back checks, and the seam later phases will synthesise through.

The CheapTrick envelope follows WORLD's power scale, which is the convention
the engine's vocoders consume.
"""

from __future__ import annotations

import numpy as np


def band_to_bin_aperiodicity(band_ap: np.ndarray, edges: np.ndarray, sample_rate: int,
                             n_bins: int) -> np.ndarray:
    """Interpolate ``(T, B)`` band aperiodicity onto ``n_bins`` linear bins."""
    band_ap = np.atleast_2d(np.asarray(band_ap, dtype=np.float64))
    edges = np.asarray(edges, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    freqs = np.linspace(0.0, sample_rate / 2.0, int(n_bins))
    if centres.size == 1:
        return np.repeat(band_ap, n_bins, axis=1)
    out = np.empty((band_ap.shape[0], int(n_bins)))
    for t in range(band_ap.shape[0]):
        out[t] = np.interp(freqs, centres, band_ap[t])
    return np.clip(out, 0.0, 1.0)


def to_engine_sequence(features, frame_period_ms: float | None = None):
    """Build an engine ``AcousticFrameSequence`` from extracted features."""
    from hmvc.engine.core.features import AcousticFrameSequence

    sp = features.envelope()
    ap = band_to_bin_aperiodicity(features.band_aperiodicity, features.aperiodicity_edges,
                                  features.sample_rate, sp.shape[1])
    return AcousticFrameSequence(
        f0=np.asarray(features.f0, dtype=np.float64).copy(), sp=sp, ap=ap,
        frame_period=float(frame_period_ms or features.frame_period_ms),
        fs=int(features.sample_rate), fft_size=int(features.fft_size))
