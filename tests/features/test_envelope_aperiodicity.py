"""Spectral envelopes (CheapTrick, true envelope, lifter) and band aperiodicity."""
from __future__ import annotations

import numpy as np
import pytest

from hmvc.features import aperiodicity as A
from hmvc.features import cepstrum as C
from hmvc.features import envelope as E
from hmvc.features.framing import frame_signal, get_window, power_spectrum
from tests._support import feature_signals as S

FS, HOP, NFFT = 16000, 80, 1024
FREQS = np.linspace(0, FS / 2, NFFT // 2 + 1)
BAND = (FREQS >= 80) & (FREQS <= 0.9 * FS / 2)


def steady(f0, formants=S.FORMANTS_A, noise_db=None):
    sig = S.vowel(FS, 0.5, f0, f0, 0.0, 0.0, formants, noise_db, 0.3, 0)
    return sig["x"], S.frame_f0(sig["f0"], HOP)


def shape_error(est, formants=S.FORMANTS_A):
    true = S.formant_amplitude(FREQS, FS, formants) ** 2
    est = est[10:-10]
    return float(np.mean(C.log_spectral_distance(est[:, BAND],
                                                 np.broadcast_to(true, est.shape)[:, BAND],
                                                 remove_gain=True)))


@pytest.mark.parametrize("f0, limit", [(100.0, 1.2), (150.0, 1.5), (220.0, 2.0), (300.0, 3.0)])
@pytest.mark.parametrize("formants", [S.FORMANTS_A, S.FORMANTS_I])
def test_cheaptrick_recovers_the_known_envelope(f0, limit, formants):
    x, f0_frames = steady(f0, formants)
    env = E.cheaptrick(x, FS, f0_frames, HOP, NFFT)
    assert env.shape == (len(f0_frames), NFFT // 2 + 1) and np.all(env > 0)
    assert shape_error(env, formants) < limit


def test_cheaptrick_is_f0_invariant():
    shapes = []
    for f0 in (100.0, 150.0, 220.0, 300.0):
        x, f0_frames = steady(f0)
        logm = 10 * np.log10(E.cheaptrick(x, FS, f0_frames, HOP, NFFT)[10:-10]).mean(axis=0)
        shapes.append(logm[BAND] - logm[BAND].mean())
    assert np.mean(np.std(np.stack(shapes), axis=0)) < 1.2          # dB


def test_cheaptrick_beats_fixed_lifter_and_matches_true_envelope_ordering():
    x, f0_frames = steady(220.0)
    power = power_spectrum(frame_signal(x, 640, HOP), get_window("hann", 640), NFFT)
    ct = shape_error(E.cheaptrick(x, FS, f0_frames, HOP, NFFT))
    te = shape_error(E.true_envelope(power, FS, f0_frames))
    lf = shape_error(E.lifter_envelope(power, FS, f0_frames))
    assert ct < te < lf


def test_cheaptrick_block_and_frame_range_invariance():
    x, f0s, _ = S.reference_signals()["mixed"]
    f0_frames = S.frame_f0(f0s, HOP)
    full = E.cheaptrick(x, FS, f0_frames, HOP, NFFT)
    # identical up to FFT round-off (batched transforms of different shapes)
    np.testing.assert_allclose(E.cheaptrick(x, FS, f0_frames, HOP, NFFT, block=29), full,
                               rtol=1e-9, atol=1e-14)
    part = E.cheaptrick_frames(x, FS, f0_frames[40:60], 40, HOP, NFFT)
    np.testing.assert_allclose(part, full[40:60], rtol=1e-9, atol=1e-14)


def test_envelopes_of_silence_and_unvoiced_frames_are_finite():
    zeros = np.zeros(4000)
    f0 = np.zeros(51)
    env = E.cheaptrick(zeros, FS, f0, HOP, NFFT)
    assert np.all(np.isfinite(env)) and np.allclose(env, 1e-12)
    power = np.zeros((5, NFFT // 2 + 1))
    for fn in (E.true_envelope, E.lifter_envelope):
        out = fn(power, FS, np.zeros(5))
        assert np.all(np.isfinite(out)) and np.all(out > 0)
    noise = np.random.default_rng(0).standard_normal(4000) * 0.1
    env = E.cheaptrick(noise, FS, f0, HOP, NFFT)                    # unvoiced: 500 Hz window
    assert np.all(np.isfinite(env)) and np.all(env > 0)


def test_true_envelope_is_an_upper_envelope():
    x, f0_frames = steady(150.0)
    power = power_spectrum(frame_signal(x, 640, HOP), get_window("hann", 640), NFFT)[10:-10]
    env = E.true_envelope(power, FS, f0_frames[10:-10])
    above = 10 * np.log10(power[:, BAND] / env[:, BAND])
    assert np.percentile(above, 99) < 1.5                         # threshold 1 dB


def test_adaptive_order_and_f0_limit():
    orders = E.adaptive_order(FS, np.array([0.0, 100.0, 200.0, 400.0]), NFFT // 2 + 1)
    assert orders[1] > orders[2] > orders[3]
    assert E.cheaptrick_f0_limit(FS, 1024) < 60.0 < E.cheaptrick_f0_limit(FS, 256)


# -- aperiodicity --------------------------------------------------------------------


def test_band_edges():
    edges = A.band_edges(5, FS)
    assert edges[0] == 0 and edges[-1] == pytest.approx(FS / 2) and np.all(np.diff(edges) > 0)
    widths = np.diff(edges)
    assert np.all(np.diff(widths) > 0)                             # mel spacing
    assert A.mel_to_hz(A.hz_to_mel(1234.5)) == pytest.approx(1234.5)


@pytest.mark.parametrize("contour", ["steady", "vibrato"])
@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.3, 0.5, 0.8])
def test_band_aperiodicity_measures_noise_fraction(contour, fraction):
    n = int(0.6 * FS)
    t = np.arange(n) / FS
    f0 = np.full(n, 150.0) if contour == "steady" else \
        150.0 * 2 ** (50 / 1200 * np.sin(2 * np.pi * 5.5 * t))
    harm = S.harmonic_signal(f0, FS, lambda f: np.ones_like(f))
    harm /= np.sqrt(np.mean(harm ** 2))
    noise = np.random.default_rng(5).standard_normal(n)
    x = 0.2 * (np.sqrt(1 - fraction) * harm + np.sqrt(fraction) * noise)
    edges = A.band_edges(5, FS)
    ap = A.band_aperiodicity(x, FS, S.frame_f0(f0, HOP), HOP, edges)[10:-10, :4]
    assert np.mean(ap) == pytest.approx(fraction, abs=0.05)
    assert np.all((ap >= 0) & (ap <= 1))


def test_aperiodicity_of_unvoiced_and_silent_frames_is_one():
    edges = A.band_edges(5, FS)
    x = np.random.default_rng(1).standard_normal(4000) * 0.1
    ap = A.band_aperiodicity(x, FS, np.zeros(51), HOP, edges)
    assert ap.shape == (51, 5) and np.all(ap == 1.0)
    voiced_silence = A.band_aperiodicity(np.zeros(4000), FS, np.full(51, 150.0), HOP, edges)
    assert np.all(voiced_silence == 1.0)


def test_aperiodicity_block_invariance():
    x, f0s, _ = S.reference_signals()["mixed"]
    f0_frames = S.frame_f0(f0s, HOP)
    edges = A.band_edges(5, FS)
    full = A.band_aperiodicity(x, FS, f0_frames, HOP, edges)
    np.testing.assert_allclose(A.band_aperiodicity(x, FS, f0_frames, HOP, edges, block=17),
                               full, rtol=1e-9, atol=1e-12)
    # a frame's value does not depend on which other frames are analysed
    alone = f0_frames.copy()
    alone[np.arange(alone.size) != 30] = 0.0
    np.testing.assert_allclose(A.band_aperiodicity(x, FS, alone, HOP, edges)[30], full[30],
                               rtol=1e-9, atol=1e-12)
