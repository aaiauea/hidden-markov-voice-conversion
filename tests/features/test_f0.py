"""F0 / voicing estimation against signals with known F0."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hmvc.features import AnalysisConfig, FeatureExtractor
from hmvc.features import f0 as F0
from tests._support import feature_signals as S

GROSS_CENTS = 1200 * math.log2(1.2)          # 20 % gross pitch error


def cents(est, ref):
    return 1200.0 * np.abs(np.log2(est / ref))


def track(x, fs, **kwargs):
    hop = int(round(0.005 * fs))
    return F0.estimate_f0(x, fs, hop, **kwargs), hop


@pytest.mark.parametrize("f", [65.0, 100.0, 220.0, 440.0, 750.0])
def test_pure_sines(f):
    fs = 16000
    x = 0.5 * np.sin(2 * np.pi * f * np.arange(fs) / fs)
    pitch, _ = track(x, fs)
    inner = slice(10, -10)
    assert pitch.voiced[inner].mean() > 0.95
    assert np.median(cents(pitch.f0[inner][pitch.voiced[inner]], f)) < 2.0
    assert np.all(pitch.periodicity[inner][pitch.voiced[inner]] > 0.9)


def test_missing_fundamental_harmonic_complex():
    fs, f0 = 16000, 150.0
    contour = np.full(fs, f0)
    x = S.harmonic_signal(contour, fs, lambda f: (f > 1.5 * f0).astype(float) * 1.0)
    pitch, _ = track(0.3 * x / np.abs(x).max(), fs)
    v = pitch.f0[10:-10][pitch.voiced[10:-10]]
    assert v.size > 150 and np.median(cents(v, f0)) < 5.0


@pytest.mark.parametrize("fs", [8000, 11025, 16000, 22050, 32000, 44100, 48000])
def test_sample_rates(fs):
    sig = S.vowel(fs, 0.4, 130.0, 170.0, 5.0, 25.0, S.FORMANTS_A, -40.0, 0.3, seed=fs)
    pitch, hop = track(sig["x"], fs)
    truth = S.frame_f0(sig["f0"], hop)
    both = pitch.voiced & (truth > 0)
    assert both.sum() >= 0.95 * (truth > 0).sum()
    c = cents(pitch.f0[both], truth[both])
    assert np.median(c) < 3.0 and np.percentile(c, 95) < 12.0 and c.max() < GROSS_CENTS


@pytest.mark.parametrize("name", ["vowel_low", "vowel_high", "mixed", "vowel_22k"])
def test_reference_signals_accuracy(name):
    x, f0s, fs = S.reference_signals()[name]
    feats = FeatureExtractor().extract(x, fs)
    truth = S.frame_f0(f0s, feats.hop_length)
    tv = truth > 0
    both = feats.voiced & tv
    c = cents(feats.f0[both], truth[both])
    assert both.sum() >= 0.97 * tv.sum()
    assert np.median(c) < 2.5 and np.percentile(c, 95) < 8.0 and c.max() < GROSS_CENTS
    # frames voiced outside the true voiced regions may only touch a boundary
    extra = np.flatnonzero(feats.voiced & ~tv)
    for i in extra:
        assert tv[max(i - 1, 0):i + 2].any(), f"frame {i} voiced away from any voicing"


def test_refinement_improves_vibrato_accuracy():
    x, f0s, fs = S.reference_signals()["vowel_low"]
    errs = {}
    for periods in (0.0, 2.0):
        pitch, hop = track(x, fs, refine_periods=periods)
        truth = S.frame_f0(f0s, hop)
        both = pitch.voiced & (truth > 0)
        errs[periods] = np.median(cents(pitch.f0[both], truth[both]))
    assert errs[2.0] < 0.6 * errs[0.0]


def test_silence_noise_and_fricatives_are_unvoiced():
    fs = 16000
    rng = np.random.default_rng(7)
    for x in (np.zeros(fs), 1e-9 * rng.standard_normal(fs), 0.1 * rng.standard_normal(fs),
              0.5 * rng.standard_normal(fs), S.highpass_noise(fs, fs, 3000.0, 0.1, 3)):
        pitch, _ = track(x, fs)
        assert not pitch.voiced.any()
        assert np.all(pitch.f0 == 0.0) and np.all(np.isfinite(pitch.periodicity))


def test_quiet_but_clean_voicing_is_kept():
    fs = 16000
    sig = S.vowel(fs, 0.4, 150.0, 150.0, 0.0, 0.0, S.FORMANTS_A, None, 10 ** (-60 / 20), 0)
    pitch, _ = track(sig["x"], fs)                     # -60 dBFS, above the -80 dB gate
    assert pitch.voiced[10:-10].mean() > 0.95


def test_relative_silence_gate():
    fs = 16000
    loud = S.vowel(fs, 0.3, 150.0, 150.0, 0.0, 0.0, S.FORMANTS_A, None, 0.3, 0)["x"]
    faint = loud * 10 ** (-60 / 20)                    # 60 dB below the loud part
    pitch, hop = track(np.concatenate([loud, faint]), fs)
    n = loud.size // hop
    assert pitch.voiced[5:n - 5].mean() > 0.95 and not pitch.voiced[n + 5:].any()


def test_tracking_and_hysteresis_can_be_disabled():
    x, f0s, fs = S.reference_signals()["mixed"]
    base = FeatureExtractor(AnalysisConfig(f0_jump_cost=0.0, f0_max_jump=0.0,
                                           voicing_extension_threshold=0.25)).extract(x, fs)
    full = FeatureExtractor().extract(x, fs)
    assert base.voiced.sum() <= full.voiced.sum()
    assert base.voiced[full.voiced].mean() > 0.9        # same regions, hysteresis only grows


def test_grow_respects_eligibility_and_continuity():
    seed = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    eligible = np.array([1, 1, 1, 1, 1, 0, 1], bool)
    log_f = np.log([100, 100, 100, 105, 200, 200, 200])
    grown = F0._grow(seed, eligible, log_f, 0.2)
    assert grown.tolist() == [True, True, True, True, False, False, False]
    assert F0._grow(seed, eligible, log_f, 0.0).tolist() == seed.tolist()


def test_viterbi_prefers_continuous_path():
    periods = np.array([[100.0, 50.0], [50.0, 100.0], [100.0, 50.0]])
    costs = np.array([[0.1, 0.2], [0.05, 0.1], [0.1, 0.2]])   # middle frame: octave error
    np.testing.assert_allclose(F0._viterbi(periods, costs, 2.0), [100.0, 100.0, 100.0])
    np.testing.assert_allclose(F0._viterbi(periods, costs, 0.0), [100.0, 50.0, 100.0])


def test_block_size_invariance():
    x, _, fs = S.reference_signals()["mixed"]
    a, _ = track(x, fs, block=512)
    b, _ = track(x, fs, block=37)
    np.testing.assert_array_equal(a.voiced, b.voiced)
    np.testing.assert_allclose(a.f0, b.f0, rtol=1e-10, atol=0)


def test_degenerate_lengths():
    for n in (0, 1, 10, 79, 80, 300):
        pitch, hop = track(np.ones(n) * 0.1, 16000)
        expected = 0 if n == 0 else 1 + n // hop
        assert pitch.f0.shape == (expected,) and not pitch.voiced.any()


def test_continuous_log_f0():
    f0 = np.array([0.0, 100.0, 0.0, 0.0, 400.0, 0.0])
    lf = F0.continuous_log_f0(f0, f0 > 0, 200.0)
    np.testing.assert_allclose(np.exp(lf), [100, 100, 158.74, 251.98, 400, 400], rtol=1e-4)
    np.testing.assert_allclose(F0.continuous_log_f0(np.zeros(3), np.zeros(3, bool), 200.0),
                               np.log(200.0))
    assert F0.continuous_log_f0(np.zeros(0), np.zeros(0, bool), 200.0).size == 0


@pytest.mark.parametrize("fs", [8000, 16000, 44100])
def test_dense_pure_tone_sweep_has_no_octave_errors(fs):
    # Regression: a period crowded out of the candidate list by its own
    # multiples (which fall closer to integer lags) produced 2T / 3T tracks.
    hop = int(round(0.005 * fs))
    for f in np.geomspace(62.0, 780.0, 40):
        x = 0.5 * np.sin(2 * np.pi * f * np.arange(fs // 4) / fs)
        pitch = F0.estimate_f0(x, fs, hop)
        v = pitch.f0[8:-8][pitch.voiced[8:-8]]
        assert v.size >= 0.95 * (pitch.f0.size - 16), f
        assert np.max(cents(v, f)) < 20.0, f
