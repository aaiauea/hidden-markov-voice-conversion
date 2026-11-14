"""End-to-end extraction: streams, robustness, consistency, persistence, CLI, interop."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hmvc.features import (AcousticFeatures, AnalysisConfig, FeatureExtractor, extract,
                           extract_file, write_wav)
from hmvc.features import cepstrum as C
from hmvc.features.__main__ import main as cli_main
from hmvc.features.descriptors import DESCRIPTORS
from tests._support import feature_signals as S


@pytest.fixture(scope="module")
def mixed():
    x, f0, fs = S.reference_signals()["mixed"]
    return x, f0, fs


@pytest.fixture(scope="module")
def mixed_features(mixed):
    x, _, fs = mixed
    return FeatureExtractor(AnalysisConfig(keep_envelope=True)).extract(x, fs)


def check_streams(feats: AcousticFeatures, n_samples: int, fs: int, cfg=AnalysisConfig()):
    g = cfg.resolve(fs)
    t = 0 if n_samples == 0 else 1 + n_samples // g.hop_length
    assert len(feats) == t and feats.sample_rate == fs
    assert feats.mcep.shape == (t, g.mcep_order + 1)
    assert feats.band_aperiodicity.shape[0] == t
    for name in ("f0", "voiced", "periodicity", "continuous_log_f0", "energy_rms",
                 "energy_db", "speech"):
        assert getattr(feats, name).shape == (t,), name
    assert set(feats.spectral) == set(DESCRIPTORS)
    feats.validate()


def test_stream_layout_and_semantics(mixed, mixed_features):
    x, f0, fs = mixed
    f = mixed_features
    check_streams(f, x.size, fs)
    assert f.voiced.dtype == bool and f.speech.dtype == bool
    assert np.all((f.periodicity >= 0) & (f.periodicity <= 1))
    assert np.all(np.isnan(f.log_f0[~f.voiced])) and np.allclose(np.exp(f.log_f0[f.voiced]),
                                                                  f.f0[f.voiced])
    np.testing.assert_allclose(np.exp(f.continuous_log_f0[f.voiced]), f.f0[f.voiced])
    np.testing.assert_allclose(f.times[:3], [0.0, 0.005, 0.01])
    assert f.frame_period_ms == 5.0 and f.mcep_order == 24
    assert f.duration == pytest.approx(x.size / fs)
    # silent lead-in is not speech; vowels are
    truth = S.frame_f0(f0, f.hop_length)
    assert not f.speech[:8].any() and f.speech[truth > 0].mean() > 0.95


def test_stream_access_and_stacking(mixed_features):
    f = mixed_features
    assert f.stream("centroid") is f.spectral["centroid"]
    assert f.stream("spectral_centroid") is f.spectral["centroid"]
    np.testing.assert_array_equal(f.stream("mcep"), f.mcep)
    m = f.stack(["mcep", "continuous_log_f0", "band_aperiodicity", "voiced", "centroid"])
    assert m.shape == (len(f), 25 + 1 + 5 + 1 + 1) and m.dtype == np.float64
    assert f.stack([]).shape == (len(f), 0)
    with pytest.raises(KeyError):
        f.stream("formants")


def test_kept_and_rebuilt_envelope(mixed_features):
    f = mixed_features
    assert f.envelope() is f.spectral_envelope
    rebuilt = C.mc2sp(f.mcep, f.mcep_alpha, f.fft_size)
    speech = f.speech
    lsd = C.log_spectral_distance(rebuilt[speech], f.spectral_envelope[speech])
    assert np.median(lsd) < 3.0                       # order-24 MCEP is a smoothing
    np.testing.assert_allclose(C.sp2mc(f.spectral_envelope, 24, f.mcep_alpha), f.mcep,
                               atol=1e-9)


@pytest.mark.parametrize("fs", [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000])
def test_every_common_sample_rate(fs):
    sig = S.vowel(fs, 0.3, 150.0, 170.0, 5.0, 20.0, S.FORMANTS_A, -40.0, 0.3, seed=3)
    f = extract(sig["x"], fs)
    check_streams(f, sig["x"].size, fs)
    truth = S.frame_f0(sig["f0"], f.hop_length)
    both = f.voiced & (truth > 0)
    assert both.mean() > 0.9
    assert np.median(1200 * np.abs(np.log2(f.f0[both] / truth[both]))) < 3.0


@pytest.mark.parametrize("cfg", [
    AnalysisConfig(hop_ms=2.5), AnalysisConfig(hop_ms=10.0, frame_ms=40.0),
    AnalysisConfig(hop_length=64, frame_length=512), AnalysisConfig(frame_ms=20.0, window="hamming"),
    AnalysisConfig(window="blackman"), AnalysisConfig(window="rectangular"),
    AnalysisConfig(envelope_method="true_envelope"), AnalysisConfig(envelope_method="lifter"),
    AnalysisConfig(mcep_order=39, mcep_alpha=0.0), AnalysisConfig(n_aperiodicity_bands=1),
    AnalysisConfig(aperiodicity_bands=(0.0, 1000.0, 2000.0, 4000.0, 8000.0)),
    AnalysisConfig(f0_floor=80.0, f0_ceil=400.0), AnalysisConfig(f0_refine_periods=0.0),
    AnalysisConfig(remove_dc=False, min_voiced_ms=0.0, f0_median_frames=1)])
def test_frame_and_analysis_configurations(mixed, cfg):
    x, f0, fs = mixed
    f = FeatureExtractor(cfg).extract(x, fs)
    check_streams(f, x.size, fs, cfg)
    truth = S.frame_f0(f0, f.hop_length)
    assert f.voiced[truth > 0].mean() > 0.8 and f.voiced[:5].sum() == 0


def test_aperiodicity_band_edges_above_nyquist_are_rejected():
    with pytest.raises(ValueError):
        extract(np.zeros(1000), 8000, AnalysisConfig(aperiodicity_bands=(0.0, 2000.0, 8000.0)))


def test_resampling_to_a_target_rate():
    fs = 44100
    sig = S.vowel(fs, 0.3, 200.0, 200.0, 0.0, 0.0, S.FORMANTS_I, None, 0.3, 0)
    f = extract(sig["x"], fs, AnalysisConfig(sample_rate=16000))
    check_streams(f, int(math.ceil(sig["x"].size * 16000 / fs)), 16000)
    assert f.diagnostics["resampled_to"] == 16000
    assert np.median(f.f0[f.voiced]) == pytest.approx(200.0, rel=2e-3)


SR = 16000


@pytest.mark.parametrize("label, signal", [
    ("empty", np.zeros(0)), ("one sample", np.array([0.3])), ("shorter than a frame", 0.1 *
     np.ones(50)), ("silence", np.zeros(SR)), ("denormal", np.full(SR, 1e-310)),
    ("dc", np.full(SR, 0.7)), ("all nan", np.full(SR, np.nan)),
    ("nan bursts", np.where(np.arange(SR) % 1000 < 5, np.nan, 0.1 * np.sin(np.arange(SR) * 0.1))),
    ("inf", np.r_[np.full(10, np.inf), np.zeros(SR)]),
    ("clipped square", np.sign(np.sin(2 * np.pi * 150 * np.arange(SR) / SR))),
    ("impulse", np.r_[np.zeros(SR // 2), 1.0, np.zeros(SR // 2)]),
    ("huge amplitude", 1e4 * np.sin(2 * np.pi * 150 * np.arange(SR) / SR)),
    ("int16", (np.sin(np.arange(SR) * 0.05) * 20000).astype(np.int16)),
    ("stereo", np.stack([np.zeros(SR), 0.1 * np.ones(SR)], axis=1)),
    ("nyquist tone", 0.5 * np.cos(np.pi * np.arange(SR))),
    ("white noise", np.random.default_rng(0).standard_normal(SR)),
])
def test_unusual_input_never_crashes(label, signal):
    f = extract(signal, SR, AnalysisConfig(keep_envelope=True))
    f.validate()
    assert all(np.all(np.isfinite(f.stream(s))) for s in
               ("mcep", "energy_db", "continuous_log_f0", "band_aperiodicity", "periodicity"))
    assert np.all(np.isfinite(f.spectral_envelope)) and np.all(f.spectral_envelope > 0)
    if label in ("empty", "silence", "denormal", "dc", "all nan", "white noise", "impulse"):
        assert not f.voiced.any()
    if label == "silence":
        assert np.allclose(f.mcep[:, 0], 0.5 * np.log(1e-12)) and np.allclose(f.mcep[:, 1:], 0)
        assert np.all(f.energy_db == -100.0)
    if label == "clipped square":
        assert np.median(f.f0[f.voiced]) == pytest.approx(150.0, rel=5e-3)


def test_empty_features_support_stacking_and_normalisation():
    from hmvc.features import FeatureNormalizer
    f = extract(np.zeros(0), SR)
    assert f.stack(["mcep", "f0", "band_aperiodicity"]).shape == (0, 25 + 1 + 5)
    FeatureNormalizer().partial_fit(f)
    assert f.envelope().shape == (0, f.fft_size // 2 + 1)


def test_nonfinite_samples_are_reported():
    f = extract(np.r_[np.nan, np.inf, np.zeros(100)], SR)
    assert f.diagnostics["nonfinite_samples"] == 2


def test_determinism(mixed):
    x, _, fs = mixed
    a = extract(x, fs)
    b = extract(x.copy(), fs)
    for name in ("f0", "voiced", "mcep", "band_aperiodicity", "energy_db"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_block_size_invariance(mixed):
    x, _, fs = mixed
    a = extract(x, fs)
    b = extract(x, fs, AnalysisConfig(block_frames=19))
    np.testing.assert_array_equal(a.voiced, b.voiced)
    for name in ("f0", "mcep", "band_aperiodicity", "energy_rms"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-8, atol=1e-10)
    for name in DESCRIPTORS:
        np.testing.assert_allclose(a.spectral[name], b.spectral[name], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("n", [79, 80, 81, 1599, 1600, 4001, 12345])
def test_variable_lengths(n):
    x = np.random.default_rng(n).standard_normal(n) * 0.05
    f = extract(x, SR)
    assert len(f) == 1 + n // 80
    f.validate()


def test_prefix_consistency_of_steady_signal():
    sig = S.vowel(SR, 1.0, 150.0, 150.0, 0.0, 0.0, S.FORMANTS_A, -50.0, 0.3, 0)
    full = extract(sig["x"], SR)
    half = extract(sig["x"][:SR // 2], SR)
    inner = slice(10, len(half) - 10)
    np.testing.assert_allclose(half.f0[inner], full.f0[inner], rtol=1e-9)
    assert np.max(C.mel_cepstral_distortion(half.mcep[inner], full.mcep[inner])) < 1e-6


# -- persistence -----------------------------------------------------------------------


def test_save_load_round_trip(tmp_path, mixed_features):
    f = mixed_features
    path = tmp_path / "feats" / "u.npz"
    f.save(path)
    g = AcousticFeatures.load(path)
    g.validate()
    for name in ("f0", "voiced", "periodicity", "continuous_log_f0", "energy_rms",
                 "energy_db", "speech", "mcep", "band_aperiodicity", "aperiodicity_edges",
                 "spectral_envelope"):
        np.testing.assert_array_equal(getattr(g, name), getattr(f, name))
    assert g.spectral.keys() == f.spectral.keys() and g.config == f.config
    assert (g.sample_rate, g.hop_length, g.fft_size, g.mcep_alpha) == \
        (f.sample_rate, f.hop_length, f.fft_size, f.mcep_alpha)
    assert AnalysisConfig.from_dict(g.config) == AnalysisConfig(keep_envelope=True)


def test_load_rejects_foreign_files(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, meta=np.array(json.dumps({"format": "something-else"})))
    with pytest.raises(ValueError):
        AcousticFeatures.load(path)
    np.savez(path, meta=np.array(json.dumps({"format": "hmvc.features.acoustic",
                                              "version": 99})))
    with pytest.raises(ValueError):
        AcousticFeatures.load(path)


def test_validate_catches_inconsistency(mixed_features):
    import dataclasses
    f = mixed_features
    bad = [dataclasses.replace(f, f0=f.f0[:-1]),
           dataclasses.replace(f, mcep=np.where(np.arange(f.mcep.size).reshape(f.mcep.shape)
                                                == 3, np.nan, f.mcep)),
           dataclasses.replace(f, f0=np.where(f.voiced, 0.0, f.f0)),
           dataclasses.replace(f, band_aperiodicity=f.band_aperiodicity + 2.0),
           dataclasses.replace(f, n_samples=f.n_samples + 1000)]
    for item in bad:
        with pytest.raises(ValueError):
            item.validate()


def test_extract_file_and_cli(tmp_path, mixed):
    x, _, fs = mixed
    wav = tmp_path / "in.wav"
    write_wav(wav, x, fs, "float32")
    f = extract_file(wav)
    assert f.diagnostics["source"]["format"] == "float32"
    np.testing.assert_allclose(f.f0, extract(x, fs).f0, rtol=1e-6)

    out = tmp_path / "out.npz"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"n_aperiodicity_bands": 3}))
    assert cli_main(["extract", str(wav), str(out), "--config", str(cfg), "--hop-ms", "10",
                     "--mcep-order", "30", "--keep-envelope"]) == 0
    g = AcousticFeatures.load(out)
    assert g.hop_length == 160 and g.mcep_order == 30 and g.band_aperiodicity.shape[1] == 3
    assert g.spectral_envelope is not None
    assert cli_main(["info", str(out)]) == 0


# -- engine interop (optional bridge) ---------------------------------------------------


def test_engine_sequence_and_vocoder_loopback():
    from hmvc.engine.vocoder.builtin import BuiltinVocoder
    from hmvc.engine.vocoder.mlsa import MLSAVocoder
    from hmvc.features.interop import band_to_bin_aperiodicity, to_engine_sequence

    # Steady F0: the engine vocoders have their own frame-timing conventions,
    # so a gliding contour would test their timing rather than the bridge.
    sig = S.vowel(SR, 0.6, 150.0, 150.0, 0.0, 0.0, S.FORMANTS_A, -50.0, 0.3, 4)
    f = extract(sig["x"], SR)
    seq = to_engine_sequence(f)
    assert seq.sp.shape == (len(f), f.fft_size // 2 + 1) and seq.ap.shape == seq.sp.shape
    assert seq.fs == SR and seq.frame_period == 5.0 and np.all((seq.ap >= 0) & (seq.ap <= 1))
    for vocoder in (BuiltinVocoder, MLSAVocoder):
        y = vocoder(fft_size=f.fft_size, fs=SR, frame_period=5.0).synthesize(seq)
        assert np.all(np.isfinite(y)) and abs(y.size - sig["x"].size) <= 2 * f.hop_length
    # analysis -> MLSA synthesis -> analysis preserves pitch and voicing.  (The
    # builtin vocoder's noise excitation of the -93 dB top band, aperiodicity
    # ~1, is audible as jitter to a re-analysis, so it is only smoke-tested.)
    g = extract(y, SR)
    both = f.voiced & g.voiced[:len(f)]
    assert both.sum() > 0.9 * f.voiced.sum()
    assert np.median(1200 * np.abs(np.log2(g.f0[:len(f)][both] / f.f0[both]))) < 5.0
    flat = band_to_bin_aperiodicity(np.array([[0.2]]), np.array([0.0, 8000.0]), SR, 9)
    assert flat.shape == (1, 9) and np.allclose(flat, 0.2)
