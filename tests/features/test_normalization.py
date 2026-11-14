"""Corpus / utterance feature normalisation."""
from __future__ import annotations

import numpy as np
import pytest

from hmvc.features import AnalysisConfig, FeatureNormalizer, extract, normalize_utterance
from hmvc.features.normalization import RunningMoments
from tests._support import feature_signals as S


@pytest.fixture(scope="module")
def corpus():
    sigs = [S.vowel(16000, 0.3, 110.0 + 40 * i, 130.0 + 40 * i, 5.0, 20.0,
                    (S.FORMANTS_A, S.FORMANTS_I)[i % 2], -40.0, 0.2 + 0.05 * i, seed=i)
            for i in range(4)]
    mixed = S.mixed_utterance(16000, seed=9)
    return [extract(s["x"], 16000) for s in sigs] + [extract(mixed["x"], 16000)]


def test_running_moments_merge_equals_batch():
    rng = np.random.default_rng(0)
    parts = [rng.standard_normal((n, 3)) * 5 + 100 for n in (1, 7, 0, 300, 2)]
    merged = RunningMoments.empty(3)
    for p in parts:
        merged = merged.merge(RunningMoments.from_values(p))
    allv = np.concatenate(parts)
    assert merged.count == len(allv)
    np.testing.assert_allclose(merged.mean, allv.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(merged.variance, allv.var(axis=0), rtol=1e-10)


def test_fit_matches_pooled_statistics_and_frame_selection(corpus):
    norm = FeatureNormalizer().fit(corpus)
    mcep = np.concatenate([f.mcep[f.speech] for f in corpus])
    lf0 = np.concatenate([f.continuous_log_f0[f.voiced] for f in corpus])
    np.testing.assert_allclose(norm.mean("mcep"), mcep.mean(axis=0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(norm.std("mcep"), mcep.std(axis=0), rtol=1e-9)
    np.testing.assert_allclose(norm.mean("continuous_log_f0"), [lf0.mean()], rtol=1e-12)
    # incremental fitting is order independent
    rev = FeatureNormalizer().fit(corpus[::-1])
    np.testing.assert_allclose(rev.mean("mcep"), norm.mean("mcep"), rtol=1e-12, atol=1e-14)


def test_transform_round_trip_and_zscore(corpus):
    norm = FeatureNormalizer().fit(corpus)
    out = norm.transform(corpus[0])
    assert set(out) == set(norm.streams)
    pooled = np.concatenate([norm.transform(f)["mcep"][f.speech] for f in corpus])
    np.testing.assert_allclose(pooled.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(pooled.std(axis=0), 1.0, rtol=1e-8)
    np.testing.assert_allclose(norm.denormalize("mcep", out["mcep"]), corpus[0].mcep,
                               atol=1e-10)


def test_log_f0_nan_is_preserved_and_constant_dims_are_safe(corpus):
    norm = FeatureNormalizer(streams=("log_f0", "voiced")).fit(corpus)
    out = norm.transform(corpus[-1])
    assert np.all(np.isnan(out["log_f0"][~corpus[-1].voiced]))
    assert np.all(np.isfinite(out["log_f0"][corpus[-1].voiced]))
    const = FeatureNormalizer(streams=("speech",)).fit(corpus)   # all-True on speech frames
    assert np.all(np.isfinite(const.transform(corpus[0])["speech"]))


def test_save_load_and_config_guard(tmp_path, corpus):
    norm = FeatureNormalizer().fit(corpus)
    path = tmp_path / "stats.json"
    norm.save(path)
    loaded = FeatureNormalizer.load(path)
    assert loaded.geometry == norm.geometry
    for name in norm.streams:
        np.testing.assert_allclose(loaded.mean(name), norm.mean(name), rtol=1e-15)
        np.testing.assert_allclose(loaded.std(name), norm.std(name), rtol=1e-15)
    other = extract(np.zeros(1600), 16000, AnalysisConfig(hop_ms=10.0))
    with pytest.raises(ValueError):
        loaded.transform(other)
    with pytest.raises(ValueError):
        FeatureNormalizer().fit(corpus).partial_fit(other)
    with pytest.raises(ValueError):
        FeatureNormalizer.from_dict({"format": "nope"})
    with pytest.raises(KeyError):
        FeatureNormalizer().mean("mcep")



def test_normalizer_rejects_mixed_native_sample_rates():
    # With sample_rate=None the requested config is identical, and both rates
    # use 25 MCEP coefficients, but their resolved warping and band grids differ.
    config = AnalysisConfig()
    at_16k = extract(np.zeros(1600), 16000, config)
    at_11025 = extract(np.zeros(1103), 11025, config)
    assert at_16k.config == at_11025.config
    assert at_16k.mcep.shape[1] == at_11025.mcep.shape[1]
    assert at_16k.mcep_alpha != at_11025.mcep_alpha

    normalizer = FeatureNormalizer(streams=("mcep",)).partial_fit(at_16k)
    with pytest.raises(ValueError, match="resolved analysis geometry"):
        normalizer.partial_fit(at_11025)
    with pytest.raises(ValueError, match="resolved analysis geometry"):
        normalizer.transform(at_11025)


def test_utterance_normalisation(corpus):
    out = normalize_utterance(corpus[1])
    sel = corpus[1].speech
    np.testing.assert_allclose(out["mcep"][sel].mean(axis=0), 0.0, atol=1e-9)


def test_silent_utterance_does_not_break_fitting(corpus):
    silent = extract(np.zeros(8000), 16000)
    norm = FeatureNormalizer().fit(corpus + [silent])
    ref = FeatureNormalizer().fit(corpus)
    np.testing.assert_allclose(norm.mean("mcep"), ref.mean("mcep"))    # no speech frames
