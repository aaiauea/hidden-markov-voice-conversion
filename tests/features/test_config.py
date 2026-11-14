"""AnalysisConfig validation, geometry resolution and serialisation."""
from __future__ import annotations

import json

import pytest

from hmvc.features import AnalysisConfig, default_mcep_order
from hmvc.features.cepstrum import mcep_alpha


def test_defaults_resolve_to_documented_geometry():
    g = AnalysisConfig().resolve(16000)
    assert (g.hop_length, g.frame_length, g.fft_size) == (80, 400, 1024)
    assert (g.mcep_order, round(g.mcep_alpha, 3)) == (24, 0.41)
    assert g.n_bins == 513 and g.frame_period_ms == 5.0


@pytest.mark.parametrize("fs, fft, order, alpha", [
    (8000, 512, 24, 0.312), (11025, 1024, 24, 0.357), (16000, 1024, 24, 0.41),
    (22050, 2048, 34, 0.455), (24000, 2048, 34, 0.466), (32000, 2048, 39, 0.504),
    (44100, 4096, 39, 0.544), (48000, 4096, 39, 0.554)])
def test_rate_dependent_defaults(fs, fft, order, alpha):
    g = AnalysisConfig().resolve(fs)
    assert g.fft_size == fft
    assert g.mcep_order == order == default_mcep_order(fs)
    assert g.mcep_alpha == pytest.approx(alpha, abs=1e-12)
    assert mcep_alpha(fs) == pytest.approx(alpha, abs=1e-12)
    assert g.hop_length == round(0.005 * fs)
    # the FFT must hold three periods of the lowest F0 (CheapTrick requirement)
    assert g.fft_size >= 3 * fs / g.f0_floor


def test_explicit_lengths_override_milliseconds():
    g = AnalysisConfig(hop_length=100, frame_length=512, fft_size=2048,
                       mcep_order=30, mcep_alpha=0.3).resolve(16000)
    assert (g.hop_length, g.frame_length, g.fft_size, g.mcep_order, g.mcep_alpha) == \
        (100, 512, 2048, 30, 0.3)


@pytest.mark.parametrize("changes", [
    {"hop_ms": 0.0}, {"hop_ms": -5.0}, {"frame_ms": 0.0}, {"hop_length": 0},
    {"f0_floor": 0.0}, {"f0_floor": 500.0, "f0_ceil": 400.0},
    {"window": "triangle-ish"}, {"envelope_method": "lpc"},
    {"voicing_threshold": 1.5}, {"yin_threshold": 0.0},
    {"voicing_threshold": 0.5, "voicing_extension_threshold": 0.4},
    {"f0_max_jump": -0.1}, {"f0_jump_cost": -1.0}, {"f0_refine_periods": 0.5},
    {"f0_refine_periods": 9.0}, {"mcep_alpha": 1.0}, {"mcep_order": 0},
    {"rolloff_percent": 1.5}, {"power_floor": 0.0}, {"n_aperiodicity_bands": 0},
    {"aperiodicity_bands": (0.0, 3000.0, 1000.0)}, {"block_frames": 0},
    {"sample_rate": 100}])
def test_invalid_settings_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        AnalysisConfig(**changes)


def test_fft_size_must_hold_the_frame():
    with pytest.raises(ValueError):
        AnalysisConfig(frame_length=2048, fft_size=1024).resolve(16000)


def test_f0_ceiling_above_nyquist_is_rejected_at_resolution():
    with pytest.raises(ValueError):
        AnalysisConfig(f0_ceil=5000.0).resolve(8000)


def test_dict_and_json_round_trip():
    cfg = AnalysisConfig(sample_rate=22050, hop_ms=10.0, aperiodicity_bands=(0, 1000, 4000),
                         tilt_range_hz=(50.0, 4000.0), keep_envelope=True)
    data = cfg.to_dict()
    json.dumps(data)                                   # plain JSON types only
    assert AnalysisConfig.from_dict(data) == cfg
    assert AnalysisConfig.from_json(cfg.to_json()) == cfg


def test_unknown_keys_are_rejected():
    with pytest.raises((ValueError, TypeError)):
        AnalysisConfig.from_dict({"hop_ms": 5.0, "not_an_option": 1})


def test_replace_validates_and_is_frozen():
    cfg = AnalysisConfig()
    assert cfg.replace(hop_ms=10.0).hop_ms == 10.0 and cfg.hop_ms == 5.0
    with pytest.raises(ValueError):
        cfg.replace(hop_ms=-1.0)
    with pytest.raises(Exception):
        cfg.hop_ms = 3.0                                # type: ignore[misc]
