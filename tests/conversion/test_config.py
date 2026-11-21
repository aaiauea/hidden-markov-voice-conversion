"""Config and preset behavior (the original UI knobs)."""
from __future__ import annotations

import pytest

from hmvc.conversion import (
    PRESETS, VOICE_RANGES, ConversionConfig, preset_params,
)


def test_defaults_match_ui():
    cfg = ConversionConfig()
    assert cfg.mode == "voc"
    assert cfg.pitch_semitones == 0.0
    assert cfg.formant_factor == 1.0
    assert cfg.envelope_blend == 0.8
    assert cfg.neighbours == 6
    assert (cfg.fmin, cfg.fmax) == VOICE_RANGES["standard"]
    assert cfg.auto_pitch is True
    assert cfg.target_max_seconds == 150.0


def test_voice_ranges():
    assert VOICE_RANGES["low"] == (60.0, 300.0)
    assert VOICE_RANGES["standard"] == (70.0, 450.0)
    assert VOICE_RANGES["high"] == (120.0, 700.0)


def test_presets():
    assert preset_params("Deeper voice") == (-5.0, 0.88, 0.8)
    assert preset_params("Higher voice") == (5.0, 1.15, 0.8)
    assert preset_params("Child") == (8.0, 1.3, 0.8)
    assert preset_params("Giant") == (-10.0, 0.75, 0.8)
    assert preset_params("Match target only") == (0.0, 1.0, 1.0)
    assert preset_params("Custom") is None
    with pytest.raises(ValueError):
        preset_params("Not a preset")


def test_config_validation():
    ConversionConfig(pitch_semitones=12.0, formant_factor=0.6)
    ConversionConfig(envelope_blend=0.0)
    ConversionConfig(neighbours=16)
    with pytest.raises(ValueError):
        ConversionConfig(mode="paint")
    with pytest.raises(ValueError):
        ConversionConfig(pitch_semitones=12.5)
    with pytest.raises(ValueError):
        ConversionConfig(formant_factor=2.0)
    with pytest.raises(ValueError):
        ConversionConfig(envelope_blend=1.5)
    with pytest.raises(ValueError):
        ConversionConfig(neighbours=0)
    with pytest.raises(ValueError):
        ConversionConfig(fmin=800.0, fmax=100.0)
    with pytest.raises(ValueError):
        ConversionConfig(target_max_seconds=0.0)
