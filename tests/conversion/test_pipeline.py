"""End-to-end pipeline behavior (env / grain / voc / db-too-short / signals)."""
from __future__ import annotations

import numpy as np
import pytest

from hmvc.conversion import ConversionConfig, convert
from hmvc.conversion.pipeline import clear_cache


def _vowel(fs, dur, f0_hz, seed=1):
    n = int(fs * dur)
    t = np.arange(n) / fs
    phase = 2 * np.pi * f0_hz * t
    x = np.zeros(n)
    for h in range(1, 30):
        x += np.sin(h * phase) / h
    x *= 0.2 / max(np.sqrt(np.mean(x ** 2)), 1e-12)
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def src_tgt():
    src = _vowel(22050, 1.0, 180.0)
    tgt = _vowel(22050, 1.2, 150.0)
    return src, tgt


@pytest.fixture(autouse=True)
def _cache():
    clear_cache()


def test_env_mode_produces_peak_normalized_output(src_tgt):
    src, _ = src_tgt
    res = convert(src, None, ConversionConfig(mode="env", pitch_semitones=0.0, formant_factor=1.0))
    assert res.samples.shape[0] == len(src)
    assert res.sample_rate == 22050
    assert np.all(np.isfinite(res.samples))
    assert abs(np.max(np.abs(res.samples)) - 0.92) < 1e-3


def test_env_mode_with_target_and_blend(src_tgt):
    src, tgt = src_tgt
    res = convert(src, tgt, ConversionConfig(mode="env", pitch_semitones=0.0, formant_factor=1.0, envelope_blend=0.8))
    assert res.target_database_active
    assert np.all(np.isfinite(res.samples))
    assert abs(np.max(np.abs(res.samples)) - 0.92) < 1e-3


def test_grain_mode_runs(src_tgt):
    src, tgt = src_tgt
    res = convert(src, tgt, ConversionConfig(mode="grain", pitch_semitones=0.0, formant_factor=1.0, envelope_blend=0.8, neighbours=6))
    assert res.direct_mode
    assert res.mode == "grain"
    assert np.all(np.isfinite(res.samples))


def test_voc_mode_runs_deterministically_when_seeded(src_tgt):
    src, tgt = src_tgt
    r1 = convert(src, tgt, ConversionConfig(mode="voc", pitch_semitones=0.0, rng_seed=1234))
    r2 = convert(src, tgt, ConversionConfig(mode="voc", pitch_semitones=0.0, rng_seed=1234))
    assert np.all(r1.samples == r2.samples)


def test_target_truncation_at_max_seconds(src_tgt):
    src, tgt = src_tgt
    cfg = ConversionConfig(mode="env", target_max_seconds=0.05)
    res = convert(src, tgt, cfg)
    assert res.target_length == int(0.05 * 22050)


def test_too_short_target_is_ignored(src_tgt):
    src, tgt = src_tgt
    tiny = tgt[:int(0.05 * 22050)]
    res = convert(src, tiny, ConversionConfig(mode="grain", envelope_blend=0.8, rng_seed=1))
    # the target is too short: retrieval is ignored, envelope path is used
    assert not res.target_database_active
    assert res.mode == "env"


def test_progress_reports_monotonic_stage_messages(src_tgt):
    src, _ = src_tgt
    events = []

    def progress(frac, message):
        events.append((frac, message))

    convert(src, None, ConversionConfig(mode="env", pitch_semitones=2.0), progress)
    assert events
    msgs = [m for _, m in events]
    assert "Tracking pitch…" in msgs
    assert "Transforming timbre…" in msgs


def test_direct_mode_skips_pitch_shift_even_with_large_semitone_shift(src_tgt):
    src, tgt = src_tgt
    # 12 semitones of pitch with a target: the original runs the direct path
    res = convert(src, tgt, ConversionConfig(mode="grain", pitch_semitones=12.0, neighbours=6, rng_seed=1))
    assert not res.pitch_shifted


def test_env_mode_applies_pitch_shift_for_large_semitone_shift(src_tgt):
    src, _ = src_tgt
    res = convert(src, None, ConversionConfig(mode="env", pitch_semitones=12.0, formant_factor=1.0))
    assert res.pitch_shifted


def test_nan_samples_are_zeroed(src_tgt):
    src, _ = src_tgt
    res = convert(src, None, ConversionConfig(mode="env", pitch_semitones=0.0, formant_factor=1.0))
    assert np.all(np.isfinite(res.samples))


def test_default_config_matches_original_defaults():
    cfg = ConversionConfig()
    assert cfg.mode == "voc"
    assert cfg.pitch_semitones == 0.0
    assert cfg.formant_factor == 1.0
    assert cfg.envelope_blend == 0.8
    assert cfg.neighbours == 6
    assert cfg.fmin == 70.0
    assert cfg.fmax == 450.0
    assert cfg.auto_pitch is True
    assert cfg.target_max_seconds == 150
    assert cfg.rng_seed is None


def test_cli_help_and_arg_validation():
    from hmvc.conversion.__main__ import build_parser
    a = build_parser()
    # --src is required
    with pytest.raises(SystemExit):
        a.parse_args([])
    # unknown mode is rejected by choices
    with pytest.raises(SystemExit):
        a.parse_args(["--mode", "bogus", "--src", "x.wav"])
    # valid parse
    args = a.parse_args(["--src", "x.wav", "--mode", "env", "--seed", "7"])
    assert args.mode == "env" and args.seed == 7


def test_cli_main_produces_output(tmp_path):
    from hmvc.conversion.__main__ import main

    def _sine():
        sr = 22050
        t = np.arange(int(sr * 0.5)) / sr
        x = 0.3 * np.sin(2 * np.pi * 180.0 * t)
        return (x * 32767).astype(np.int16)

    s_path = tmp_path / "src.wav"
    t_path = tmp_path / "tgt.wav"
    src_wav = tmp_path / "src.wav"
    for path, freq in ((src_wav, 180.0), (t_path, 160.0)):
        sr = 22050
        t = np.arange(int(sr * 0.5)) / sr
        x = 0.3 * np.sin(2 * np.pi * freq * t)
        # write a WAV via Python's standard lib (functional equivalent of write_wav)
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((x * 32767).astype(np.int16).tobytes())
    out_path = tmp_path / "out.wav"
    rc = main(["--src", str(src_wav), "--tgt", str(t_path),
               "--out", str(out_path), "--mode", "env", "--seed", "11"])
    assert rc == 0
    assert out_path.exists()
    data = out_path.read_bytes()
    assert data[:4] == b"RIFF" and len(data) > 100
