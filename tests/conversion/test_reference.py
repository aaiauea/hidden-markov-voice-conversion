"""Parity tests for the HMVC port against frozen conversion reference fixtures.

The checked-in reference data records outputs from the original implementation
on fixed synthetic signals. The original implementation and its generator are
not included in this repository. The Python port must reproduce the recorded
cases bit-for-bit. See tests/data/conversion-reference/README.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hmvc.conversion import ConversionConfig, convert
from hmvc.conversion.pipeline import clear_cache

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "conversion-reference"
SIGNALS = json.loads((DATA / "signals.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((DATA / "reference.json").read_text(encoding="utf-8"))
SIGNAL_SRC = np.array(SIGNALS["signals"]["source"], dtype=np.float32)
SIGNAL_TGT = np.array(SIGNALS["signals"]["target"], dtype=np.float32)

CASES = {
    "env-no-target": dict(mode="env", pitch_semitones=3.0, formant_factor=1.0,
                         target=None, extra={}),
    "env-with-target": dict(mode="env", pitch_semitones=3.0, formant_factor=1.15,
                           target=SIGNAL_TGT, extra={}),
    "grain": dict(mode="grain", pitch_semitones=0.0, formant_factor=1.0,
                  target=SIGNAL_TGT, extra={}),
    "voc": dict(mode="voc", pitch_semitones=0.0, formant_factor=1.0,
                target=SIGNAL_TGT, extra={"rng_seed": 1234}),
}


@pytest.fixture(scope="module", autouse=True)
def _clear_db_cache():
    clear_cache()


def test_fixture_recipe_unchanged():
    recipe = (ROOT / "_support" / "conversion_signals.py").read_bytes()
    assert hashlib.sha256(recipe).hexdigest() == SIGNALS["recipe_sha256"], \
        "signal recipe changed: regenerate the fixture (see tests/data/conversion-reference/README.md)"


def test_signals_geometry():
    assert SIGNALS["sample_rate"] == 22050
    assert SIGNALS["lengths"]["source"] == SIGNAL_SRC.size
    assert SIGNALS["lengths"]["target"] == SIGNAL_TGT.size


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_reproduces_original_bit_for_bit(name):
    spec = CASES[name]
    clear_cache()
    res = convert(
        SIGNAL_SRC,
        spec["target"],
        ConversionConfig(
            mode=spec["mode"],
            pitch_semitones=spec["pitch_semitones"],
            formant_factor=spec["formant_factor"],
            envelope_blend=0.8,
            neighbours=6,
            **spec["extra"],
        ),
    )
    ref = REFERENCE["cases"][name]
    np.testing.assert_array_equal(res.samples, np.array(ref["samples"], dtype=np.float32))
    assert res.pitch_ratio == pytest.approx(ref["ratio"], rel=0, abs=1e-12)
    direct = spec["mode"] in ("grain", "voc")
    assert res.pitch_shifted == (abs(ref["ratio"] - 1.0) > 0.005 and not direct)
    np.testing.assert_array_equal(res.f0, np.array(ref["f0"], dtype=np.float32))
    np.testing.assert_array_equal(res.voiced, np.array(ref["vo"], dtype=np.uint8))
    assert res.source_length == SIGNAL_SRC.size
    assert res.target_length == (SIGNAL_TGT.size if spec["target"] is not None else None)
    assert res.n_frames == ref["nfr"]
    assert res.target_database_active == (ref["db"] is not None)
