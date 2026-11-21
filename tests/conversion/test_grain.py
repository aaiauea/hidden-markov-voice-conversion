"""Grain synth: structure, output bounds, no NaN."""
from __future__ import annotations

import numpy as np

from hmvc.conversion.constants import HOP, SR
from hmvc.conversion.grain import grain_synth
from hmvc.conversion.retrieve import build_db


def _vowel(fs, dur, f0_hz):
    n = int(fs * dur)
    t = np.arange(n) / fs
    phase = 2 * np.pi * f0_hz * t
    x = np.zeros(n)
    for h in range(1, 30):
        x += np.sin(h * phase) / h
    x *= 0.22 / max(np.sqrt(np.mean(x ** 2)), 1e-12)
    return x.astype(np.float32)


def _db_and_marks():
    target = _vowel(22050, 1.0, 190.0)
    db = build_db(target, 70.0, 450.0)
    nfr = int(len(target) / 256) + 1
    from hmvc.conversion.constants import D
    sf = np.zeros(nfr * D, dtype=np.float32)
    rs = np.zeros(nfr, dtype=np.float32)
    for f in range(nfr):
        rs[f] = 0.1
    return db, nfr, sf, rs


def test_grain_synth_runs_and_is_finite():
    target = _vowel(22050, 1.0, 190.0)
    db = build_db(target, 70.0, 450.0)
    nfr = int(len(target) / 256) + 1
    f0 = np.full(nfr, 180.0, dtype=np.float32)
    vo = np.ones(nfr, dtype=np.uint8)
    from hmvc.conversion.psola import make_marks
    from hmvc.conversion.constants import D
    sf = np.zeros(nfr * D, dtype=np.float32)
    rs = np.full(nfr, 0.1, dtype=np.float32)
    nb = np.zeros(nfr * 6, dtype=np.int32).reshape(nfr, 6).astype(np.int32).reshape(-1)
    # match every frame to the first target frame (j=0 valid)
    nb[0::6] = 0
    out = grain_synth(len(target), f0, vo, nb, 6, db, 1.0, rs, nfr, sf)
    assert len(out) == len(target)
    assert np.all(np.isfinite(out))


def test_grain_synth_output_is_bounded():
    target = _vowel(22050, 1.0, 190.0)
    db = build_db(target, 70.0, 450.0)
    nfr = int(len(target) / 256) + 1
    f0 = np.full(nfr, 170.0, dtype=np.float32)
    vo = np.ones(nfr, dtype=np.uint8)
    from hmvc.conversion.constants import D
    sf = np.zeros(nfr * D, dtype=np.float32)
    rs = np.full(nfr, 0.1, dtype=np.float32)
    nb = np.full(nfr * 6, 0, dtype=np.int32)
    out = grain_synth(len(target), f0, vo, nb, 6, db, 1.0, rs, nfr, sf)
    # OLA division by max(ws, 0.35) keeps amplitudes sane-ish
    assert np.max(np.abs(out)) <= 5.0
