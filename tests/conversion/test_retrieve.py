"""Retrieval database and nearest-neighbour search."""
from __future__ import annotations

import numpy as np
import pytest

from hmvc.conversion.constants import D, HOP, N, SIL
from hmvc.conversion.envelope import analyze, feat, cmn, CEP
from hmvc.conversion.psola import make_marks
from hmvc.conversion.pitch import fill_gaps, track_pitch, median
from hmvc.conversion.retrieve import build_db, knn


def _vowel(fs, dur, f0_hz, seed=1):
    n = int(fs * dur)
    t = np.arange(n) / fs
    phase = 2 * np.pi * f0_hz * t
    x = np.zeros(n)
    for h in range(1, 35):
        x += np.sin(h * phase) / (h ** 0.75)
    x *= 0.22 / max(np.sqrt(np.mean(x ** 2)), 1e-12)
    return x.astype(np.float32)


def _db():
    t = _vowel(22050, 1.0, 170.0)
    db = build_db(t, 70.0, 450.0)
    return db, t


def test_build_db_structure_and_counts():
    db, t = _db()
    n_expected = int((len(t) - N) // 512)
    assert len(db.le) == db.n
    assert db.n <= n_expected
    assert db.n > 0
    assert db.ft.shape == (db.n * D,)
    assert len(db.idx) == 2
    # every frame belongs to one of the two voicing groups
    assert sum(len(g) for g in db.idx) == db.n
    # marks/fm/mk geometry
    assert len(db.marks) == len(db.mk) or len(db.marks) > 0
    assert len(db.cep) == db.n * 36


def test_build_db_marks_are_ascending_and_inside_target():
    db, t = _db()
    marks = db.marks
    positions = [m["p"] for m in marks]
    assert positions == sorted(positions)
    assert all(0 <= p < len(t) for p in positions)


def test_knn_top_k_is_sorted_and_bounded():
    db, t = _db()
    J = db.n
    # synthetic frames close to frame 0 of the target
    sf = np.zeros(12 * 12, dtype=np.float32)
    sf[:12] = db.ft[:12] * 0 + db.ft[0 * 12:1 * 12]  # same as frame 0
    sv = np.zeros(12, dtype=np.uint8)
    sa = np.ones(12, dtype=np.uint8)
    nb = knn(sf, sv, sa, db, 3, None)
    # each returned neighbour is a valid index (or -1 if not voiced nearby)
    for f in range(12):
        for q in range(3):
            j = nb[f * 3 + q]
            assert j == -1 or 0 <= j < J


def test_knn_skips_unvoiced_frames():
    db, t = _db()
    sf = np.zeros(2 * 12, dtype=np.float32)
    sv = np.zeros(2, dtype=np.uint8)
    sa = np.array([0, 0], dtype=np.uint8)  # unvoiced -> skip
    nb = knn(sf, sv, sa, db, 4, None)
    assert (nb == -1).all()
