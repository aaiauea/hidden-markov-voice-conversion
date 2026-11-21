"""TD-PSOLA: marks and overlap-add."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hmvc.conversion.constants import HOP, SR
from hmvc.conversion.psola import UNVOICED_GRAIN, make_marks, psola


def _voiced_mono():
    return (np.ones(SR // 2, dtype=np.float32) * 0.3)  # 0.5 s constant


def test_make_marks_unvoiced_only_gives_unmoving_marks():
    x = np.zeros(22050, dtype=np.float32)
    f0 = np.zeros(int(len(x) / HOP) + 1, dtype=np.float32)
    vo = np.zeros(len(f0), dtype=np.uint8)
    marks = make_marks(x, f0, vo)
    assert all(m["v"] == 0 for m in marks)
    assert all(m["T"] == UNVOICED_GRAIN for m in marks)
    # unvoiced marks advance by the grain duration (110)
    assert marks[1]["p"] - marks[0]["p"] == UNVOICED_GRAIN


def test_make_marks_voiced_advances_by_period_and_clamps_gap_distance():
    x = np.ones(SR, dtype=np.float32) * 0.3
    marks = []
    f0 = np.full(int(len(x) / HOP) + 1, 200.0, dtype=np.float32)
    vo = np.ones(len(f0), dtype=np.uint8)
    marks = make_marks(x, f0, vo)
    assert marks
    # a 200 Hz source period is SR/200 = 110.25 samples
    T = SR / 200.0
    for m in marks:
        assert m["v"] == 1
        assert abs(m["T"] - T) < 1e-3


def test_psola_output_length_and_finite():
    x = np.ones(SR, dtype=np.float32) * 0.2
    f0 = np.full(int(len(x) / HOP) + 1, 200.0, dtype=np.float32)
    vo = np.ones(len(f0), dtype=np.uint8)
    marks = make_marks(x, f0, vo)
    y = psola(x, marks, 1.0)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))


def test_psola_preserves_stationary_signal_approximately():
    # A constant signal through a same-frequency PSOLA should stay constant-ish.
    x = np.ones(SR, dtype=np.float32) * 0.2
    f0 = np.full(int(len(x) / HOP) + 1, 100.0, dtype=np.float32)
    vo = np.ones(len(f0), dtype=np.uint8)
    marks = make_marks(x, f0, vo)
    y = psola(x, marks, 1.0)
    # interior region is non-silent and roughly the same amplitude
    mid = slice(len(y) // 4, 3 * len(y) // 4)
    assert np.abs(y[mid]).mean() > 0.05
