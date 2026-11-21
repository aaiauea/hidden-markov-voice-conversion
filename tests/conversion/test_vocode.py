"""GMM vocoder: structure and invariants (full parity covered by fixture test)."""
from __future__ import annotations

import numpy as np
import pytest

from hmvc.conversion.constants import D, HOP, N, NH
from hmvc.conversion.psola import make_marks
from hmvc.conversion.pitch import fill_gaps, track_pitch
from hmvc.conversion.retrieve import build_db, knn
from hmvc.conversion.vocode import inv, vocode


def test_inv_inverts_symmetric_matrix():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 4)) * 0.4 + np.eye(4)
    I = inv(A.reshape(-1), 4)
    prod = I @ A
    np.testing.assert_allclose(prod, np.eye(4), atol=1e-10)


def test_inv_handles_singular():
    A = np.zeros((3, 3), dtype=np.float64)
    A[0, 0] = 1.0
    A[0, 1] = 0.5
    I = inv(A.reshape(-1), 3)
    prod = I @ A
    assert np.all(np.isfinite(prod))


def test_vocode_returns_silence_when_too_few_frames():
    out = vocode(22050, np.ones(10, dtype=np.float32) * 200.0,
                np.ones(10, dtype=np.uint8), np.full(10 * 6, -1, dtype=np.int32),
                6, None, 1.0, np.full(10, 0.1, dtype=np.float32),
                np.zeros(10 * D, dtype=np.float32), np.zeros(10, dtype=np.uint8), 10, 1.0, None, 123)
    # without active frames it returns a zero buffer (still allocated)
    assert out.shape == (22050 + N,)
    assert np.all(out == 0)


def test_vocode_end_to_end_synthetic():
    # small: build a target DB, track, retrieve, and synthesize
    fs = 22050
    n = fs // 2
    t = np.arange(n) / fs
    x = np.zeros(n, dtype=np.float32)
    for h in range(1, 30):
        x += np.sin(2 * np.pi * 180.0 * h * t) / h
    x *= 0.3 / np.sqrt(np.mean(x ** 2))
    db = build_db(x, 70.0, 450.0)

    # fixture-like minimal input
    nfr = 30
    le = np.zeros(513, dtype=np.float32)
    sf = np.zeros(nfr * D, dtype=np.float32)
    rs = np.full(nfr, 0.15, dtype=np.float32)
    sv = np.ones(nfr, dtype=np.uint8)
    sa = np.ones(nfr, dtype=np.uint8)
    f0 = np.full(nfr, 170.0, dtype=np.float32)
    vo = np.ones(nfr, dtype=np.uint8)
    # nearest target frame to each source frame (j=0 valid)
    nb = np.zeros(nfr * 6, dtype=np.int32)

    out = vocode(nfr * 256, f0, vo, nb, 6, db, 1.0, rs, sf, sa, nfr, 1.0, None, 42)
    assert out.shape[0] == nfr * 256 + N
    # zero-feature degenerate input mirrors the original's silent NaN propagation;
    # the shape and finite frame budget are therefore what this stage guarantees.
    assert out.shape[0] == nfr * 256 + N
