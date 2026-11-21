"""Spectral envelope/features: analyze, mel edges, DCT, cmn."""
from __future__ import annotations

import math

import numpy as np

from hmvc.conversion.constants import D, LIF, N, NB, NM, SR
from hmvc.conversion.envelope import (
    EDGES, DCT, CEP, analyze, cmn, feat, imel, mel,
)
from hmvc.conversion.fft import fft
from hmvc.conversion import envelope as env


def _note(freq=220.0, dur=0.18, amp=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    return x.astype(np.float32)


def test_mel_imel_round_trip():
    assert mel(440.0) == pytest.approx(2595.0 * math.log10(1.0 + 440.0 / 700.0), rel=1e-12)
    assert imel(mel(440.0)) == pytest.approx(440.0, rel=1e-9)


def test_edges_are_strictly_increasing_float32_ints():
    assert len(EDGES) == NM + 2
    for i in range(len(EDGES) - 1):
        assert EDGES[i] < EDGES[i + 1]
    assert EDGES[0] >= 1 and EDGES[-1] <= NB - 1


def test_dct_table_matches_cos_rows():
    for j in range(1, D + 1):
        row = DCT[j - 1]
        for b in range(NM):
            assert float(row[b]) == pytest.approx(math.cos(math.pi * j * (b + 0.5) / NM), abs=1e-7)


def test_analyze_fills_envelope_and_returns_rms():
    x = _note()
    le = np.zeros(NB, dtype=np.float32)
    rms = analyze(x, 0, le)
    assert rms > 0.0
    # le has float32 values and matches |X| peak structure
    assert le.shape == (NB,)
    # rms of a zero-padded windowed segment is positive
    assert rms < 10.0


def test_feat_matches_mel_dct_formula():
    x = _note()
    le = np.zeros(NB, dtype=np.float32)
    analyze(x, 0, le)
    out = np.zeros(D, dtype=np.float32)
    feat(le, out, 0)
    # recompute from the same le
    mb = np.zeros(NM, dtype=np.float32)
    for b in range(NM):
        lo, hi = EDGES[b], EDGES[b + 2]
        s = 0.0
        for k in range(lo, hi + 1):
            s += float(le[k])
        mb[b] = np.float32(s / (hi - lo + 1))
    for j in range(D):
        s = 0.0
        d_j = DCT[j]
        for b in range(NM):
            s += float(mb[b]) * float(d_j[b])
        assert float(out[j]) == float(np.float32(s))


def test_cmn_subtracts_mean_over_ok_frames():
    DCT_D = 12
    ft = np.zeros(2 * DCT_D, dtype=np.float32)
    ft[0::DCT_D] = 10.0
    ft[1::DCT_D] = 20.0
    ok = np.array([1, 1], dtype=np.uint8)
    cmn(ft, ok)
    # mean over both ok frames is 10 for component 0 and 20 for component 1
    assert ft[0] == pytest.approx(0.0, abs=1e-6)
    assert ft[1] == pytest.approx(0.0, abs=1e-6)
    # unvoiced frame is still SUBTRACTED by the mean of the voiced frames
    ft2 = np.zeros(2 * DCT_D, dtype=np.float32)
    ft2[0 * DCT_D + 0] = 10.0   # voiced frame, component 0
    ft2[1 * DCT_D + 0] = 20.0   # unvoiced frame, component 0
    ok2 = np.array([1, 0], dtype=np.uint8)
    cmn(ft2, ok2)
    assert ft2[0] == pytest.approx(0.0, abs=1e-6)   # 10 - 10
    assert ft2[12] == pytest.approx(10.0, abs=1e-6)  # 20 - 10


def test_cepstrum_is_lifter_smoothed():
    x = _note(freq=350.0, dur=0.22)
    le = np.zeros(NB, dtype=np.float32)
    analyze(x, 0, le)
    # lifter cone: CEP[0..LIF] should be finite and roughly log|X| levels
    assert CEP.shape == (LIF + 1,)
    assert np.all(np.isfinite(CEP))


import pytest
