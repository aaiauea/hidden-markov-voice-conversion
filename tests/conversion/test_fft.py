"""FFT port checks (butterfly, bit-reversal, inverse round-trip)."""
from __future__ import annotations

import numpy as np

from hmvc.conversion.constants import N
from hmvc.conversion.fft import _REV, cosT, sinT, WIN, fft


def test_tables_match_original_definitions():
    for i in range(N // 2):
        assert isinstance(cosT[i], np.float32)
        assert abs(float(cosT[i]) - np.cos(2 * np.pi * i / N)) < 1e-6
        assert abs(float(sinT[i]) - -np.sin(2 * np.pi * i / N)) < 1e-6
    for i in range(N):
        assert abs(float(WIN[i]) - (0.5 - 0.5 * np.cos(2 * np.pi * i / N))) < 1e-6


def test_bit_reversal_is_a_permutation_of_range():
    rev = np.array(_REV, dtype=np.int64)
    assert sorted(rev) == list(range(N))


def test_forward_then_inverse_recovers_signal():
    rng = np.random.default_rng(7)
    re = rng.standard_normal(N).astype(np.float32)
    im = np.zeros(N, dtype=np.float32)
    re0 = re.copy()
    fft(re, im, 0)
    fft(re, im, 1)
    assert np.max(np.abs(re - re0)) < 1e-3


def test_inverse_recovers_constant_input():
    re = np.full(N, 1.0, dtype=np.float32)
    im = np.zeros(N, dtype=np.float32)
    fft(re, im, 0)
    fft(re, im, 1)
    assert abs(float(re[0]) - 1.0) < 1e-3
    assert abs(float(re[123]) - 1.0) < 1e-3
