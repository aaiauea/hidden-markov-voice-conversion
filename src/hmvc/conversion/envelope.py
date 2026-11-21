"""Spectral envelope and DCT features — a line-for-line port of `analyze`/`feat`/`cmn`.

The original code (``voice-converter (4).html``) keeps shared module-level
buffers (``Re``, ``Im``, ``Cr``, ``Ci``, ``CEP``).  This module does the same
(nothing is returned that the caller could keep alive), but with explicit
scratch buffers per call for the windowed analysis frame.  The numerics —
float32 storage, float64 scalar math, ``Math.round`` semantics (half away from
zero), the ``1e-5`` cepstral guard, the lifter window and the SIL gate — are
identical.
"""

from __future__ import annotations

import math

import numpy as np

from hmvc.conversion.constants import D, LIF, N, NB, NM, SIL, SR
from hmvc.conversion.fft import WIN, fft

#: Shared analysis buffers (same convention as the original's module-level
#: ``Re``/``Im``/``Cr``/``Ci``).  Exposed so the pipeline can apply the
#: envelope transfer over the same buffers.
Re = np.zeros(N, dtype=np.float32)
Im = np.zeros(N, dtype=np.float32)
_Cr = np.zeros(N, dtype=np.float32)
_Ci = np.zeros(N, dtype=np.float32)
#: Module-level cepstral accumulator (float32), shared like the original.
CEP = np.zeros(LIF + 1, dtype=np.float32)

#: float64 view of the window; the original reads win[i] as float64.
_WIN64 = WIN.astype(np.float64)


def mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def imel(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _js_round(x: float) -> int:
    """JavaScript ``Math.round`` (half toward +inf)."""
    return math.floor(x + 0.5)


def build_edges():
    """Bin-edge list (NM+2 entries); the original builds this once at load."""
    edges = []
    for i in range(NM + 2):
        b = _js_round(imel(mel(100.0) + (mel(8000.0) - mel(100.0)) * i / (NM + 1)) / (SR / N))
        if i and b <= edges[i - 1]:
            b = edges[i - 1] + 1
        edges.append(b)
    return edges


def build_dct():
    """``D`` float32 cos tables; ``build_dct()[j][b]`` is ``cos(pi*(j+1)*(b+.5)/NM)``."""
    dct = np.zeros((D, NM), dtype=np.float32)
    for j in range(1, D + 1):
        for b in range(NM):
            dct[j - 1, b] = np.float32(math.cos(math.pi * j * (b + 0.5) / NM))
    return dct


EDGES = build_edges()
DCT = build_dct()


def analyze(x, pos: int, le: np.ndarray) -> float:
    """Fill ``le`` with the lifter-smoothed log envelope; return frame RMS.

    Mirrors the original ``analyze`` exactly (shared ``Re``/``Im``/``Cr``/
    ``Ci`` buffers, float32 storage, float64 scalar math).
    """
    win = _WIN64
    lo = max(0, pos)
    hi = min(len(x), pos + N)
    if hi > lo:
        src = x[lo:hi]
        pad = max(0, -pos)
        v = np.zeros(N, dtype=np.float64)
        v[pad:pad + len(src)] = src * win[pad:pad + len(src)]
    else:
        v = np.zeros(N, dtype=np.float64)
    Re[:] = v
    Im[:] = 0.0
    e = 0.0
    for i in range(N):
        e += float(v[i]) * float(v[i])

    fft(Re, Im, 0)

    for i in range(N):
        _Cr[i] = math.log(math.hypot(float(Re[i]), float(Im[i])) + 1e-5)
        _Ci[i] = 0.0
    fft(_Cr, _Ci, 1)

    for i in range(LIF + 1, N - LIF):
        _Cr[i] = 0.0
    _Ci[:] = 0.0
    for q in range(0, LIF + 1):
        CEP[q] = _Cr[q]

    fft(_Cr, _Ci, 0)
    le[:] = _Cr[:NB]
    return math.sqrt(e / (N * 0.375))


def feat(le: np.ndarray, o: np.ndarray, off: int) -> None:
    """Write the 12-D Mel-DCT of ``le`` into ``o[off:off+D]``."""
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
        o[off + j] = np.float32(s)


def cmn(ft: np.ndarray, ok: np.ndarray) -> None:
    """Subtract the mean of ``ft`` (over ``ok`` frames), in place.

    Both voicing gates are applied in the mean but the *subtraction* uses
    every frame — quirk preserved from the original.
    """
    m = np.zeros(D, dtype=np.float32)
    n = 0
    for f in range(ok.shape[0]):
        if not ok[f]:
            continue
        n += 1
        for j in range(D):
            m[j] = np.float32(float(m[j]) + float(ft[f * D + j]))
    for j in range(D):
        m[j] = np.float32(float(m[j]) / max(n, 1))
    for f in range(ok.shape[0]):
        for j in range(D):
            ft[f * D + j] = np.float32(float(ft[f * D + j]) - float(m[j]))
