"""TD-PSOLA pitch shifting — ``makeMarks`` + ``psola``.

Faithful line-for-line port of the original.  ``U = 110`` unvoiced grain
(micro-mark) duration in samples, raised-cosine synthesis with the same
window ``.5 + .5 * cos(pi * i / L)`` and the same ``.35`` overlap floor.
"""

from __future__ import annotations

import math

import numpy as np

from hmvc.conversion.constants import HOP, SR

UNVOICED_GRAIN = 110          # U


def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def make_marks(x: np.ndarray, f0: np.ndarray, vo: np.ndarray):
    """Build TD-PSOLA analysis marks (list of dicts, like the original)."""
    marks = []
    fr = lambda t: min(len(f0) - 1, js_round(t / HOP))

    pos = 0.0
    prev = False

    while pos < len(x):
        fi = fr(pos)
        if vo[fi]:
            T = SR / float(f0[fi])
            c = _peak(x, pos, T * 0.3 if prev else T * 0.5)
            l = marks[-1]["p"] if marks else -1e9
            if c > l + T * 0.5:
                marks.append({"p": c, "T": T, "v": 1, "f": float(f0[fi])})
                pos = c + T
            else:
                pos += T * 0.5
            prev = True
        else:
            marks.append({"p": js_round(pos), "T": UNVOICED_GRAIN, "v": 0, "f": 0.0})
            pos += UNVOICED_GRAIN
            prev = False
    return marks


def _peak(x: np.ndarray, c: float, r: float) -> int:
    start = max(0, js_round(c - r))
    bv = -9.0
    b = start
    for i in range(start, min(len(x) - 1, js_round(c + r)) + 1):
        if float(x[i]) > bv:
            bv = float(x[i])
            b = i
    return b


def psola(x: np.ndarray, marks, ratio: float):
    """TD-PSOLA — overlap-add with the same grain window and normalizer."""
    o = np.zeros(len(x) + 4096, dtype=np.float32)
    ws = np.zeros(len(x) + 4096, dtype=np.float32)
    ts = 0.0
    mi = 0
    while ts < len(x) and marks:
        while mi < len(marks) - 1 and abs(marks[mi + 1]["p"] - ts) <= abs(marks[mi]["p"] - ts):
            mi += 1
        k = marks[mi]
        L = js_round(k["T"])
        g = js_round(ts)
        for i in range(-L, L + 1):
            a = k["p"] + i
            b = g + i
            if a < 0 or a >= len(x) or b < 0 or b >= len(o):
                continue
            w = 0.5 + 0.5 * math.cos(math.pi * i / L)
            o[b] = np.float32(float(o[b]) + float(x[a]) * w)
            ws[b] = np.float32(float(ws[b]) + w)
        ts += SR / min(800.0, max(50.0, k["f"] * ratio)) if k["v"] else k["T"]
    for i in range(len(x)):
        o[i] = np.float32(float(o[i]) / max(float(ws[i]), 0.35))
    return o[:len(x)].copy()
