"""Pitch tracking — normalized autocorrelation, median smoothing, gap filling.

Line-for-line port of the original ``trackPitch`` (window W=800, silence gate
``e0/W < 2.5e-5``, relative autocorrelation gating at 0.45, first local peak at
≥ 0.85·best with ``r[l] >= r[l-1]`` and ``r[l] >= r[l+1]``, parabolic
refinement clamped to ±1) plus the ``median`` and ``fillGaps`` utilities.
Float32 storage, float64 scalar math.
"""

from __future__ import annotations

import math

import numpy as np

from hmvc.conversion.constants import HOP, SR
from hmvc.conversion.jsmath import js_exp, js_log

WINDOW = 800          # W, kept exactly as the original
GATE_RMS = 2.5e-5     # e0/W below this -> frame skipped (silence gate)
THRESHOLD = 0.45      # best relative autocorrelation to be considered voiced
PEAK_FACTOR = 0.85    # local peak must reach >= 0.85 * best
MIN_VOICED_VOTES = 3  # a voiced frame survives only if >= 3 voiced frames in ±2
NEIGHBORHOOD = 2
FILL_GAP_MAX = 3      # unvoiced runs of <= 3 frames are bridged


def js_round(x: float) -> int:
    """JavaScript ``Math.round`` (half toward +inf)."""
    return math.floor(x + 0.5)


def track_pitch(x: np.ndarray, fmin: float, fmax: float, progress=None):
    """Return ``(f0, vo)`` — float32/uint8 arrays of length ``floor(len/HOP)+1``.

    ``progress`` is an optional callable ``progress(fraction)`` — the
    fraction is the stage-relative value in ``[0, 1]`` (the original pipeline's
    inner ``pg`` callback).
    """
    lmin = math.floor(SR / fmax)
    lmax = math.ceil(SR / fmin)
    nf = math.floor(len(x) / HOP) + 1
    f0 = np.zeros(nf, dtype=np.float32)
    vo = np.zeros(nf, dtype=np.uint8)
    r = np.zeros(lmax + 2, dtype=np.float32)

    for f in range(nf):
        if f % 60 == 0 and progress is not None:
            progress(f / nf)
        s = f * HOP - (WINDOW // 2)
        if s < 0 or s + WINDOW + lmax + 2 >= len(x):
            continue

        win_f64 = x[s:s + WINDOW].astype(np.float64)
        el_f64 = x[s + lmin:s + lmin + WINDOW].astype(np.float64)
        e0 = float(np.dot(win_f64, win_f64))
        el = float(np.dot(el_f64, el_f64))
        if e0 / WINDOW < GATE_RMS:
            continue

        best = 0.0
        win = win_f64
        for l in range(lmin, lmax + 1):
            shifted = x[s + l:s + l + WINDOW]
            c = float(np.dot(win, shifted.astype(np.float64)))
            r[l] = np.float32(c / math.sqrt(e0 * el + 1e-9))
            if float(r[l]) > best:
                best = float(r[l])
            el += float(x[s + l + WINDOW]) * float(x[s + l + WINDOW]) - float(x[s + l]) * float(x[s + l])

        if best < THRESHOLD:
            continue
        bl = 0
        for l in range(lmin + 1, lmax):
            if float(r[l]) >= PEAK_FACTOR * best and float(r[l]) >= float(r[l - 1]) and float(r[l]) >= float(r[l + 1]):
                bl = l
                break
        if not bl:
            continue
        a = float(r[bl - 1])
        b = float(r[bl])
        c = float(r[bl + 1])
        dn = a - 2.0 * b + c
        d = (0.5 * (a - c) / dn) if dn else 0.0
        f0[f] = np.float32(SR / (bl + max(-1.0, min(1.0, d))))
        vo[f] = 1

    # Median smoothing; drops isolated voiced blips (>= 3 votes in +/- 2).
    o = np.zeros(nf, dtype=np.float32)
    ov = np.zeros(nf, dtype=np.uint8)
    for i in range(nf):
        if not vo[i]:
            continue
        votes = [float(f0[j]) for j in range(i - NEIGHBORHOOD, i + NEIGHBORHOOD + 1)
                 if 0 <= j < nf and vo[j]]
        if len(votes) < MIN_VOICED_VOTES:
            continue
        votes.sort()
        o[i] = np.float32(votes[len(votes) >> 1])
        ov[i] = 1
    return o, ov


def median(f0, vo) -> float:
    """Median of the voiced ``f0`` values (0 when none)."""
    votes = sorted(float(f0[i]) for i in range(len(vo)) if vo[i])
    return float(votes[len(votes) >> 1]) if votes else 0.0


def fill_gaps(f0, vo):
    """Bridge short (<= 3-frame) unvoiced gaps and smooth the contour.

    Returns ``(f0, vo)`` — new arrays (inputs are not modified).
    """
    n = len(f0)
    o = f0.copy()
    v = vo.copy()
    i = 0
    while i < n:
        if v[i]:
            i += 1
            continue
        j = i
        while j < n and not v[j]:
            j += 1
        if i > 0 and j < n and j - i <= FILL_GAP_MAX:
            for k in range(i, j):
                a = (k - i + 1) / (j - i + 1)
                o[k] = np.float32(js_exp(js_log(float(f0[i - 1])) * (1.0 - a) + js_log(float(f0[j])) * a))
                v[k] = 1
        i = j
    s = o.copy()
    for i in range(1, n - 1):
        if v[i] and v[i - 1] and v[i + 1]:
            o[i] = np.float32(js_exp(0.25 * js_log(float(s[i - 1])) + 0.5 * js_log(float(s[i])) + 0.25 * js_log(float(s[i + 1]))))
    return o, v
