"""Grain retrieval — rebuild speech from the target's own waveform.

Faithful line-for-line port of the original ``grainSynth``.  The greedy
grain selection, the composite score penalty terms, the phase alignment and
the temporal gain EMA are unchanged; the per-grain inner loops use NumPy
vectorisation for the same arithmetic (results identical to the original
within float64 rounding).
"""

from __future__ import annotations

import math

import numpy as np

from hmvc.conversion.constants import D, H, HOP, SR
from hmvc.conversion.jsmath import js_exp, js_log
from hmvc.conversion.pitch import js_round

UNVOICED_GRAIN = 110


def grain_synth(len_: int, f0, vo, nb, K, db, ratio, rs, nfr, sf):
    """Rebuild the signal with target grains.  Matches the original exactly."""
    o = np.zeros(len_ + 4096, dtype=np.float32)
    ws = np.zeros(len_ + 4096, dtype=np.float32)
    t = db.t
    M = db.marks
    ts = 0.0
    prev = -9
    g0 = 1.0

    def fat(u):
        i = min(len(f0) - 1, int(math.floor(u)))
        j = min(len(f0) - 1, i + 1)
        a = u - i
        if vo[i] and vo[j]:
            return js_exp(js_log(float(f0[i])) * (1 - a) + js_log(float(f0[j])) * a)
        return float(f0[i]) if vo[i] else float(f0[j])

    def dd(j, f):
        d = 0.0
        for q in range(D):
            x = float(sf[f * D + q]) - float(db.ft[j * D + q])
            d += x * x
        return d

    while ts < len_:
        u = ts / HOP
        f = min(nfr - 1, js_round(ts / H))
        v = vo[min(len(vo) - 1, js_round(u))]
        Tn = SR / min(800.0, max(50.0, fat(u) * ratio)) if v else UNVOICED_GRAIN
        def pen(T, v=v, Tn=Tn):
            return (1.0 + 2.0 * abs(js_log(T / Tn))) if v else 1.0

        best = -1
        bs = 1e30
        for q in range(K):
            j = nb[f * K + q]
            if j < 0:
                continue
            m = db.mk[j]
            k = M[m]
            sc = (dd(j, f) + 0.05) * pen(k["T"]) * (2 if k["v"] != v else 1)
            if sc < bs:
                bs = sc
                best = m

        # Keep playing the target's own audio when the next grain still fits.
        if best >= 0 and prev >= 0 and prev + 1 < len(M) and M[prev + 1]["v"] == v:
            sc = (dd(db.fm[prev + 1], f) + 0.05) * pen(M[prev + 1]["T"]) * 0.7
            if sc < bs:
                bs = sc
                best = prev + 1

        if best >= 0:
            k = M[best]
            L = js_round(k["T"])
            g = js_round(ts)
            idx = np.arange(-L, L + 1)
            a_abs = k["p"] + idx
            e = 0.0
            c = 0
            for a in a_abs:
                if 0 <= a < len(t):
                    v_ = float(t[a])
                    e += v_ * v_
                    c += 1
            gn = max(0.25, min(4.0, float(rs[f]) / (math.sqrt(e / max(c, 1)) + 1e-6)))
            gn = g0 = 0.7 * g0 + 0.3 * gn

            d = 0
            if v:
                bc = -1e30
                dm = js_round(k["T"] / 3.0)
                idx_neg = np.arange(-L, 0)
                b_neg = g + idx_neg
                valid_b = b_neg >= 0
                for sh in range(-dm, dm + 1):
                    a_sh = k["p"] + sh + idx_neg
                    valid = (a_sh >= 0) & (a_sh < len(t)) & valid_b
                    if not valid.any():
                        continue
                    cc = float(np.dot(o[b_neg[valid]].astype(np.float64), t[a_sh[valid]].astype(np.float64)))
                    ee = 1e-9 + float(np.dot(t[a_sh[valid]].astype(np.float64), t[a_sh[valid]].astype(np.float64)))
                    cc = cc / math.sqrt(ee)
                    if cc > bc:
                        bc = cc
                        d = sh

            # additive overlap-add
            a_abs = k["p"] + d + idx
            b_abs = g + idx
            valid = (a_abs >= 0) & (a_abs < len(t)) & (b_abs >= 0) & (b_abs < len(o))
            w = 0.5 + 0.5 * np.cos(np.pi * idx / L)
            o[b_abs[valid]] = np.float32(o[b_abs[valid]].astype(np.float64) + (t[a_abs[valid]].astype(np.float64) * w[valid]) * gn)
            ws[b_abs[valid]] = np.float32(ws[b_abs[valid]].astype(np.float64) + w[valid])
            prev = best

        ts += Tn

    for i in range(len_):
        o[i] = np.float32(float(o[i]) / max(float(ws[i]), 0.35))
    return o[:len_].copy()
