"""GMM voice model + harmonic/noise vocoder — strict port of ``inv``/``vocode``.

This is the most elaborate stage of the original:

1. training pairs (source frame -> mean cepstrum of its K closest target frames);
2. Gaussian-mixture fit (EM, deterministic quantile-index init);
3. per-component ridge regression source features -> target cepstrum;
4. per-frame conversion, temporal smoothing, global-variance restore;
5. harmonic + envelope-shaped noise synthesis.

The numerics — EM iteration count (15), component count
``max(2, min(8, floor(n/60)))``, ridge constant ``0.15``, time smoothing
weights ``[.1, .2, .4, .2, .1]``, variance restore ``1 + 0.7 * (clamp(sqrt(cvar/vp), 1, 2.2) - 1)``,
harmonic/noise split, ``cst = 2/(N*N*0.375)``, harmonic phase fill
``2*pi*0.6*(((0.6180339*h*h)%1) - 0.5)`` — are unchanged.

Noise uses the same ``Math.random()`` call pattern (two calls per spectral bin
per frame).  With ``rng_seed`` set, a deterministic 64-bit LCG is used
(see :mod:`hmvc.conversion.rng`).
"""

from __future__ import annotations

import math

import numpy as np

from hmvc.conversion.constants import (
    D, H, HOP, LIF, N, NB, NH, SR, STFT_SCALE, WINDOW_NORM,
)
from hmvc.conversion.fft import WIN, fft
from hmvc.conversion.jsmath import js_exp, js_log
from hmvc.conversion.rng import make_rng

#: module-level scratch buffers, mirroring the original shared buffers
_Re = np.zeros(N, dtype=np.float32)
_Im = np.zeros(N, dtype=np.float32)

PHASE_RETURN_COEFF = 0.6180339


def _js_round(x: float) -> int:
    return math.floor(x + 0.5)


_js_log = js_log
_js_exp = js_exp


def inv(A: np.ndarray, n: int) -> np.ndarray:
    """Gauss-Jordan inverse (partial pivoting) — matches the original."""
    I = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        I[i, i] = 1.0
    A = A.reshape(n, n).copy()
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(A[r, c]) > abs(A[p, c]):
                p = r
        if p != c:
            tmp = A[c].copy(); A[c] = A[p].copy(); A[p] = tmp
            tmp = I[c].copy(); I[c] = I[p].copy(); I[p] = tmp
        d = A[c, c]
        if not d or d != d:
            d = 1e-9
        for k in range(n):
            A[c, k] /= d
            I[c, k] /= d
        for r in range(n):
            if r == c:
                continue
            f = A[r, c]
            if not f:
                continue
            for k in range(n):
                A[r, k] -= f * A[c, k]
                I[r, k] -= f * I[c, k]
    return I


def vocode(len_: int, f0, vo, nb, K, db, ratio, rs, sf, sa, nfr, form,
           progress=None, rng_seed=None):
    """Full GMM-vocoder voicing path.  Matches the original exactly."""
    C = LIF
    out = np.zeros(len_ + N, dtype=np.float32)
    act = []
    for f in range(nfr):
        if sa[f] and nb[f * K] >= 0:
            act.append(f)
    n = len(act)
    if n < 30:
        return out
    rng = make_rng(rng_seed)

    # 1) training pairs: source frame -> mean cepstrum of K closest target frames
    X = np.zeros((n, D), dtype=np.float32)
    Y = np.zeros((n, C), dtype=np.float32)
    for i in range(n):
        f = act[i]
        for q in range(D):
            X[i, q] = sf[f * D + q]
        c = 0
        for k in range(K):
            j = nb[f * K + k]
            if j < 0:
                continue
            c += 1
            for r in range(C):
                Y[i, r] += db.cep[j * C + r]
        for r in range(C):
            Y[i, r] = np.float32(float(Y[i, r]) / max(c, 1))

    # 2) fit a Gaussian mixture on the source features with EM
    M = max(2, min(8, n // 60))
    mu = np.zeros((M, D), dtype=np.float64)
    vr = np.zeros((M, D), dtype=np.float64)
    pi = np.full(M, 1.0 / M, dtype=np.float64)
    G = np.zeros((n, M), dtype=np.float64)
    gm = np.zeros(D, dtype=np.float64)
    gv = np.zeros(D, dtype=np.float64)
    g = np.zeros(M, dtype=np.float64)
    for i in range(n):
        for q in range(D):
            gm[q] += float(X[i, q]) / n
    for i in range(n):
        for q in range(D):
            gv[q] += (float(X[i, q]) - gm[q]) ** 2 / n
    for m in range(M):
        i = math.floor((m + 0.5) * n / M)
        for q in range(D):
            mu[m, q] = float(X[i, q])
            vr[m, q] = gv[q]

    def post(x, o, gg):
        # x: flat np.float32 array; o: start index (frame offset, like JS)
        mx = -1e300
        lp = np.zeros(M, dtype=np.float64)
        for m in range(M):
            l = _js_log(pi[m] + 1e-12)
            for q in range(D):
                t = float(x[o + q]) - mu[m, q]
                v = vr[m, q]
                l -= 0.5 * (_js_log(v) + t * t / v)
            lp[m] = l
            if l > mx:
                mx = l
        z = 0.0
        for m in range(M):
            lp[m] = _js_exp(lp[m] - mx)
            z += lp[m]
        for m in range(M):
            gg[m] = lp[m] / z

    def E():
        for i in range(n):
            post(X.ravel(), i * D, g)
            for m in range(M):
                G[i, m] = g[m]

    for it in range(15):
        E()
        for m in range(M):
            w = 1e-9
            s1 = np.zeros(D, dtype=np.float64)
            s2 = np.zeros(D, dtype=np.float64)
            for i in range(n):
                t = G[i, m]
                w += t
                for q in range(D):
                    x = float(X[i, q])
                    s1[q] += t * x
                    s2[q] += t * x * x
            for q in range(D):
                a = s1[q] / w
                mu[m, q] = a
                vr[m, q] = max(s2[q] / w - a * a, gv[q] * 0.02)
            pi[m] = w / n
        _progress(progress, it / 15 * 0.3, "Training voice model…")
    E()

    # 3) per-component linear ridge regression source -> target cepstrum
    MX = []
    MY = []
    AA = []
    for m in range(M):
        w = 1e-9
        mx = np.zeros(D, dtype=np.float64)
        my = np.zeros(C, dtype=np.float64)
        dx = np.zeros(D, dtype=np.float64)
        dy = np.zeros(C, dtype=np.float64)
        Sxx = np.zeros((D, D), dtype=np.float64)
        Syx = np.zeros((C, D), dtype=np.float64)
        for i in range(n):
            t = G[i, m]
            w += t
            for q in range(D):
                mx[q] += t * float(X[i, q])
            for r in range(C):
                my[r] += t * float(Y[i, r])
        for q in range(D):
            mx[q] /= w
        for r in range(C):
            my[r] /= w
        for i in range(n):
            t = G[i, m]
            if t < 1e-3:
                continue
            for q in range(D):
                dx[q] = float(X[i, q]) - mx[q]
            for r in range(C):
                dy[r] = float(Y[i, r]) - my[r]
            for a in range(D):
                for b in range(D):
                    Sxx[a, b] += t * dx[a] * dx[b]
            for r in range(C):
                for b in range(D):
                    Syx[r, b] += t * dy[r] * dx[b]
        tr = 0.0
        for q in range(D):
            tr += Sxx[q, q]
        for q in range(D):
            Sxx[q, q] += 0.15 * tr / D + 1e-6
        iv = inv(Sxx.reshape(D * D), D)
        A = np.zeros((C, D), dtype=np.float64)
        for r in range(C):
            for b in range(D):
                z = 0.0
                for c in range(D):
                    z += Syx[r, c] * iv[c, b]
                A[r, b] = z
        MX.append(mx)
        MY.append(my)
        AA.append(A)

    # 4) convert every frame, smooth in time, restore variance
    _progress(progress, 0.35, "Converting spectra…")
    P = np.zeros(nfr * C, dtype=np.float32)
    for f in range(nfr):
        if not sa[f]:
            continue
        post(sf, f * D, g)
        for m in range(M):
            for r in range(C):
                v = MY[m][r]
                for b in range(D):
                    v += AA[m][r, b] * (float(sf[f * D + b]) - MX[m][b])
                P[f * C + r] = np.float32(float(P[f * C + r]) + g[m] * v)

    P2 = P.copy()
    wt = [0.1, 0.2, 0.4, 0.2, 0.1]
    for f in range(2, nfr - 2):
        if not sa[f]:
            continue
        for r in range(C):
            v = 0.0
            z = 0.0
            for d in range(-2, 3):
                if sa[f + d]:
                    v += wt[d + 2] * float(P[(f + d) * C + r])
                    z += wt[d + 2]
            P2[f * C + r] = np.float32(v / z)
    P = P2

    mp = np.zeros(C, dtype=np.float64)
    vp = np.zeros(C, dtype=np.float64)
    for f in act:
        for r in range(C):
            mp[r] += float(P[f * C + r]) / n
    for f in act:
        for r in range(C):
            vp[r] += (float(P[f * C + r]) - mp[r]) ** 2 / n
    sc = np.zeros(C, dtype=np.float64)
    for r in range(C):
        sc[r] = 1.0 + 0.7 * (max(1.0, min(2.2, math.sqrt(db.cvar[r] / (vp[r] + 1e-12)))) - 1.0)
    for f in act:
        for r in range(C):
            P[f * C + r] = np.float32(mp[r] + (float(P[f * C + r]) - mp[r]) * sc[r])

    # 5) synthesis: harmonics at the (smoothed) source pitch + envelope-shaped noise
    COS = []
    for q in range(1, C + 1):
        COS.append(np.array([2.0 * math.cos(2.0 * math.pi * q * k / N) for k in range(NB)], dtype=np.float32))

    Am = np.zeros(nfr * NH, dtype=np.float32)
    F0 = np.zeros(nfr, dtype=np.float32)
    HM = np.zeros(nfr, dtype=np.int32)
    le = np.zeros(NB, dtype=np.float32)
    tmp = np.zeros(NB, dtype=np.float32)
    Pw = np.zeros(NB, dtype=np.float32)
    Nw = np.zeros(NB, dtype=np.float32)
    Ph = np.zeros(NB, dtype=np.float32)

    def ss(a, b, x):
        t = max(0.0, min(1.0, (x - a) / (b - a)))
        return t * t * (3.0 - 2.0 * t)

    def fat(u):
        i = min(len(f0) - 1, int(math.floor(u)))
        j = min(len(f0) - 1, i + 1)
        a = u - i
        if vo[i] and vo[j]:
            return _js_exp(_js_log(float(f0[i])) * (1 - a) + _js_log(float(f0[j])) * a)
        return float(f0[i]) if vo[i] else float(f0[j])

    for f in range(nfr):
        if f % 40 == 0:
            _progress(progress, 0.4 + 0.3 * f / nfr, "Building voice…")
        if not sa[f]:
            continue
        le[:] = 0.0
        for q in range(C):
            y = float(P[f * C + q])
            cs = COS[q]
            le = np.float32(le.astype(np.float64) + y * cs.astype(np.float64))
        if form != 1.0:
            tmp[:] = le
            for k in range(NB):
                u = k / form
                i = min(NB - 2, int(math.floor(u)))
                a = min(1.0, u - i)
                le[k] = np.float32(float(tmp[i]) * (1 - a) + float(tmp[i + 1]) * a)
        mxv = -1e9
        for k in range(NB):
            if le[k] > mxv:
                mxv = float(le[k])
        for k in range(NB):
            le[k] = np.float32(max(float(le[k]) - mxv, -9.0))

        u = f * H / HOP
        v = vo[min(len(vo) - 1, _js_round(u))]
        f0f = min(800.0, max(50.0, fat(u) * ratio)) if v else 0.0
        Eh = 0.0
        En = 0.0
        for k in range(NB):
            nw = (0.03 + 0.55 * ss(3500.0, 9000.0, k * SR / N)) if v else 1.0
            pw = _js_exp(2.0 * float(le[k]))
            Pw[k] = np.float32(pw)
            Nw[k] = np.float32(nw)
            # The original stores Ph[k] (float32) but reads it back when summing;
            # En is accumulated from the UNROUNDED ephemeral values.
            Ph[k] = np.float32((1.0 - nw) * pw if v else 0.0)
            Eh += float(Ph[k])
            En += nw * pw
        Eh *= STFT_SCALE
        En *= STFT_SCALE
        gn = math.sqrt(float(rs[f]) * float(rs[f]) / (Eh + En + 1e-20))

        if v:
            Hn = min(NH, int(math.floor(10500.0 / f0f)))
            sp = 0.0
            tp = np.zeros(Hn, dtype=np.float32)
            for h in range(1, Hn + 1):
                b = h * f0f * N / SR
                i = min(NB - 2, int(math.floor(b)))
                a = b - i
                tp[h - 1] = np.float32(float(Ph[i]) * (1 - a) + float(Ph[i + 1]) * a)
                sp += float(tp[h - 1])
            for h in range(1, Hn + 1):
                Am[f * NH + h - 1] = np.float32(gn * math.sqrt(2.0 * Eh * float(tp[h - 1]) / (sp + 1e-20)))
            HM[f] = Hn
            F0[f] = np.float32(f0f)

        for k in range(1, N // 2):
            m = gn * math.sqrt(float(Nw[k]) * float(Pw[k]) / WINDOW_NORM) * math.sqrt(-math.log(rng.random() + 1e-9))
            a = rng.random() * 6.2831853
            _Re[k] = np.float32(m * math.cos(a))
            _Im[k] = np.float32(m * math.sin(a))
            _Re[N - k] = _Re[k]
            _Im[N - k] = np.float32(-float(_Im[k]))
        _Re[0] = 0.0
        _Im[0] = 0.0
        _Re[N // 2] = 0.0
        _Im[N // 2] = 0.0
        fft(_Re, _Im, 1)
        p = f * H - N // 2
        for i in range(N):
            q = p + i
            if 0 <= q < len(out):
                out[q] = np.float32(float(out[q]) + float(_Re[i]) * float(WIN[i]) / 1.2247)

    PH = np.zeros(NH + 1, dtype=np.float32)
    for h in range(1, NH + 1):
        PH[h] = np.float32(2.0 * math.pi * 0.6 * (((PHASE_RETURN_COEFF * h * h) % 1.0) - 0.5))
    ph = 0.0
    for n2 in range(len_):
        if (n2 & 16383) == 0:
            _progress(progress, 0.7 + 0.3 * n2 / len_, "Synthesizing voice…")
        u = n2 / H
        f = min(nfr - 1, int(math.floor(u)))
        f1 = min(nfr - 1, f + 1)
        a = u - f
        a0 = float(F0[f])
        a1 = float(F0[f1])
        fi = (a0 * (1 - a) + a1 * a) if (a0 > 0 and a1 > 0) else (a0 if a0 > 0 else a1)
        if fi <= 0.0:
            continue
        ph += 2.0 * math.pi * fi / SR
        if ph > 6.2831853:
            ph -= 6.2831853
        Hn = max(HM[f], HM[f1])
        o0 = f * NH
        o1 = f1 * NH
        z = 0.0
        for h in range(1, Hn + 1):
            z += (float(Am[o0 + h - 1]) * (1 - a) + float(Am[o1 + h - 1]) * a) * math.sin(h * ph + float(PH[h]))
        out[n2] = np.float32(float(out[n2]) + z)

    return out


def _progress(progress, fraction, message):
    if progress is not None:
        progress(fraction, message)
