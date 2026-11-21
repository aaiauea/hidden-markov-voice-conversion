"""Target-voice retrieval database + nearest-neighbour search.

Faithful line-for-line port of ``buildDB`` and ``knn``.  The database is a
list of per-frame envelopes/CMN'd features/voicing/Cepstrum alongside the
TD-PSOLA analysis marks of the target voice (``mk``) and the nearest-mark map
(``fm``), exactly like the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hmvc.conversion.constants import D, LIF, N, HOP, SIL, NB
from hmvc.conversion.envelope import analyze, feat, cmn, CEP
from hmvc.conversion.psola import make_marks
from hmvc.conversion.pitch import js_round, median, track_pitch

DB_HOP = 512
FEATURE_hop_correction = 0.0  # kept for parity documentation; no correction


@dataclass
class RetrievalDB:
    le: list            # list of np.float32 (NB,)
    ft: np.ndarray      # (n*D,) CMN'd DCT features
    idx: list           # [[i...], [i...]] by voicing 0/1
    med: float          # median of the target's voiced f0
    n: int
    t: np.ndarray       # target signal
    marks: list         # TD-PSOLA analysis marks of the target
    mk: np.ndarray      # (n,) nearest-mark index per frame
    fm: np.ndarray      # (M,) nearest-frame index per mark
    cep: np.ndarray     # (n*LIF,) CEP[1:] per frame
    cvar: np.ndarray    # (LIF,) target cepstral variance (float64)


def build_db(t: np.ndarray, fmin: float, fmax: float, progress=None):
    """Index the target voice.  Mirrors ``buildDB`` (hop 512, SIL gate)."""
    f0, vo = track_pitch(t, fmin, fmax, lambda frac: _progress(progress, frac * 0.5))
    n_frames = int(np.floor((len(t) - N) / DB_HOP))
    le0 = np.zeros(NB, dtype=np.float32)
    les = []
    fts = []
    vs = []
    cs = []
    cps = []
    for f in range(n_frames):
        if f % 80 == 0 and n_frames:
            _progress(progress, 0.5 + 0.5 * f / n_frames)
        if analyze(t, f * DB_HOP, le0) < SIL:
            continue
        cs.append(f * DB_HOP + N // 2)
        row = np.zeros(D, dtype=np.float32)
        feat(le0, row, 0)
        fts.append(row)
        les.append(le0.copy())
        cps.append(CEP[1:].copy())
        vs.append(vo[min(len(vo) - 1, js_round((f * DB_HOP + N // 2) / HOP))])

    n = len(les)
    ft = np.zeros(n * D, dtype=np.float32)
    ok = np.ones(n, dtype=np.uint8)
    idx = [[], []]
    for i in range(n):
        ft[i * D:(i + 1) * D] = fts[i]
    cmn(ft, ok)
    for i in range(n):
        idx[int(vs[i])].append(i)

    marks = make_marks(t, f0, vo)
    mk = np.zeros(n, dtype=np.int32)
    m = 0
    for j in range(n):
        while m < len(marks) - 1 and abs(marks[m + 1]["p"] - cs[j]) <= abs(marks[m]["p"] - cs[j]):
            m += 1
        mk[j] = m

    cep = np.zeros(n * LIF, dtype=np.float32)
    cmu = np.zeros(LIF, dtype=np.float64)
    cvar = np.zeros(LIF, dtype=np.float64)
    for i in range(n):
        cep[i * LIF:(i + 1) * LIF] = cps[i]
        for q in range(LIF):
            cmu[q] += float(cps[i][q]) / n
    for i in range(n):
        for q in range(LIF):
            cvar[q] += (float(cps[i][q]) - cmu[q]) ** 2 / n

    fm = np.zeros(len(marks), dtype=np.int32)
    jj = 0
    for q in range(len(marks)):
        while jj < n - 1 and abs(cs[jj + 1] - marks[q]["p"]) <= abs(cs[jj] - marks[q]["p"]):
            jj += 1
        fm[q] = jj

    return RetrievalDB(
        le=les, ft=ft, idx=idx, med=median(f0, vo), n=n, t=t,
        marks=marks, mk=mk, fm=fm, cep=cep, cvar=cvar,
    )


def _progress(progress, fraction):
    if progress is not None:
        progress(fraction)


def knn(sf: np.ndarray, sv: np.ndarray, sa: np.ndarray,
        db: RetrievalDB, K: int, progress=None):
    """K-nearest target frames per source frame (None for unsegated frames)."""
    nf = len(sv)
    res = np.full(nf * K, -1, dtype=np.int32)
    bd = np.full(K, 1e30, dtype=np.float32)
    bi = np.full(K, -1, dtype=np.int32)
    for f in range(nf):
        if f % 48 == 0 and progress is not None:
            _progress(progress, f / nf)
        if not sa[f]:
            continue
        cand = db.idx[int(sv[f])] if len(db.idx[int(sv[f])]) > 20 else db.idx[1 - int(sv[f])]
        bd[:] = 1e30
        bi[:] = -1
        for c in range(len(cand)):
            j = cand[c]
            d = 0.0
            for q in range(D):
                t = float(sf[f * D + q]) - float(db.ft[j * D + q])
                d += t * t
            if d < float(bd[K - 1]):
                p = K - 1
                while p > 0 and float(bd[p - 1]) > d:
                    bd[p] = bd[p - 1]
                    bi[p] = bi[p - 1]
                    p -= 1
                bd[p] = np.float32(d)
                bi[p] = j
        res[f * K:(f + 1) * K] = bi
    return res
