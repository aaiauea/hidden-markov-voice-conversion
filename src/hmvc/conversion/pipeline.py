"""Main conversion pipeline — a strict 1:1 port of the original ``convert()``.

The control flow, the analysis order and the thresholds are unchanged:
tracking -> contour fill -> retrieval database -> mode resolution -> pitch
adjustment -> (TD-PSOLA + analysis) -> synthesis (grain/vocoder/envelope) ->
final normalization.  ``progress`` is an optional callable
``(absolute_fraction, stage_message)`` — the original browser put it into the
"bar"; the port reports the identical absolute fraction and the identical
stage messages.
"""

from __future__ import annotations

import math

import numpy as np

from hmvc.conversion.config import ConversionConfig
from hmvc.conversion.constants import D, DB_MIN_FRAMES, H, HOP, N, NB, SIL, SR
from hmvc.conversion.envelope import Im, Re, analyze, cmn, feat
from hmvc.conversion.fft import WIN, fft
from hmvc.conversion.jsmath import js_exp
from hmvc.conversion.grain import grain_synth
from hmvc.conversion.psola import make_marks, psola
from hmvc.conversion.pitch import fill_gaps, js_round, median, track_pitch
from hmvc.conversion.retrieve import build_db, knn
from hmvc.conversion.types import ConversionResult
from hmvc.conversion.vocode import vocode

TARGET_MAX_SECONDS = 150   # the UI truncates the target to 150 s on load

#: Stage messages (identical to the original UI)
MSG_TRACKING = "Tracking pitch…"
MSG_INDEX = "Indexing target voice…"
MSG_SHIFT = "Shifting pitch…"
MSG_ANALYSE = "Analysing source…"
MSG_RETRIEVE = "Retrieving closest frames…"
MSG_TRANSFORM = "Transforming timbre…"
MSG_TOO_SHORT = "Target too short or quiet: ignoring retrieval."

#: UI-session memoization of the retrieval database — the original key is
#: ``tgt.length:fmin:fmax:tgt[tgt.length>>1]`` (same value, same meaning).
#: ``clear_cache()`` resets it, e.g. between sessions.
_CACHE = {"key": None, "db": None}


def clear_cache() -> None:
    """Drop the memoized retrieval database (the original memoizes per session)."""
    _CACHE["key"] = None
    _CACHE["db"] = None


def _progress(progress, absolute_fraction, message):
    if progress is not None:
        progress(absolute_fraction, message)


def convert(x, target=None, config=None, progress=None) -> ConversionResult:
    """Convert a source voice into the target voice (or pure pitch/formants).

    Parameters
    ----------
    x : array-like
        Source audio at 22050 Hz (mono).  The caller is responsible for
        loading and resampling (see the CLI).
    target : array-like | None
        Target voice audio (mono).  Truncated to ``target_max_seconds``
        (150 s in the original UI) before the retrieval database is built.
    config : ConversionConfig | None
        Defaults to the original UI defaults (mode="voc", blend 0.8,
        neighbours 6, auto-pitch on, standard voice range 70-450 Hz).
    progress : callable | None
        ``progress(absolute_fraction, stage_message)`` — optional.
    """
    config = config or ConversionConfig()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    t = None
    if target is not None:
        t = np.asarray(target, dtype=np.float32).reshape(-1)
        max_len = int(config.target_max_seconds * SR)
        if len(t) > max_len:
            t = t[:max_len]
        if len(t) == 0:
            t = None

    # -- 1) pitch tracking ---------------------------------------------------

    def track_cb(frac):
        _progress(progress, 0.15 * frac, MSG_TRACKING)

    f0r, vor = track_pitch(x, config.fmin, config.fmax, track_cb)
    f0, vo = fill_gaps(f0r, vor)

    # -- 2) target retrieval database --------------------------------------

    db = None
    if t is not None and (config.envelope_blend > 0.0 or config.mode != "env"):
        key = f"{len(t)}:{config.fmin}:{config.fmax}:{t[len(t) >> 1]}"
        if key == _CACHE["key"]:
            db = _CACHE["db"]
        else:
            _progress(progress, 0.15, MSG_INDEX)

            def index_cb(frac):
                _progress(progress, 0.15 + 0.25 * frac, MSG_INDEX)

            db = build_db(t, config.fmin, config.fmax, index_cb)
            _CACHE["key"] = key
            _CACHE["db"] = db
        if db is not None and db.n < DB_MIN_FRAMES:
            db = None
            _progress(progress, 0.40, MSG_TOO_SHORT)

    grain = db is not None and config.mode == "grain"
    voc = db is not None and config.mode == "voc"
    direct = grain or voc

    # -- 3) pitch adjustment ------------------------------------------------
    _progress(progress, 0.4, MSG_SHIFT)

    ratio = 2.0 ** (config.pitch_semitones / 12.0)
    if db is not None and config.auto_pitch and db.med:
        sm = median(f0, vo)
        if sm:
            ratio *= db.med / sm
    ratio = max(0.4, min(3.0, ratio))

    if abs(ratio - 1.0) > 0.005 and not direct:
        y = psola(x, make_marks(x, f0, vo), ratio)
        pitch_shifted = True
    else:
        y = x.copy()
        pitch_shifted = False

    # -- 4) source analysis + retrieval -------------------------------------
    nfr = int(math.floor((len(y) + N / 2) / H)) + 1
    le = np.zeros(NB, dtype=np.float32)
    sv = np.zeros(nfr, dtype=np.uint8)
    sa = np.zeros(nfr, dtype=np.uint8)
    rs = np.zeros(nfr, dtype=np.float32)
    nb = None
    sf = None

    if db is not None:
        sf = np.zeros(nfr * D, dtype=np.float32)

        def analyse_cb(frac):
            _progress(progress, 0.4 + 0.1 * frac, MSG_ANALYSE)

        for f in range(nfr):
            if f % 80 == 0:
                analyse_cb(f / nfr)
            p = f * H - N // 2
            rs[f] = analyze(y, p, le)
            sa[f] = 1 if float(rs[f]) >= SIL else 0
            feat(le, sf, f * D)
            sv[f] = vo[min(len(vo) - 1, max(0, js_round((p + N // 2) / HOP)))]

        cmn(sf, sa)
        sm_prev = sf.copy()
        for f in range(1, nfr - 1):
            for q in range(D):
                sf[f * D + q] = np.float32(
                    0.25 * float(sm_prev[(f - 1) * D + q])
                    + 0.5 * float(sm_prev[f * D + q])
                    + 0.25 * float(sm_prev[(f + 1) * D + q])
                )

        def retrieve_cb(frac):
            _progress(progress, 0.5 + 0.2 * frac, MSG_RETRIEVE)

        nb = knn(sf, sv, sa, db, config.neighbours, retrieve_cb)
    else:
        for f in range(nfr):
            sv[f] = vo[min(len(vo) - 1, max(0, js_round((f * H) / HOP)))]

    # -- 5) synthesis --------------------------------------------------------
    if grain:
        o = grain_synth(len(y), f0, vo, nb, config.neighbours, db, ratio, rs, nfr, sf)
    elif voc:
        def syn_cb(frac, message):
            _progress(progress, 0.7 + 0.3 * frac, message)

        o = vocode(len(y), f0, vo, nb, config.neighbours, db, ratio, rs, sf, sa, nfr,
                   config.formant_factor, syn_cb, config.rng_seed)
    else:
        o = np.zeros(len(y) + N, dtype=np.float32)

    # -- 6) envelope transfer + overlap-add (env path only) -----------------
    tl = np.zeros(NB, dtype=np.float32)
    retv = np.zeros(NB, dtype=np.float32)
    prev = np.zeros(NB, dtype=np.float32)
    have = False
    if not direct:
        def transform_cb(frac):
            _progress(progress, 0.7 + 0.3 * frac, MSG_TRANSFORM)

        for f in range(nfr):
            if f % 60 == 0:
                transform_cb(f / nfr)
            p = f * H - N // 2
            rms = analyze(y, p, le)
            if rms >= SIL:
                mixed = le
                if nb is not None and nb[f * config.neighbours] >= 0 and config.envelope_blend > 0.0:
                    retv[:] = 0.0
                    c = 0
                    for q in range(config.neighbours):
                        j = nb[f * config.neighbours + q]
                        if j < 0:
                            continue
                        c += 1
                        e = db.le[j]
                        for k in range(NB):
                            retv[k] = np.float32(float(retv[k]) + float(e[k]))
                    ms = 0.0
                    mr = 0.0
                    for k in range(NB):
                        retv[k] = np.float32(float(retv[k]) / c)
                        ms += float(le[k])
                        mr += float(retv[k])
                    off = (ms - mr) / NB
                    for k in range(NB):
                        r = float(retv[k]) + off
                        if have:
                            r = 0.6 * r + 0.4 * float(prev[k])
                        prev[k] = np.float32(r)
                        tl[k] = np.float32(float(le[k]) * (1.0 - config.envelope_blend) + r * config.envelope_blend)
                    have = True
                    mixed = tl
                for k in range(NB):
                    tv = float(mixed[k])
                    if config.formant_factor != 1.0:
                        u = k / config.formant_factor
                        i = min(NB - 2, int(math.floor(u)))
                        a = min(1.0, u - i)
                        tv = float(mixed[i]) * (1 - a) + float(mixed[i + 1]) * a
                    d = max(-3.0, min(3.0, tv - float(le[k])))
                    if not sv[f]:
                        d *= 0.6
                    g = js_exp(d)
                    Re[k] = np.float32(float(Re[k]) * g)
                    Im[k] = np.float32(float(Im[k]) * g)
                    if 0 < k < N // 2:
                        Re[N - k] = np.float32(float(Re[N - k]) * g)
                        Im[N - k] = np.float32(float(Im[N - k]) * g)
            fft(Re, Im, 1)
            for i in range(N):
                q = p + i
                if 0 <= q < len(o):
                    o[q] = np.float32(float(o[q]) + float(Re[i]) * float(WIN[i]) / 1.5)

    # -- 7) final normalization --------------------------------------------
    res = o[:len(y)].copy()
    pk = 0.0
    for i in range(len(res)):
        if not (res[i] == res[i]):
            res[i] = 0.0
        pk = max(pk, abs(float(res[i])))
    gn = 0.92 / pk if pk > 0.0 else 1.0
    for i in range(len(res)):
        res[i] = np.float32(float(res[i]) * gn)

    return ConversionResult(
        samples=res,
        sample_rate=SR,
        mode="grain" if grain else ("voc" if voc else "env"),
        pitch_ratio=ratio,
        pitch_shifted=pitch_shifted,
        direct_mode=direct,
        f0=f0,
        voiced=vo,
        source_length=len(x),
        target_length=len(t) if t is not None else None,
        n_frames=nfr,
        target_database_active=db is not None,
    )
