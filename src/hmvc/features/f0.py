"""Fundamental frequency and voicing estimation (YIN, NumPy only).

Algorithm (A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency
estimator for speech and music", JASA 111(4), 2002):

1. difference function ``d(tau) = sum_j (x_j - x_{j+tau})^2`` over an
   integration window of ``W = ceil(fs / f0_floor)`` samples, computed for all
   frames at once with FFT cross-correlation and cumulative energies;
2. cumulative-mean-normalised difference ``d'(tau)``;
3. the first ``tau`` in ``[fs/f0_ceil, fs/f0_floor]`` with
   ``d' < yin_threshold``, descended to its local minimum (the global minimum
   when no dip crosses the threshold);
4. parabolic interpolation of ``d`` around that lag (sub-sample period);
5. continuity tracking (default on): the ``n_candidates`` best local CMND
   minima of every frame (parabolically interpolated) are candidates and a
   Viterbi search over each run of voicing-eligible frames minimises
   ``sum CMND + OCTAVE_COST * log2 T + jump_cost * |d ln T|``.  This
   removes YIN's sub-octave picks (first dip below threshold at 2*T0) and
   short formant-period (T0/3, T0/4) excursions;
6. refinement (HMVC addition, default on): for voiced frames the normalised
   difference function is re-evaluated around the coarse period with a short
   window of ``refine_periods`` periods placed *symmetrically* about the frame
   centre (the pairs ``x_j, x_{j+tau}`` are centred on ``t * hop``), followed by
   another parabolic fit.  YIN's long fixed window smears gliding / vibrato F0
   and is centred half a period late; on the synthetic vibrato vowels this
   step reduces the median error from 1.5-4.8 to 0.2-2.1 cents.  (Low-pass
   pre-filtering of the YIN input was also evaluated and brought no gain.)

Voicing is decided separately from the period search, with hysteresis: frames
whose aperiodicity ``d'(tau*)`` is below ``voicing_threshold`` seed voiced
regions, which then grow into neighbouring frames whose aperiodicity is below
the looser ``extension_threshold`` *and* whose period candidate is continuous
with the neighbour's (``|ln(f_t / f_{t+-1})| < max_jump``).  Every voiced frame
must also be above ``absolute_silence_db`` (dBFS) and ``silence_threshold_db``
relative to the loudest frame.  ``periodicity = 1 - d'(tau*)`` of the
tracked candidate (clipped to [0, 1]) is returned as a continuous
voicing-strength feature.  Post-processing: a median filter of
``median_frames`` over each voiced run (removes isolated octave spikes) and
removal of voiced runs shorter than ``min_voiced_frames``.

Measured on real speech (CMU ARCTIC arctic_a0007, ``tools/evaluate_features.py``),
continuity tracking plus hysteresis moved the agreement with WORLD from
voicing 0.74 / 0.87 (Harvest / DIO) and gross pitch error 3.1 % / 1.6 % to
voicing 0.80 / 0.91 and GPE 1.9 % / 0.3 %.  Harvest and DIO themselves agree on
voicing for 0.81 of the frames and differ by 1.5 % GPE.  White noise stays
unvoiced (its aperiodicity is ~0.65 or more and its candidates are not
continuous); on the synthetic signals the only frames voiced outside the true
voiced regions are single frames adjacent to a voicing boundary, where the
analysis window already overlaps the voiced segment.

The HMS engine's ``autocorrelation_f0`` was evaluated and not reused: it picks
integer lags only (at 16 kHz a 440 Hz tone reads 444.4 Hz, +1 % / 17 cents;
the quantisation error grows with F0, up to 3.8 % between 600 and 790 Hz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hmvc.features.config import next_power_of_two
from hmvc.features.framing import blocks, frame_signal


@dataclass
class PitchTrack:
    f0: np.ndarray            # (T,) Hz, 0.0 where unvoiced
    voiced: np.ndarray        # (T,) bool
    periodicity: np.ndarray   # (T,) in [0, 1]; 1 - YIN aperiodicity
    candidate_f0: np.ndarray  # (T,) best period estimate even when unvoiced (Hz)
    level_db: np.ndarray      # (T,) RMS of the YIN integration window, dBFS


def yin_lags(sample_rate: int, f0_floor: float, f0_ceil: float):
    """``(tau_min, tau_max, window)`` in samples."""
    tau_max = int(math.ceil(sample_rate / f0_floor))
    tau_min = max(2, int(math.floor(sample_rate / f0_ceil)))
    return tau_min, tau_max, tau_max


def _yin_block(frames: np.ndarray, window: int, tau_max: int):
    """Difference function and CMND ``(B, tau_max + 2)`` for frames ``(B, L)``."""
    n_lags = tau_max + 2
    n_fft = next_power_of_two(frames.shape[1] + window)
    spec_full = np.fft.rfft(frames, n_fft, axis=1)
    spec_head = np.fft.rfft(frames[:, :window], n_fft, axis=1)
    acf = np.fft.irfft(np.conj(spec_head) * spec_full, n_fft, axis=1)[:, :n_lags]
    sq = np.concatenate([np.zeros((frames.shape[0], 1)),
                         np.cumsum(frames ** 2, axis=1)], axis=1)
    e0 = sq[:, window]
    lags = np.arange(n_lags)
    e_tau = sq[:, lags + window] - sq[:, lags]
    diff = np.maximum(e0[:, None] + e_tau - 2.0 * acf, 0.0)
    diff[:, 0] = 0.0
    cum = np.cumsum(diff[:, 1:], axis=1)
    cmnd = np.ones_like(diff)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = diff[:, 1:] * lags[1:][None, :] / cum
    # Silent / constant frames: cumulative sum 0 -> undefined -> aperiodic.
    cmnd[:, 1:] = np.where(cum > 1e-12 * np.maximum(e0[:, None], 1e-300) + 1e-300, ratio, 1.0)
    cmnd = np.where(np.isfinite(cmnd), cmnd, 1.0)
    return diff, cmnd, e0


def _pick(diff: np.ndarray, cmnd: np.ndarray, tau_min: int, tau_max: int,
          threshold: float):
    rows = np.arange(cmnd.shape[0])
    search = cmnd[:, tau_min:tau_max + 1]
    below = search < threshold
    has_dip = below.any(axis=1)
    tau = np.where(has_dip, np.argmax(below, axis=1), np.argmin(search, axis=1)) + tau_min
    # Descend to the local minimum following the first dip.
    for _ in range(tau_max - tau_min + 1):
        nxt = np.minimum(tau + 1, tau_max)
        move = has_dip & (tau < tau_max) & (cmnd[rows, nxt] < cmnd[rows, tau])
        if not move.any():
            break
        tau = np.where(move, nxt, tau)
    aperiodicity = cmnd[rows, tau]
    # Parabolic interpolation on the raw difference function.
    left = diff[rows, tau - 1]
    mid = diff[rows, tau]
    right = diff[rows, tau + 1]
    denom = left - 2.0 * mid + right
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(np.abs(denom) > 1e-20, 0.5 * (left - right) / denom, 0.0)
    shift = np.clip(np.where(np.isfinite(shift), shift, 0.0), -1.0, 1.0)
    return tau + shift, aperiodicity


#: Praat-style octave cost (per octave of period) that breaks near-ties between
#: a period and its multiples in favour of the shorter period.
OCTAVE_COST = 0.02


def _candidates(diff: np.ndarray, cmnd: np.ndarray, tau_min: int, tau_max: int,
                k: int):
    """Up to ``k`` lowest local CMND minima per frame -> (periods, costs) ``(B, k)``.

    Periods are refined by parabolic interpolation of ``diff``, costs are the
    parabolically interpolated CMND minima (an integer-lag CMND favours
    multiples of the period that happen to fall on integer lags, e.g. 3 T0 of
    a 750 Hz tone at 16 kHz).  Missing candidates have cost ``inf``.
    """
    lags = np.arange(tau_min, tau_max + 1)

    def parabola(curve):
        left, centre, right = curve[:, lags - 1], curve[:, lags], curve[:, lags + 1]
        denom = left - 2.0 * centre + right
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(denom > 1e-20, 0.5 * (left - right) / denom, 0.0)
        shift = np.clip(np.nan_to_num(shift), -1.0, 1.0)
        return shift, centre - 0.25 * (left - right) * shift

    mid = cmnd[:, lags]
    is_min = (mid < cmnd[:, lags - 1]) & (mid <= cmnd[:, lags + 1])
    shift, _ = parabola(diff)
    _, minimum = parabola(cmnd)
    values = np.where(is_min, np.maximum(minimum, 0.0), np.inf)
    # Select on the same biased cost the tracker minimises, so that a period is
    # not crowded out by its (equally good) multiples.
    biased = values + OCTAVE_COST * np.log2(lags)[None, :]
    k = min(k, lags.size)
    order = np.argpartition(biased, k - 1, axis=1)[:, :k] if k < lags.size \
        else np.argsort(biased, axis=1)
    periods = lags[order] + np.take_along_axis(shift, order, axis=1)
    costs = np.take_along_axis(values, order, axis=1)
    return periods, costs


def _viterbi(periods: np.ndarray, costs: np.ndarray, jump_cost: float,
             octave_cost: float = OCTAVE_COST) -> np.ndarray:
    """Minimum-cost candidate path for one run.

    Cost = sum of CMND + ``octave_cost * log2(T)`` + ``jump_cost * |d ln T|``.
    """
    n = periods.shape[0]
    log_p = np.log(np.maximum(periods, 1e-9))
    costs = costs + octave_cost * log_p / np.log(2.0)
    total = costs[0].copy()
    back = np.zeros(periods.shape, dtype=np.int64)
    for t in range(1, n):
        trans = total[:, None] + jump_cost * np.abs(log_p[t - 1][:, None] - log_p[t][None, :])
        back[t] = np.argmin(trans, axis=0)
        total = trans[back[t], np.arange(periods.shape[1])] + costs[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return periods[np.arange(n), path]


def _refine(x: np.ndarray, fs: int, hop: int, frames: np.ndarray, period: np.ndarray,
            periods: float, span: int = 2) -> np.ndarray:
    """Symmetric short-window refinement of ``period`` for frame indices ``frames``."""
    n = x.size
    t_int = np.round(period).astype(np.int64)
    width = np.maximum(np.round(periods * period).astype(np.int64), 8)
    offsets = np.arange(-span - 1, span + 2)
    lags = t_int[:, None] + offsets[None, :]                     # (B, K)
    lags = np.maximum(lags, 1)
    centre = frames.astype(np.int64) * int(hop)
    j = np.arange(int(width.max()))[None, :]
    inside_w = j < width[:, None]
    dist = np.empty(lags.shape)
    for k in range(lags.shape[1]):
        tau = lags[:, k]
        start = centre - (width + tau) // 2
        ia = start[:, None] + j
        ib = ia + tau[:, None]
        a = np.where((ia >= 0) & (ia < n) & inside_w, x[np.clip(ia, 0, n - 1)], 0.0)
        b = np.where((ib >= 0) & (ib < n) & inside_w, x[np.clip(ib, 0, n - 1)], 0.0)
        energy = np.sum(a * a + b * b, axis=1)
        dist[:, k] = np.where(energy > 1e-20, np.sum((a - b) ** 2, axis=1)
                              / np.maximum(energy, 1e-20), 1.0)
    rows = np.arange(lags.shape[0])
    best = np.argmin(dist[:, 1:-1], axis=1) + 1
    left, mid, right = dist[rows, best - 1], dist[rows, best], dist[rows, best + 1]
    denom = left - 2.0 * mid + right
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(np.abs(denom) > 1e-20, 0.5 * (left - right) / denom, 0.0)
    refined = lags[rows, best] + np.clip(np.nan_to_num(shift), -1.0, 1.0)
    # Keep the coarse estimate if the refinement wandered (should not happen).
    return np.where(np.abs(refined / period - 1.0) < 0.05, refined, period)


def _grow(seed: np.ndarray, eligible: np.ndarray, log_f: np.ndarray,
          max_jump: float) -> np.ndarray:
    """Grow ``seed`` regions into ``eligible`` frames with a continuous F0 candidate."""
    voiced = seed.copy()
    if max_jump <= 0 or not voiced.any():
        return voiced
    step = np.abs(np.diff(log_f)) < max_jump                 # continuity t <-> t+1
    left_ok = eligible[:-1] & step                           # frame t can join from t+1
    right_ok = eligible[1:] & step                           # frame t+1 can join from t
    while True:
        new = np.zeros_like(voiced)
        new[:-1] |= left_ok & voiced[1:] & ~voiced[:-1]
        new[1:] |= right_ok & voiced[:-1] & ~voiced[1:]
        if not new.any():
            return voiced
        voiced |= new


def _runs(mask: np.ndarray):
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _median_in_runs(f0: np.ndarray, voiced: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return f0
    out = f0.copy()
    half = width // 2
    for start, stop in _runs(voiced):
        run = f0[start:stop]
        if run.size < 3:
            continue
        padded = np.pad(run, (half, half), mode="edge")
        view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
        out[start:stop] = np.median(view, axis=1)
    return out


def estimate_f0(x: np.ndarray, sample_rate: int, hop: int, f0_floor: float = 60.0,
                f0_ceil: float = 800.0, yin_threshold: float = 0.15,
                voicing_threshold: float = 0.25, silence_threshold_db: float = -50.0,
                absolute_silence_db: float = -80.0, median_frames: int = 3,
                min_voiced_frames: int = 3, refine_periods: float = 2.0,
                extension_threshold: float = 0.6, max_jump: float = 0.2,
                jump_cost: float = 2.0, n_candidates: int = 8,
                block: int = 512) -> PitchTrack:
    """YIN pitch track on the shared frame grid (frames centred at ``t * hop``)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n_frames = 0 if x.size == 0 else 1 + x.size // int(hop)
    empty = np.zeros(n_frames)
    if n_frames == 0:
        return PitchTrack(empty, empty.astype(bool), empty, empty, empty)
    tau_min, tau_max, window = yin_lags(sample_rate, f0_floor, f0_ceil)
    if tau_min >= tau_max:
        raise ValueError("f0 range too narrow for this sample rate")
    frame_length = window + tau_max + 2
    period = np.empty(n_frames)
    aperiodicity = np.empty(n_frames)
    energy = np.empty(n_frames)
    cand_period = np.empty((n_frames, n_candidates))
    cand_cost = np.empty((n_frames, n_candidates))
    for start, stop in blocks(n_frames, block):
        # Centre the integration window (not the whole frame) on t * hop.
        frames = frame_signal(x, frame_length, hop, stop - start, start, left=window // 2)
        diff, cmnd, e0 = _yin_block(frames, window, tau_max)
        p, ap = _pick(diff, cmnd, tau_min, tau_max, yin_threshold)
        period[start:stop], aperiodicity[start:stop], energy[start:stop] = p, ap, e0
        if jump_cost > 0:
            cp, cc = _candidates(diff, cmnd, tau_min, tau_max, n_candidates)
            width = cp.shape[1]
            cand_period[start:stop, :width], cand_cost[start:stop, :width] = cp, cc
            cand_period[start:stop, width:], cand_cost[start:stop, width:] = 1.0, np.inf
    level_db = 10.0 * np.log10(np.maximum(energy / window, 1e-20))
    gate = max(absolute_silence_db, float(level_db.max()) + silence_threshold_db)
    allowed = level_db > gate
    eligible = allowed & (aperiodicity < max(extension_threshold, voicing_threshold))
    if jump_cost > 0:
        # Continuity tracking over every run of voicing-eligible frames.
        for lo, hi in _runs(eligible):
            costs = cand_cost[lo:hi]
            ok = np.isfinite(costs).any(axis=1)
            if ok.all():
                period[lo:hi] = _viterbi(cand_period[lo:hi], costs, jump_cost)
        rows = np.arange(n_frames)
        # Report the CMND of the tracked candidate as the frame's aperiodicity.
        match = np.abs(cand_period - period[:, None]) < 1e-9     # exact copies
        tracked = np.where(match.any(axis=1),
                           cand_cost[rows, np.argmax(match, axis=1)], aperiodicity)
        aperiodicity = np.where(eligible, np.minimum(tracked, 1.0), aperiodicity)
    candidate = sample_rate / np.maximum(period, 1.0)
    in_range = (candidate >= f0_floor * 0.97) & (candidate <= f0_ceil * 1.03)
    allowed &= in_range
    voiced = (aperiodicity < voicing_threshold) & allowed
    voiced = _grow(voiced, allowed & (aperiodicity < max(extension_threshold, voicing_threshold)),
                   np.log(candidate), max_jump)
    f0 = _median_in_runs(np.where(voiced, candidate, 0.0), voiced, int(median_frames))
    if min_voiced_frames > 1:
        for start, stop in _runs(voiced):
            if stop - start < min_voiced_frames:
                voiced[start:stop] = False
    if refine_periods > 0 and voiced.any():
        idx = np.flatnonzero(voiced)
        refined = np.empty(idx.size)
        for lo, hi in blocks(idx.size, block):
            refined[lo:hi] = sample_rate / _refine(x, sample_rate, hop, idx[lo:hi],
                                                   sample_rate / f0[idx[lo:hi]],
                                                   refine_periods)
        f0[idx] = refined
    f0 = np.where(voiced, f0, 0.0)
    periodicity = np.clip(1.0 - aperiodicity, 0.0, 1.0)
    return PitchTrack(f0=f0, voiced=voiced, periodicity=periodicity,
                      candidate_f0=candidate, level_db=level_db)


def continuous_log_f0(f0: np.ndarray, voiced: np.ndarray, fill_hz: float) -> np.ndarray:
    """Natural-log F0 linearly interpolated across unvoiced frames.

    Edges are held at the nearest voiced value; an utterance with no voiced
    frame is filled with ``log(fill_hz)`` so the stream is always finite.
    """
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    voiced = np.asarray(voiced, dtype=bool).reshape(-1) & (f0 > 0)
    if f0.size == 0:
        return np.zeros(0)
    if not voiced.any():
        return np.full(f0.size, math.log(fill_hz))
    idx = np.flatnonzero(voiced)
    return np.interp(np.arange(f0.size), idx, np.log(f0[idx]))
