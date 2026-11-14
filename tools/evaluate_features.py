#!/usr/bin/env python3
"""Representation study and reference evaluation for HMVC Phase 1 features.

Maintainer tool; writes ``docs/feature-extraction/evaluation.json``, the evidence
behind ``docs/feature-extraction.md``.  Run it in a separate environment that
also has ``pyworld``, ``pysptk`` and ``librosa`` (used only as references)::

    PYTHONPATH=src python tools/evaluate_features.py --write

Sections
--------
envelope_study        envelope estimators vs the analytically known envelope of
                      additive-synthesis vowels across F0 (accuracy and F0
                      invariance), including HMS's builtin lifter and WORLD
representation_study  round-trip loss vs dimensionality for MCEP (all-pass
                      warped), linear cepstrum, HMS's mel-filterbank DCT and
                      the log spectrum; decorrelation; F0 sensitivity of SPTK's
                      UELS ``mcep`` vs envelope-based MCEP
aperiodicity          band aperiodicity vs known noise fractions (and WORLD D4C)
f0_synthetic          F0/voicing accuracy on signals with known F0
speech                real speech (CMU ARCTIC arctic_a0007, bundled with pysptk):
                      F0/voicing vs Harvest/DIO, CheapTrick vs WORLD on the
                      same F0, MCEP vs pysptk, speed vs pyworld
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests" / "_support"))

import feature_signals as S  # noqa: E402
from hmvc.features import AnalysisConfig, FeatureExtractor, read_wav  # noqa: E402
from hmvc.features import cepstrum as C  # noqa: E402
from hmvc.features import envelope as E  # noqa: E402
from hmvc.features import f0 as F0  # noqa: E402
from hmvc.features.framing import frame_signal, get_window, power_spectrum  # noqa: E402

OUT = PROJECT / "docs" / "feature-extraction" / "evaluation.json"
FS = 16000
HOP = 80
NFFT = 1024


def r(v, n=3):
    return None if v is None else round(float(v), n)


def lsd_shape(est, true, band):
    """Gain-removed RMS log-spectral distance (dB) over a bin mask."""
    return C.log_spectral_distance(est[:, band], true[:, band], remove_gain=True)


def steady_vowel(f0, formants, fs=FS, duration=0.6, noise_db=None, seed=0):
    sig = S.vowel(fs, duration, f0, f0, 0.0, 0.0, formants, noise_db, 0.3, seed)
    return sig["x"], sig["f0"]


# --------------------------------------------------------------------------
# Envelope study
# --------------------------------------------------------------------------


def envelope_study():
    from hmvc.engine.vocoder.builtin import BuiltinVocoder
    import pyworld

    freqs = np.linspace(0, FS / 2, NFFT // 2 + 1)
    band = (freqs >= 80) & (freqs <= 0.9 * FS / 2)
    f0s = (100.0, 150.0, 220.0, 300.0, 400.0)
    window = get_window("hann", 640)                       # 40 ms fixed window
    results = {}
    for vowel_name, formants in (("a", S.FORMANTS_A), ("i", S.FORMANTS_I)):
        true = S.formant_amplitude(freqs, FS, formants) ** 2
        per_method = {}
        shapes = {}
        for f0 in f0s:
            x, f0_samples = steady_vowel(f0, formants)
            tf = S.frame_f0(f0_samples, HOP)
            sl = slice(10, -10)                             # skip edges
            frames = frame_signal(x, 640, HOP)
            power = power_spectrum(frames, window, NFFT)
            ests = {
                "cheaptrick (HMVC)": E.cheaptrick(x, FS, tf, HOP, NFFT),
                "true_envelope (HMVC)": E.true_envelope(power, FS, tf),
                "lifter (HMVC, fs/2F0)": E.lifter_envelope(power, FS, tf),
                "cheaptrick (WORLD)": pyworld.cheaptrick(x, tf, np.arange(len(tf)) * 0.005, FS,
                                                         fft_size=NFFT, f0_floor=60.0),
            }
            hms = BuiltinVocoder(fft_size=NFFT, fs=FS, frame_period=5.0)
            _, sp_hms, _ = hms.analyze(x, fs=FS, frame_period=5.0)
            ests["HMS builtin lifter (order 40)"] = sp_hms
            for name, est in ests.items():
                est = est[sl]
                d = lsd_shape(est, np.broadcast_to(true, est.shape), band)
                per_method.setdefault(name, {})[str(int(f0))] = r(np.mean(d), 2)
                logm = 10 * np.log10(np.maximum(est, 1e-30)).mean(axis=0)
                shapes.setdefault(name, []).append(logm[band] - logm[band].mean())
        invariance = {name: r(np.mean(np.std(np.stack(v), axis=0)), 2)
                      for name, v in shapes.items()}
        results[vowel_name] = {"lsd_db_by_f0": per_method,
                               "f0_invariance_std_db": invariance}
    return {"description": "Gain-removed RMS log-spectral distance (dB, 80 Hz-7.2 kHz) "
                           "between estimated and true power envelope of steady additive-"
                           "synthesis vowels, given the true F0; and the mean over frequency "
                           "of the std-dev across F0 of the estimated envelope shape "
                           "(F0 invariance; lower is better).",
            "f0_hz": list(f0s), "fixed_window_ms": 40, "results": results}


# --------------------------------------------------------------------------
# Representation study
# --------------------------------------------------------------------------


def corpus_envelopes(speech_x):
    """CheapTrick envelopes of real speech (voiced and unvoiced) + synthetic vowels."""
    feats = FeatureExtractor(AnalysisConfig(keep_envelope=True)).extract(speech_x, FS)
    speech_env = feats.spectral_envelope[feats.speech]
    synth = []
    for f0 in (110.0, 180.0, 260.0):
        for formants in (S.FORMANTS_A, S.FORMANTS_I):
            x, f0s = steady_vowel(f0, formants, noise_db=-40.0)
            synth.append(E.cheaptrick(x, FS, S.frame_f0(f0s, HOP), HOP, NFFT)[10:-10:4])
    return speech_env, np.concatenate(synth), feats


def representation_study(speech_x):
    from hmvc.engine.core.features import mcep_to_power, power_to_mcep

    speech_env, synth_env, feats = corpus_envelopes(speech_x)
    alpha = C.mcep_alpha(FS)
    freqs = np.linspace(0, FS / 2, NFFT // 2 + 1)
    melw = C.mel_weights(len(freqs), FS)
    dims = (13, 20, 25, 30, 35, 40, 50, 60)
    out = {}
    for label, env in (("speech_cheaptrick", speech_env), ("synthetic_vowels", synth_env)):
        table = {}
        for d in dims:
            rec = {
                "mcep_allpass (alpha=%.3f)" % alpha: C.mc2sp(C.sp2mc(env, d - 1, alpha), alpha, NFFT),
                "linear_cepstrum (alpha=0)": C.mc2sp(C.sp2mc(env, d - 1, 0.0), 0.0, NFFT),
                "HMS mel-filterbank DCT": mcep_to_power(power_to_mcep(env, NFFT, FS, d), NFFT, FS, d),
            }
            for name, est in rec.items():
                table.setdefault(name, {})[str(d)] = {
                    "lsd_db": r(np.mean(C.log_spectral_distance(est, env)), 3),
                    "mel_weighted_lsd_db": r(np.mean(C.log_spectral_distance(est, env, weights=melw)), 3),
                }
        out[label] = {"frames": int(env.shape[0]), "by_dimension": table}

    # Where the error goes: per-band round-trip error on speech envelopes.
    bands = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000))
    per_band = {}
    for d in (25, 35, 40):
        for a in (0.0, alpha, 0.55):
            rec = C.mc2sp(C.sp2mc(speech_env, d - 1, a), a, NFFT)
            err = 10 * np.log10(rec / speech_env)
            per_band[f"D={d} alpha={a:.2f}"] = {
                f"{lo}-{hi} Hz": r(np.sqrt(np.mean(err[:, (freqs >= lo) & (freqs < hi)] ** 2)), 2)
                for lo, hi in bands}
    out["speech_band_rms_error_db"] = per_band

    # Decorrelation of candidate representations on speech frames.
    def offdiag(m):
        cc = np.corrcoef(m, rowvar=False)
        mask = ~np.eye(cc.shape[0], dtype=bool)
        return r(np.mean(np.abs(cc[mask])), 3)

    order = 24
    decor = {
        "log_spectrum (513 bins)": offdiag(np.log(speech_env)),
        "mcep c1..c24": offdiag(C.sp2mc(speech_env, order, alpha)[:, 1:]),
        "HMS mel-filterbank DCT c1..c24": offdiag(power_to_mcep(speech_env, NFFT, FS, order + 1)[:, 1:]),
    }

    # F0 sensitivity: SPTK UELS mcep on windowed frames vs envelope-based MCEP.
    import pysptk
    sens = {}
    for formants_name, formants in (("a", S.FORMANTS_A), ("i", S.FORMANTS_I)):
        reps = {"uels": {}, "envelope": {}}
        true = S.formant_amplitude(freqs, FS, formants) ** 2
        true_mc = C.sp2mc(true[None, :], order, alpha)[0]
        for f0 in (100.0, 200.0, 300.0):
            # Noiseless: the true envelope has -120 dB valleys that any noise fills.
            x, f0s = steady_vowel(f0, formants, noise_db=None)
            frames = frame_signal(x, 640, HOP)[10:-10:5] * np.blackman(640)
            frames = np.pad(frames, ((0, 0), (0, NFFT - 640)))   # SPTK needs 2^k length
            rows = []
            for fr in frames:
                try:
                    rows.append(pysptk.mcep(fr, order=order, alpha=alpha, maxiter=200,
                                            etype=1, eps=1e-8))
                except RuntimeError:          # SPTK's Toeplitz solve may fail to converge
                    pass
            uels = np.array(rows)
            env = E.cheaptrick(x, FS, S.frame_f0(f0s, HOP), HOP, NFFT)[10:-10:5]
            reps["uels"][f0] = uels.mean(axis=0)
            reps["envelope"][f0] = C.sp2mc(env, order, alpha).mean(axis=0)
        sens[formants_name] = {
            kind: {"mcd_100_vs_300_hz_db": r(C.mel_cepstral_distortion(v[100.0][None], v[300.0][None])[0], 2),
                   "mcd_to_true_envelope_db": {str(int(k)): r(C.mel_cepstral_distortion(vv[None], true_mc[None])[0], 2)
                                               for k, vv in v.items()}}
            for kind, v in reps.items()}
    return {"round_trip": out, "decorrelation_mean_abs_offdiag_corr": decor,
            "f0_sensitivity": sens,
            "description": "Round-trip RMS log-spectral distance (dB, all bins; and mel-"
                           "weighted) of envelope -> representation with D coefficients -> "
                           "envelope. MCD values exclude c0."}


# --------------------------------------------------------------------------
# Aperiodicity
# --------------------------------------------------------------------------


def aperiodicity_study():
    import pyworld
    from hmvc.features import aperiodicity as A

    rng = np.random.default_rng(5)
    n = int(0.6 * FS)
    t = np.arange(n) / FS
    edges = A.band_edges(5, FS)
    out = {}
    for contour in ("steady", "vibrato"):
        f0 = np.full(n, 150.0) if contour == "steady" else \
            150.0 * 2 ** (50 / 1200 * np.sin(2 * np.pi * 5.5 * t))
        harm = S.harmonic_signal(f0, FS, lambda f: np.ones_like(f))
        harm /= np.sqrt(np.mean(harm ** 2))
        noise = rng.standard_normal(n)
        tf = S.frame_f0(f0, HOP)
        rows = {}
        for frac in (0.0, 0.1, 0.3, 0.5, 0.8):
            x = np.sqrt(1 - frac) * harm + np.sqrt(frac) * noise
            x *= 0.2
            entry = {}
            for periods in (3.0, 4.0, 6.0):
                ap = A.band_aperiodicity(x, FS, tf, HOP, edges, periods)[10:-10]
                entry[f"hmvc_periods_{periods:g}"] = r(np.mean(ap[:, :4]), 3)  # bands < 4.2 kHz
            # threshold=0 disables D4C's own voicing gate, which otherwise marks
            # these flat-spectrum frames unvoiced and returns aperiodicity 1.
            d4c = pyworld.d4c(x, tf, np.arange(len(tf)) * 0.005, FS, fft_size=NFFT,
                              threshold=0.0)[10:-10]
            freqs = np.linspace(0, FS / 2, d4c.shape[1])
            entry["world_d4c"] = r(np.mean(d4c[:, (freqs > 0) & (freqs < edges[4])]), 3)
            rows[str(frac)] = entry
        out[contour] = rows
    return {"description": "Mean estimated aperiodicity (bands below ~4.2 kHz) for a flat "
                           "harmonic spectrum mixed with white noise at a known noise power "
                           "fraction (the key).  Ideal value equals the key.",
            "results": out}


# --------------------------------------------------------------------------
# F0 on synthetic signals
# --------------------------------------------------------------------------


def f0_scores(f0, voiced, truth):
    tv = truth > 0
    both = voiced & tv
    cents = 1200 * np.abs(np.log2(f0[both] / truth[both])) if both.any() else np.array([np.nan])
    return {"median_cents": r(np.median(cents), 2), "p95_cents": r(np.percentile(cents, 95), 2),
            "gross_error_rate": r(np.mean(cents > 315.6), 4),
            "voiced_recall": r(both.sum() / max(tv.sum(), 1), 3),
            "false_voiced_frames": int((voiced & ~tv).sum())}


def f0_synthetic():
    import pyworld
    from hmvc.engine.core.dsp import autocorrelation_f0

    out = {}
    for name, (x, f0s, fs) in S.reference_signals().items():
        hop = int(round(fs * 0.005))
        truth = S.frame_f0(f0s, hop)
        feats = FeatureExtractor().extract(x, fs)
        row = {"hmvc_yin_tracked": f0_scores(feats.f0, feats.voiced, truth)}
        if fs == 16000:   # WORLD's frame grid equals HMVC's only for integer 5 ms hops
            fh, _ = pyworld.harvest(x, fs, 60.0, 800.0, 5.0)
            fd, td = pyworld.dio(x, fs, 60.0, 800.0, 5.0)
            fd = pyworld.stonemask(x, fd, td, fs)
            row["world_harvest"] = f0_scores(fh, fh > 0, truth)
            row["world_dio_stonemask"] = f0_scores(fd, fd > 0, truth)
            fa = autocorrelation_f0(x, fs, 4 * hop, hop, 60.0, 800.0)[:len(truth)]
            row["hms_autocorrelation"] = f0_scores(fa, fa > 0, truth[:len(fa)])
        out[name] = row
    return out


# --------------------------------------------------------------------------
# Real speech
# --------------------------------------------------------------------------


def speech_section(x, fs):
    import librosa
    import pysptk
    import pyworld

    def agree(f0a, va, f0b, vb):
        both = va & vb
        cents = 1200 * np.abs(np.log2(f0a[both] / f0b[both]))
        return {"voicing_agreement": r(np.mean(va == vb), 3),
                "median_cents": r(np.median(cents), 2), "p90_cents": r(np.percentile(cents, 90), 1),
                "gross_error_rate": r(np.mean(cents > 315.6), 4), "both_voiced_frames": int(both.sum())}

    feats = FeatureExtractor(AnalysisConfig(keep_envelope=True)).extract(x, fs)
    fh, tpos = pyworld.harvest(x, fs, 60.0, 800.0, 5.0)
    fd, td = pyworld.dio(x, fs, 60.0, 800.0, 5.0)
    fd = pyworld.stonemask(x, fd, td, fs)
    fp, vp, _ = librosa.pyin(x, fmin=60.0, fmax=800.0, sr=fs, frame_length=1024, hop_length=HOP,
                             center=True)
    fp = np.nan_to_num(fp)
    vp = vp & (fp > 0)
    hv = fh > 0
    dv = fd > 0
    f0 = {
        "hmvc_vs_harvest": agree(feats.f0, feats.voiced, fh, hv),
        "hmvc_vs_dio": agree(feats.f0, feats.voiced, fd, dv),
        "harvest_vs_dio (reference spread)": agree(fh, hv, fd, dv),
        "librosa_pyin_vs_harvest": agree(fp, vp, fh, hv),
        "librosa_pyin_vs_dio": agree(fp, vp, fd, dv),
        "voiced_fraction": {"hmvc": r(feats.voiced.mean()), "harvest": r(hv.mean()),
                            "dio": r(dv.mean()), "pyin": r(vp.mean())},
    }
    # Same-F0 envelope agreement with WORLD on real speech (HMVC F0 fed to both).
    sp_world = pyworld.cheaptrick(x - x.mean(), feats.f0, tpos, fs, q1=-0.15, f0_floor=60.0,
                                  fft_size=feats.fft_size)
    above = sp_world > 1e-9
    diff = np.abs(10 * np.log10(feats.spectral_envelope / sp_world))
    alpha = feats.mcep_alpha
    mc_ref = pysptk.sp2mc(sp_world, feats.mcep_order, alpha)
    mc_ours = C.sp2mc(sp_world, feats.mcep_order, alpha)
    envelope = {"cheaptrick_vs_world_max_abs_db_above_floor": float(f"{diff[above].max():.3g}"),
                "cheaptrick_vs_world_mean_abs_db_above_floor": float(f"{diff[above].mean():.3g}"),
                "fraction_bins_above_floor": r(above.mean(), 4),
                "sp2mc_vs_pysptk_max_abs": float(f"{np.abs(mc_ours - mc_ref).max():.3g}"),
                "mcep_roundtrip_lsd_db": r(np.mean(C.log_spectral_distance(
                    C.mc2sp(feats.mcep, alpha, feats.fft_size),
                    feats.spectral_envelope)), 3)}

    # Speed (best of 3) on the 4 s utterance.
    def best(fn, n=3):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    dur = x.size / fs
    ext = FeatureExtractor()
    t_hmvc = best(lambda: ext.extract(x, fs))

    def world_harvest():
        f, tt = pyworld.harvest(x, fs, 60.0, 800.0, 5.0)
        pyworld.cheaptrick(x, f, tt, fs)
        pyworld.d4c(x, f, tt, fs)

    def world_dio():
        f, tt = pyworld.dio(x, fs, 60.0, 800.0, 5.0)
        f = pyworld.stonemask(x, f, tt, fs)
        pyworld.cheaptrick(x, f, tt, fs)
        pyworld.d4c(x, f, tt, fs)

    speed = {"audio_seconds": r(dur, 3),
             "hmvc_full_extraction_s": r(t_hmvc, 4), "hmvc_x_realtime": r(dur / t_hmvc, 1),
             "world_harvest_cheaptrick_d4c_s": r(best(world_harvest), 4),
             "world_dio_stonemask_cheaptrick_d4c_s": r(best(world_dio), 4),
             "note": "single-threaded; pyworld is compiled C++, HMVC is NumPy"}
    return {"file": "arctic_a0007.wav (CMU ARCTIC, 16 kHz, from the pysptk package)",
            "f0_and_voicing": f0, "envelope_and_mcep": envelope, "speed": speed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--speech", help="16 kHz WAV (default: pysptk's arctic_a0007.wav)")
    args = parser.parse_args()
    import pysptk

    audio = read_wav(args.speech or pysptk.util.example_audio_file())
    if audio.sample_rate != FS:
        raise SystemExit("the speech file must be sampled at 16 kHz")
    x = audio.samples
    t0 = time.perf_counter()
    result = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/evaluate_features.py",
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("numpy", "pyworld", "pysptk", "librosa")},
        "default_config": AnalysisConfig().to_dict(),
        "envelope_study": envelope_study(),
        "representation_study": representation_study(x),
        "aperiodicity": aperiodicity_study(),
        "f0_synthetic": f0_synthetic(),
        "speech": speech_section(x, FS),
    }
    result["runtime_s"] = r(time.perf_counter() - t0, 1)
    text = json.dumps(result, indent=1)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print("wrote", OUT)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
