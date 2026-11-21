# HMVC 2.0 — Phase 1: Strict 1:1 voice-converter port

**Status:** pipeline breakdown (before implementation). This document is
maintained as the phase proceeds; the fidelity appendix at the bottom is
finalised once the behavioral comparison has been performed.

## 1. Source

- File: `voice-converter (4).html` (temporary reference file, kept until the
  phase is verified and finalized).
- Browser: self-contained single-page app; no external requests. DSP runs at
  **SR = 22050 Hz**, **N = 1024** (FFT), **H = 256** (analysis hop), **HOP = 320**
  (pitch-tracking hop), **NB = 513** (spectral bins), **D = 12** (DCT features),
  **NM = 34** (mel bands), **LIF = 36** (cepstral lifter order).
- Output: 16-bit PCM WAV at 22050 Hz, peak-normalized to 0.92, divided by
  1.2247 for harmonic synthesis and 1.5 for the overlap-add formant path.

## 2. Pipeline in exact execution order

For each conversion (source `x`, optional target `t`):

1. **Audio preparation** — decode to 22050 Hz mono (browser: `decodeAudioData`
   + `OfflineAudioContext` render; HMVC port: WAV read + `hmvc.features.audio`
   resampling — see §5 discrepancies). Target is truncated to 150 s.
2. **Pitch tracking** (`trackPitch`): normalised autocorrelation (W = 800,
   lmin/lmax from fmin/fmax, silence gate e0/W < 2.5e-5, threshold best ≥ 0.45,
   first local peak ≥ 0.85·best with `r[l] ≥ r[l±1]`, parabolic refinement
   clamped to ±1, median-of-5-voices smoothing that drops isolated blips).
3. **Contour fill/smooth** (`fillGaps`): unvoiced runs of ≤ 3 frames are bridged
   by log-domain interpolation; the contour is then re-smoothed (.25/.5/.25 in
   log domain).
4. **Target retrieval database** (`buildDB`, only when a target is present and
   (blend > 0 or mode ≠ "env")): pitch-track target → `nf = floor((len−N)/512)`
   frames at 512-sample hop → rms gate (SIL = 0.004) → per-frame `feat` DCT row,
   `le` envelope, CEP lifter row, voicing index, mark assignments (`mk`),
   target-vs-frame nearest-mark map (`fm`), global mean/variance of `CEP[1:]`.
   If `n < 30`, the database is discarded ("Target too short or quiet").
5. **Mode resolution** — `grain` / `voc` only apply when a database exists.
   Otherwise the pipeline falls back to the **envelope** path (pitch + formant
   only), implemented as the direct envelope-transfer loop below.
6. **Pitch adjustment** — `ratio = 2^(semitones/12)`; if auto-pitch is on and a
   database exists, `ratio *= target_median_f0 / source_median_f0`; clamped to
   [0.4, 3.0].
7. **Source pitch shifting** — only for the envelope path: `TD-PSOLA`
   (`makeMarks` + `psola`) when `|ratio − 1| > 0.005`; grain/vocoders instead
   adjust grain duration/harmonics inline (no TD-PSOLA).
8. **Source analysis** (`sv`, `sa`, `rs`, `sf`): per-frame windowed analysis
   at hop `H = 256` around frame centre, `feat()` DCT features (12-D), `cmn()`
   mean subtraction, `.25/.5/.25` time smoothing of features, `sa` rms gate,
   `sv` voicing at frame centre (same index for grain/vocoder), `rs` rms.
9. **Nearest-neighbour retrieval** (`knn`, only with a database): per frame,
   candidate set = db voicing-group (or the opposite group when the same group
   has ≤ 20 entries), K = 6 closest frames by squared Euclidean distance on
   CMN'd features, kept as a bounded buffer.
10. **Synthesis** (mode-dependent):
    - **Grain** (`grainSynth`): greedy nearest-grain loop — pick the target mark
      with the best composite score `(dist(f)+0.05)·pen(T)·(v≠grain_v ? 2 : 1)`;
      keep target continuity when the *next* mark fits (`.7` weight); phase
      alignment by max-correlation search over ±round(T/3); gain clamped
      [0.25, 4] with temporal EMA (α = 0.3); raised-cosine overlap-add
      (`max(ws, .35)` normalization).
    - **Vocoder** (`vocode`): EM GMM (≤ 8 components, 15 iterations, deterministic
      init by quantiles, ridge regression per component, ≥ 30 frames gate) →
      per-frame conversion (posterior-weighted prediction) → time smoothing
      (.1/.2/.4/.2/.1) → global-variance restore (clamped [1, 2.2]) →
      harmonics at source pitch (NH = 220, `2π·0.6` phase fill, sawtooth
      pseudo-random PH phases) + envelope-shaped noise, harmonic synthesis by
      sample-wise phase accumulation.
    - **Envelope** (default when no database / mode = "env"): per-frame envelope
      transfer — blend the source envelope with the K nearest target envelopes
      (log-domain mean-energy centre match `off`, `.6/.4` continuity, global
      formant warp, ±3 log-clamp, unvoiced × 0.6) → scale the raw spectrum by
      `exp(d)` (with ±3 clamp), hann overlap-add with 1.5 normalization.
11. **Final normalization** — `pk = max |res|`; `gn = pk > 0 ? 0.92 / pk : 1`;
    all samples scaled. (NaN → 0.)
12. **Output** — 16-bit PCM WAV @ 22050 Hz, clamped to ±1, scaled by 32767,
    integer-truncated.

## 3. Component → module mapping

| Original (JS) | HMVC port module |
|---|---|
| `decode()` / `wav()` | `hmvc.conversion.wavio` (`encode_wav`, `write_wav`); audio loading via `hmvc.features.audio` |
| constants | `hmvc.conversion.constants` |
| `fft()` | `hmvc.conversion.fft` |
| `trackPitch()`, `median()`, `fillGaps()` | `hmvc.conversion.pitch` |
| `makeMarks()`, `psola()` | `hmvc.conversion.psola` |
| `analyze()`, `feat()`, `cmn()`, `SIL` | `hmvc.conversion.envelope` |
| `buildDB()`, `knn()` | `hmvc.conversion.retrieve` |
| `grainSynth()` | `hmvc.conversion.grain` |
| `inv()`, `vocode()` | `hmvc.conversion.vocode` |
| `convert()` + presets/ranges | `hmvc.conversion.config`, `hmvc.conversion.pipeline`, `hmvc.conversion.__main__` |

The original has **no HMM**. Retrieval is classical nearest-neighbour +
GMM-EM + harmonic/noise synthesis. `hmvc.engine` (the extracted HMS engine)
is **not** used by this port.

## 4. Exact parameters in use

| Parameter | Value | Notes |
|---|---|---|
| SR | 22050 | all audio is resampled to this |
| N | 1024 | FFT/window |
| H | 256 | analysis hop |
| NB | 513 | spectral bins |
| HOP | 320 | pitch-tracking hop |
| D | 12 | DCT features |
| NM | 34 | mel bands |
| LIF | 36 | lifter order |
| SIL | 0.004 | RMS gate |
| W (autocorr) | 800 | pitch-tracking lag window |
| thresholds | 0.45 / 0.85 | voicing / local-peak |
| PSOLA | U = 110, halves 0.5 / 0.3, clamp ±1 | TD-PSOLA |
| auto-correlation norm | `0.45` gate; parabolic interp clamp | ±1 semitone |
| auto-pitch clamp | [0.4, 3.0] | after `2^(semitone/12)` and auto-match |
| envelope blend default | 80 % | K = 6 default |
| voiced time smoothing | 3 frames (±1), ≥ 3 voiced votes | drops blips |
| buildDB hop | 512 | frames in target index |
| buildDB min frames | ≥ 30 | else retrieval disabled |
| grain U (unvoiced) | 110 px | grain duration cap [0.25, 4] |
| grain gain EMA | 0.3 | gain time constant |
| GMM | M = max(2, min(8, n/60)), 15 EM iterations, ridge .15 | ℓ2 ridge |
| vocoder | NH = 220, N = 1024, `cst = 2/(N·N·0.375)`, `1.2247` | |
| normalization | peak → 0.92, NaN → 0 | |
| output | 22050 Hz, 16-bit, clamp ±1, × 32767 | |

## 5. Known discrepancies (documented, not fixed)

These are deliberate — behavior is preserved from the HTML, not corrected.

### 5.1 Harness vs port (test fixture tooling, not shipped code)
- Harness LCG (`tools/generate_conversion_reference.js`): the original used the
  *default* seed 42 and a single `Math.random()`-style generator throughout.
  The harness was intentionally re-pinned to match the port exactly:
  `LcgRandom(1234)` with `MULT = 0x5851f42d4c957f2d`,
  `INC = 0x14057b7ef767814f`, `float(state>>11)/2^53`. All reference diffs are
  therefore 0.0 (max abs = 0.0 over samples, ratio, f0, vo, nfr, db).
- `feat()` mel-band means are float32 native (not float64 means cast down);
  reproducing this in the harness removed the remaining voc 2.83e-07 diff.
- FFT store-back: inverse FFT accumulates into float32 buffers per-stage
  (the same pattern the original uses).

### 5.2 Port behavior changes to replicate the HTML faithfully
- Database memoization: the pipeline caches a built target DB keyed by
  `(target_len, fmin, fmax, t[mid])`. On a cache hit, retrieval is skipped
  silently (no progress events). This matches the original's browser-session
  behavior, where the database built once per page visit is reused across
  re-runs — including the quirk that an under-threshold database result
  (n < 30) is cached with `db = None` and the "Target too short" message is
  re-sent on every call.
- JavaScript `Math` semantics are replicated in `hmvc.conversion.jsmath`
  (`js_log`, `js_exp`, `js_round`) rather than re-raising on degenerate
  values: `log(0) = -inf`, `log(negative) = NaN`, `exp` overflows to
  `Infinity`, `round` = floor(x + 0.5). The original relies on all of this;
  the port returns NaN/-inf exactly like the browser.
- `grain.py` re-associates `(w * gn)` → `(t[a_abs] * w) * gn` (pure
  re-association; bit-for-bit identical — the trunk sum is unchanged).

### 5.3 Structural differences (no behavioral change)
- The CLI, WAV I/O and `conversion.__main__` are new. The original is a
  browser app with no CLI; equivalent behavior is exposed as
  `python -m hmvc conversion <source> [--target target.wav] [--mode voc]
  [--pitch semitones] [--form 0.6..1.5] [--blend 0..1] [--neighbours 1..16]
  [--seed N]`.
- The original audio loading is browser `decodeAudioData` +
  `OfflineAudioContext` resample; the port decodes WAV via
  `hmvc.features.audio` (resample to 22050 Hz). Slicing/clamping/peak
  normalization are identical.
- TD-PSOLA `js_round` semantics use `floor(x + 0.5)` (JS round), and the
  pipeline's final peak-normalization clamps NaN → 0, exactly as the original.

### 5.4 Out of scope
- Env/grain/vocoder output is not quality-improved; the buzzy envelope,
  smooth-vocoder and grain-overlap-add artifacts are preserved by design.
