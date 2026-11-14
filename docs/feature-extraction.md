# HMVC Phase 1: classical acoustic feature extraction

**Status:** implemented in `hmvc.features` (NumPy only), tested, benchmarked.
This phase covers **feature extraction and representation only**. It contains
no conversion mapping, GMM, training pipeline, speaker embedding, neural
component or PPG.

Evidence files (regenerable):

- [`feature-extraction/evaluation.json`](feature-extraction/evaluation.json):
  representation study, accuracy vs WORLD / SPTK / librosa, and real speech
  (`tools/evaluate_features.py`, which needs pyworld, pysptk and librosa in a
  separate environment).
- [`feature-extraction/benchmark.json`](feature-extraction/benchmark.json):
  timings and memory (`tools/benchmark_features.py`, NumPy only).
- `tests/data/feature-reference/`: stored reference-library outputs that the
  test suite compares against without importing those libraries.

## 1. Summary of decisions

| Concern | Decision | Main reason (evidence in §4–§6) |
|---|---|---|
| Primary spectral representation | **All-pass mel-cepstrum (MCEP)** of a **CheapTrick** envelope; order 24 / 34 / 39 for ≤16 / ≤24 / >24 kHz; SPTK alpha (0.41 at 16 kHz) | Reconstructible (`mc2sp`) and compact, as a finite-order approximation of the envelope rather than a lossless encoding (§4.3). Also decorrelated, F0-independent and perceptually weighted. It is the standard input of classical VC mapping (GMM/DNN/frame-wise MCD). |
| Envelope estimator | CheapTrick (independent implementation, matches WORLD to <1e-5 dB) | Most accurate and most F0-invariant of the five estimators tested. |
| Lossless option | `keep_envelope=True` stores the full linear-frequency power envelope | MCEP is a smoothing (2.2 dB round-trip LSD at order 24 on speech). The envelope is the loss-free source. |
| F0 | YIN + Viterbi continuity tracking + short-window refinement + hysteresis voicing | 0.2–2.1 cents median on synthetic vibrato, no gross errors. Agrees with WORLD on speech as well as WORLD's two estimators agree with each other. |
| Voicing | Separate `voiced` (bool) and continuous `periodicity` | VC needs both a decision and a soft strength. |
| Aperiodicity | 5 mel-band noise-power fractions (Boersma harmonicity) | Calibrated: estimates are within ±0.04 of the true noise fraction. |
| Energy / descriptors | Separate streams: RMS, dB, speech gate, centroid, bandwidth, rolloff, flatness, tilt, ZCR, flux | Match librosa to ~1e-9 relative. Never forced into one vector. |
| HMS engine | Not modified. Reused only through an optional bridge (`interop.to_engine_sequence`) | Its analysis representation is lossy for VC (see §3). |
| Dependencies | numpy + PyYAML only (unchanged) | Asserted by `tests/test_engine_bootstrap.py`. |

## 2. Quick start

```python
from hmvc.features import AnalysisConfig, FeatureExtractor, FeatureNormalizer

extractor = FeatureExtractor(AnalysisConfig(sample_rate=16000))  # resample everything to 16 kHz
feats = extractor.extract_file("utt.wav")       # or extractor.extract(samples, sample_rate)

feats.mcep                  # (T, 25) mel-cepstrum, primary spectral representation
feats.f0, feats.voiced      # Hz (0 when unvoiced), bool
feats.periodicity           # (T,) soft voicing strength in [0, 1]
feats.continuous_log_f0     # ln F0, interpolated through unvoiced frames
feats.energy_db, feats.speech
feats.band_aperiodicity     # (T, 5) noise fraction per mel band
feats.spectral["centroid"]  # also bandwidth, rolloff, flatness, tilt, zcr, flux
feats.envelope()            # (T, 513) power envelope (kept, or rebuilt from MCEP)
feats.stack(["mcep", "continuous_log_f0", "band_aperiodicity"])  # opt-in concatenation
feats.save("utt.npz"); feats.validate()

norm = FeatureNormalizer().fit(corpus_features)   # per-stream z-score, JSON-serialisable
z = norm.transform(feats)["mcep"]
```

CLI: `python -m hmvc.features extract in.wav out.npz [--sample-rate 16000]
[--hop-ms 5] [--mcep-order 24] [--keep-envelope] [--config cfg.json]`, and
`python -m hmvc.features info out.npz`.

## 3. Existing infrastructure: reuse vs independent implementation

The extracted HMS engine (`src/hmvc/engine`, hash-locked, unchanged) was
probed function by function. The probe used real calls, and the numbers below
are measured.

| HMS component | Finding | Decision |
|---|---|---|
| `core/features.power_to_mcep` / `mcep_to_power` | Mel-filterbank + DCT "MCEP", not an all-pass cepstrum, and its warping factor differs from SPTK (0.459 vs 0.41 at 16 kHz). Like any finite-order cepstrum it only approximates the envelope, but its round-trip error is larger than all-pass MCEP's at every dimension tested (2.61 vs 2.18 dB at D=25 on speech). | **Not reused** for the primary representation. Its output is still reachable via the engine. |
| Builtin vocoder `analyze()` envelope (fixed lifter, order 40) | 1.7–19.5 dB envelope error vs truth and strongly F0-dependent (shape std 7–8 dB across F0) | **Rejected** as an envelope estimator. |
| `core/dsp.autocorrelation_f0` | Integer-lag only (110 Hz → 111.11, 440 → 444.44). Gross-error rate 9–14% on vibrato vowels, 24 false-voiced frames on the mixed signal. | **Replaced** by YIN with tracking and refinement. |
| `data/wavio` | Misreads IEEE-float WAV | **Own reader** (`audio.read_wav`): PCM 8/16/24/32, float32/64, EXTENSIBLE, truncated files. |
| Builtin vocoder frame grid | 205 frames per second at hop 80 vs `1 + n // hop` = 201 (WORLD) | HMVC uses the WORLD grid. The bridge passes the frame period explicitly. |
| `AcousticFrameSequence` (f0, sp, ap) and the builtin/MLSA vocoders | Useful for analysis-synthesis loop-back | **Reused** through `hmvc.features.interop` only (lazy import). |

HMS's synthesis representation was designed for HMM synthesis from scores,
where smoothing is acceptable. For audio-to-audio VC the analysis has to keep
speaker detail, so a separate representation was built. It is reconstructible,
with an approximation error that is measured (§4.3) and controllable through the
order.

## 4. Representation study

All numbers come from `docs/feature-extraction/evaluation.json`. The "speech"
material is CMU ARCTIC `arctic_a0007` (16 kHz, 4 s, bundled with pysptk). It
was used only for evaluation and is not stored in the repository.

### 4.1 Spectral envelope estimators

These are steady additive-synthesis vowels with an analytically known
envelope, given the true F0. The metric is gain-removed RMS log-spectral
distance (dB, 80 Hz–7.2 kHz).

| Estimator | /a/ 100 Hz | 150 | 220 | 300 | 400 | F0-invariance std (dB) /a/ · /i/ |
|---|---|---|---|---|---|---|
| **CheapTrick (HMVC)** | **0.79** | **0.89** | **1.42** | **1.87** | **1.99** | **0.72 · 0.95** |
| CheapTrick (WORLD, reference) | 0.79 | 0.86 | 1.40 | 1.85 | 1.99 | 0.72 · 0.95 |
| True envelope (iterative cepstral, HMVC) | 1.48 | 2.40 | 4.49 | 7.23 | 9.83 | 2.38 · 2.66 |
| F0-adaptive lifter (fs/2F0, HMVC) | 3.95 | 5.65 | 6.38 | 10.43 | 13.61 | 4.09 · 3.40 |
| HMS builtin lifter (order 40) | 1.71 | 3.24 | 5.57 | 7.93 | 18.56 | 6.99 · 7.81 |

F0 invariance matters directly for VC. An envelope that changes with pitch
leaks F0 into the "timbre" features, and a mapping would then learn pitch
artefacts. CheapTrick is best on both counts. `true_envelope` and `lifter`
remain available (`envelope_method=`) for comparison, but they are not the
default.

### 4.2 Candidate representations of the envelope

| Candidate | Reconstruction² | Dim. | Decorrelated¹ | F0-independent | Verdict |
|---|---|---|---|---|---|
| Log-magnitude spectrum (513 bins) | exact (magnitude only) | 513 | no (0.695) | no (harmonics) | **Rejected** as a primary feature: high-dimensional, correlated, and it carries harmonic structure, i.e. F0. Available as `framing.log_magnitude`. |
| Full spectral envelope (`keep_envelope`) | exact (it *is* the source) | 513 | no | yes | **Kept as an optional stream.** It is the loss-free reference, but too large and correlated for a mapping. |
| Real cepstrum of the STFT | exact at full length N | N | partly | **no** | **Rejected**: low quefrencies mix source and filter unless liftered at a pitch-dependent order (= the lifter estimator above). |
| Liftered cepstrum (linear frequency, α=0) | approximate (truncated at D) | D | yes | yes, from the envelope | **Configurable** (`mcep_alpha=0`). Best for uniform-frequency fidelity (see 4.3). |
| SPTK UELS `mcep` of windowed frames | approximate (truncated at D) | D | yes | **no** | **Rejected**: the MCD between the same vowel at 100 Hz and 300 Hz is 18.5–22.3 dB (envelope-based: 2.7–3.0 dB). The MCD to the true envelope grows to 19–23 dB at 300 Hz, because it fits harmonics. |
| HMS mel-filterbank DCT | approximate (filterbank + truncation) | D | yes (0.134) | yes, from the envelope | **Rejected**: worse round trip than MCEP at every D, and not SPTK-compatible. |
| **All-pass MCEP of the CheapTrick envelope** | approximate (truncated at D; rebuilt with `mc2sp`) | D | **yes (0.124)** | **yes** | **Selected** |

¹ Mean absolute off-diagonal correlation over speech frames (lower = more
decorrelated), excluding c0.

² Whether the original can be rebuilt from the stored values. "Approximate"
means a finite-order (smoothed) approximation of the original envelope; its
error is quantified in §4.3.

### 4.3 How much is lost, and where

Round-trip error envelope → D coefficients → envelope, measured on the
CheapTrick envelopes of all 801 speech frames. The metric is RMS dB over all
bins.

| D | MCEP α=0.41 | linear cepstrum α=0 | HMS filterbank DCT |
|---|---|---|---|
| 25 (default at 16 kHz) | 2.18 | 1.43 | 2.61 |
| 35 | 1.63 | 1.05 | 2.00 |
| 40 | 1.42 | 0.92 | 1.77 |
| 60 | 0.91 | 0.49 | 1.19 |

Measured uniformly over linear frequency, the linear cepstrum is *more*
faithful than MCEP at equal size. The per-band breakdown shows why MCEP is
still chosen.

| RMS error (dB), D = 25 | 0–1 kHz | 1–2 kHz | 2–4 kHz | 4–8 kHz |
|---|---|---|---|---|
| MCEP α = 0.55 | **0.56** | **1.04** | 2.28 | 3.55 |
| **MCEP α = 0.41 (default)** | **0.90** | **1.11** | 1.94 | 3.09 |
| linear cepstrum α = 0 | 2.16 | 1.85 | 1.94 | 1.98 |

MCEP spends its coefficients where speaker identity and intelligibility cues
are densest: F1/F2 and the low-frequency formant structure. It is 2.4× more
accurate below 1 kHz and pays for that above 4 kHz. This is the established
trade-off behind the mel-cepstral distortion (MCD) metric and classical
statistical VC. The losses are explicit and controllable:

- Raise `mcep_order` (35 → 1.63 dB, 40 → 1.42 dB at 16 kHz) when
  high-frequency detail matters more than dimensionality.
- Set `mcep_alpha=0` for uniform resolution.
- Set `keep_envelope=True` for zero representation loss. `AcousticFeatures.envelope()`
  then returns the original envelope rather than the reconstruction.

### 4.4 Default orders

Order 24 (25 values) at ≤16 kHz, 34 at ≤24 kHz and 39 above match common VC
practice. They keep the 0–2 kHz error at about 1 dB at 16 kHz. They are
defaults, not limits.

## 5. Auxiliary features

All streams share one frame grid. Frame *t* is centred on sample
*t*·hop, and an *n*-sample signal has 1 + ⌊n/hop⌋ frames (the WORLD
convention). Every stream is a separate array.

| Stream | Shape | Definition |
|---|---|---|
| `f0` | (T,) | Hz, exactly 0 where unvoiced |
| `voiced` | (T,) bool | Voicing decision |
| `periodicity` | (T,) | 1 − CMND of the tracked period, in [0, 1] |
| `log_f0` (property) | (T,) | ln F0, NaN where unvoiced |
| `continuous_log_f0` | (T,) | ln F0 linearly interpolated through unvoiced frames, held at the edges, √(floor·ceil) if nothing is voiced |
| `energy_rms`, `energy_db` | (T,) | Frame RMS (full scale = 1); 20 log10, floored at −100 dB |
| `speech` | (T,) bool | Level above max(−80 dBFS, loudest frame − 50 dB) |
| `mcep` | (T, M+1) | Primary representation (c0 = log gain) |
| `band_aperiodicity` | (T, B) | Noise power fraction per band; 1 for unvoiced frames |
| `spectral[...]` | (T,) each | centroid, bandwidth, rolloff (85%), flatness, tilt (dB/octave, 100 Hz–5 kHz), zcr, flux |
| `spectral_envelope` | (T, N/2+1) | Optional, linear power |

### 5.1 F0 and voicing

The YIN difference function is computed for all frames with FFT
cross-correlation. Then:

1. **Candidates:** the 8 best parabolically interpolated CMND minima per frame.
2. **Viterbi tracking:** cost = CMND + 0.02·log2 T + 2·|Δ ln T|. It removes
   YIN's sub-octave picks and formant-period excursions.
3. **Hysteresis voicing:** seeds at aperiodicity < 0.25. Regions grow into
   neighbours with aperiodicity < 0.6 and a continuous candidate (< 20% jump).
   All voiced frames must pass the silence gates.
4. **Refinement:** a symmetric two-period difference function re-estimates
   the period. YIN's long window smears vibrato and lags by half a period.
5. **Cleanup:** a 3-frame median within voiced runs, and runs shorter than
   15 ms are removed.

Synthetic signals with known F0 (`f0_synthetic` section; cents vs truth):

| Signal | HMVC median / p95 | GPE | WORLD Harvest | WORLD DIO+StoneMask | HMS autocorrelation |
|---|---|---|---|---|---|
| /a/ 115→150 Hz, vibrato | 0.72 / 4.1 | 0 | 0.19 / 5.8 | 0.23 / 0.97 (GPE 1%) | 4.97 / 1917 (GPE 13.9%) |
| /i/ 300→350 Hz, vibrato | 0.23 / 0.72 | 0 | 0.14 / 2.67 | 0.62 / 2.05 (GPE 1%) | 25.2 / 43.3 |
| mixed (silence, fricative, 2 vowels) | 2.13 / 6.92 | 0 | 0.31 / 24.52 | 0.54 / 16.99 (GPE 1.6%) | 13.2 / 1912 (GPE 9.4%) |
| /a/ 22.05 kHz | 0.98 / 3.22 | 0 | – | – | – |

False-voiced frames on the mixed signal: HMVC 2, Harvest 11, DIO 2, HMS 24.
Both HMVC frames are single frames adjacent to a true voicing boundary, where
the window already overlaps voicing. A dense sweep of pure tones from 62 to
780 Hz at 8, 16 and 44.1 kHz has no octave errors (a regression test). White
noise, fricatives and silence are never voiced.

Real speech (arctic_a0007, 801 frames). WORLD has no ground truth, so
agreement is shown next to the agreement between WORLD's own two estimators.

| Pair | Voicing agreement | Median cents | GPE (>20%) |
|---|---|---|---|
| **HMVC vs Harvest** | 0.80 | 7.5 | 1.9% |
| **HMVC vs DIO** | 0.91 | 7.8 | 0.3% |
| Harvest vs DIO (reference spread) | 0.81 | 5.3 | 1.5% |
| librosa pYIN vs Harvest | 0.74 | 25.2 | 0.7% |
| librosa pYIN vs DIO | 0.85 | 24.5 | 1.5% |

Before tracking and hysteresis were added, HMVC scored 0.74 / 0.87 voicing
agreement and 3.1% / 1.6% GPE against Harvest / DIO. HMVC's voicing is
**conservative**: voiced fraction 0.46, versus DIO 0.51 and Harvest 0.66. It
misses weak voicing at onsets and offsets rather than voicing noise. For VC
this is the safer error, because `continuous_log_f0` bridges the gaps and
`periodicity` exposes the soft value. All thresholds are configurable.

### 5.2 Band aperiodicity

Harmonicity is measured per mel band from the window-corrected normalised
autocorrelation at the period (Boersma). Band edges are snapped to (h + ½)·F0,
the nulls of the 4-period Hann window. Each frame's value depends only on its
own samples and F0, so it is block-size and neighbour invariant (a tested
property).

| True noise fraction | 0.0 | 0.1 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|
| HMVC, steady 150 Hz | 0.003 | 0.091 | 0.283 | 0.483 | 0.764 |
| HMVC, 50-cent vibrato | 0.022 | 0.111 | 0.302 | 0.502 | 0.788 |
| WORLD D4C (different definition²) | 0.002 | 0.083 | 0.158 | 0.234 | 0.321 |

² D4C's band values are not a power fraction, and its default voicing gate
returns 1.0 for these frames (`threshold=0` was used). It is shown for
orientation only.

### 5.3 Energy and spectral descriptors

These match librosa (stored fixtures) to ≤1e-9 relative for centroid,
bandwidth, rolloff and flatness, and to 1e-7 relative for RMS (float32
storage). ZCR is within one crossing per frame. Two documented convention
differences are handled explicitly in the tests:

- librosa floors `|X|²` at 1e-10, while HMVC's power density is `|X|²/Σw²`.
- librosa's RMS/ZCR use `L//2` left padding for odd frame lengths.

## 6. Agreement with reference implementations

Every value below is asserted by `tests/features/test_reference.py` against
stored outputs of the actual libraries.

| HMVC function | Reference | Agreement |
|---|---|---|
| `envelope.cheaptrick` | WORLD CheapTrick (pyworld 0.3.5) | ≤3.8e-6 dB on the 16 kHz vowels. 2.3e-6 dB at 22.05 kHz using WORLD's fractional frame origins. 5.3e-6 dB on the mixed signal's bins above WORLD's random silence floor. 5.6e-6 dB max on real speech. |
| `cepstrum.sp2mc` | `pysptk.sp2mc` | ≤1.3e-7 on the fixtures (float32-stored input, `floor=0`); 5e-15 on float64 speech envelopes |
| `cepstrum.mc2sp` | `pysptk.mc2sp` | 2.1e-6 dB |
| `cepstrum.freqt` (matrix and recursive) | `pysptk.freqt` | ≤4.4e-16 |
| `cepstrum.mcep_alpha` | `pysptk.util.mcepalpha` | Exact at 8 rates |
| STFT power | `librosa.stft` | 2.7e-7 (log magnitude, float32 storage) |
| Descriptors / RMS / ZCR | librosa 0.11.0 | See §5.3 |

Intentional differences:

- **Power floor.** HMVC floors power at 1e-12 before taking logs, so digital
  silence yields c0 = ln(1e-12)/2 and c1…cM = 0. WORLD adds random
  ~1e-16 noise and SPTK logs it. The tests pass `floor=0` to compare with SPTK.
- **Frame grid.** HMVC uses an integer hop (`round(5 ms · fs)`, e.g. 110
  samples at 22.05 kHz). WORLD uses a fractional 110.25.

## 7. Audio handling and robustness

- **Input.** WAV (PCM 8/16/24/32-bit, float32/64, WAVE_FORMAT_EXTENSIBLE,
  odd-sized and unknown chunks, truncated data) or arrays (float, int, uint8,
  bool, mono or `(n, channels)`).
- **Sample rate.** Analysis runs at the native rate, or every input is
  resampled to `sample_rate` with a Kaiser-windowed sinc (polyphase for
  rational ratios). In-band error is < 2e-3 RMS and alias rejection is > 57 dB.
  Geometry adapts per rate: FFT 512…4096, MCEP order and alpha, a 5 ms hop.
- **Sanitising.** NaN and ±inf become 0 (counted in `diagnostics`), and DC is
  removed (optional). Only *argument* errors raise.
- **Tested edge cases.** Empty input, one sample, shorter than a frame,
  silence, denormals, pure DC, all-NaN, NaN bursts, inf, clipped square
  waves, an impulse, ×10⁴ amplitude, int16, stereo, a Nyquist-frequency tone
  and white noise. Every stream stays finite, and `validate()` passes.
- **Consistency.** Results are deterministic and identical across
  `block_frames` sizes. Frame counts follow 1 + ⌊n/hop⌋ for any length.
  Interior frames of a prefix equal those of the full signal for a steady
  signal (the silence gate is relative to the loudest frame, so this cannot
  hold exactly for arbitrary cuts).
- **Normalisation.** `FeatureNormalizer` computes one-pass, mergeable (Chan et
  al.) per-stream statistics. Pitch streams use voiced frames only, other
  streams use speech frames. Statistics are stored as JSON together with the
  `AnalysisConfig`, and mismatched configurations are rejected.
  `normalize_utterance` gives per-utterance CMVN.

## 8. Configuration reference (`AnalysisConfig`, frozen, JSON round-trip)

| Group | Option (default) |
|---|---|
| Audio | `sample_rate` (None = native; ≥ 4000 if set), `remove_dc` (True) |
| Framing | `hop_ms` (5.0) / `hop_length`, `frame_ms` (25.0) / `frame_length`, `fft_size` (auto: ≥ frame and ≥ 3 periods of `f0_floor`), `window` (`hann`; `hamming`, `blackman`, `rectangular`) |
| F0 | `f0_floor` (60), `f0_ceil` (800, < fs/4), `yin_threshold` (0.15), `voicing_threshold` (0.25), `voicing_extension_threshold` (0.6), `f0_max_jump` (0.2; 0 = no extension), `f0_jump_cost` (2.0; 0 = no tracking), `f0_refine_periods` (2.0; 0 = off), `f0_median_frames` (3), `min_voiced_ms` (15), `silence_threshold_db` (−50), `absolute_silence_db` (−80) |
| Envelope / cepstrum | `envelope_method` (`cheaptrick`; `true_envelope`, `lifter`), `cheaptrick_q1` (−0.15), `mcep_order` (auto 24/34/39), `mcep_alpha` (auto, SPTK), `keep_envelope` (False) |
| Aperiodicity | `n_aperiodicity_bands` (5, mel-spaced) or explicit `aperiodicity_bands` (Hz edges), `aperiodicity_periods` (4.0) |
| Descriptors / floors | `rolloff_percent` (0.85), `tilt_range_hz` (100, 5000), `energy_floor_db` (−100), `power_floor` (1e-12) |
| Implementation | `block_frames` (512): the memory/speed trade-off, with no effect on results |

## 9. Performance

From `tools/benchmark_features.py`: single thread, Python 3.11, NumPy 2.4,
2-vCPU sandbox, 10 s of synthetic speech-like audio.

| Rate | FFT | Extraction | × real time | F0 | CheapTrick + MCEP | Aperiodicity | STFT descriptors |
|---|---|---|---|---|---|---|---|
| 8 kHz | 512 | 0.25 s | 39.8× | 0.08 s | 0.11 s | 0.04 s | 0.03 s |
| 16 kHz | 1024 | 0.50 s | 20.0× | 0.14 s | 0.29 s | 0.11 s | 0.07 s |
| 22.05 kHz | 2048 | 1.06 s | 9.4× | 0.26 s | 0.53 s | 0.09 s | 0.13 s |
| 44.1 kHz | 4096 | 2.37 s | 4.2× | 0.43 s | 1.50 s | 0.18 s | 0.23 s |
| 48 kHz | 4096 | 2.16 s | 4.6× | 0.42 s | 1.35 s | 0.17 s | 0.20 s |

- Time is linear in duration at 16 kHz: 1 s → 0.042 s, 10 s → 0.48 s,
  60 s → 3.2 s.
- Peak traced memory is 18 / 49 / 61 MiB for 1 / 10 / 60 s. The working set
  is bounded by `block_frames`, and the growth is the O(T) output arrays
  (about 1 kB per frame).
- For comparison, on the 4 s speech file pyworld (compiled C++) takes 1.07 s
  for Harvest + CheapTrick + D4C, or 0.20 s with DIO. HMVC's full NumPy
  extraction takes 0.19 s.
- Resampling 44.1 → 16 kHz before analysis is the cheapest way to speed up
  high-rate corpora when the band above 8 kHz is not needed.

## 10. Tests

`tests/features/` has 257 tests and runs in about 10 s. It imports NumPy
only: `tests/conftest.py` blocks pyworld, pysptk, librosa, scipy, torch and
the other heavy libraries.

| File | Covers |
|---|---|
| `test_config.py` | Defaults per rate (8–48 kHz), validation of every option, explicit lengths, JSON/dict round trip, immutability |
| `test_audio.py` | WAV formats (PCM 8/16/24/32, float32/64, EXTENSIBLE, stereo, odd chunks, truncation, malformed files), clipping on write, resampler accuracy, aliasing and block invariance, input sanitising |
| `test_dsp.py` | Frame grid and centring, windows, power-density scaling, cepstrum round trips, lifters, `freqt` matrix = recursion, inverse warp via a high-order intermediate, sp2mc/mc2sp round trip on an envelope that is exactly representable at the analysis order, round-trip error decreasing with order, distortion measures, descriptors on sines, noise and silence, tilt, flux continuity |
| `test_f0.py` | Pure sines (65–750 Hz), missing fundamental, 7 sample rates, vibrato reference signals, the refinement gain, silence, noise and fricatives unvoiced, quiet voicing, relative gate, tracking/hysteresis unit tests, block invariance, degenerate lengths, the dense tone sweep |
| `test_envelope_aperiodicity.py` | CheapTrick vs the analytic envelope at 4 F0s × 2 vowels, F0 invariance, estimator ordering, block invariance, silence, true-envelope upper bound, aperiodicity calibration (10 cases), unvoiced = 1, neighbour/block invariance |
| `test_extractor.py` | Stream layout, stacking, envelope rebuild, 8 rates, 14 frame/analysis configurations, resampling, 16 unusual inputs, determinism, block invariance, variable lengths, prefix consistency, save/load/validate, CLI, engine bridge + builtin/MLSA vocoder loop-back |
| `test_normalization.py` | Merge = batch, pooled statistics, order independence, z-score round trip, NaN preservation, save/load, configuration guard, empty and silent utterances |
| `test_reference.py` | Fixture integrity and recipe hash, exact signal reproduction, and agreement with WORLD/SPTK/librosa (§6), F0 vs librosa YIN and Harvest |

## 11. Limitations and notes for later phases

- **Voicing is conservative** on real speech (see §5.1). If a later mapping
  needs more voiced frames, lower `voicing_threshold` /
  `voicing_extension_threshold` or use `periodicity` directly.
- **MCEP at the default order smooths high frequencies** (about 3 dB RMS
  above 4 kHz at 16 kHz). Use a higher order, `mcep_alpha=0` or
  `keep_envelope` as needed (§4.3).
- **Evaluation used one real utterance** (a single male speaker) plus
  controlled synthetic signals. Broader corpora (female and child voices,
  creaky voice, noise) should be checked before tuning thresholds further.
- **Utterance-level processing.** There is no streaming API. Memory is
  bounded per block, but the silence gate and continuity tracking look at
  the whole utterance.
- **Cross-rate corpora:** set `sample_rate` so that every utterance shares
  one geometry. `FeatureNormalizer` refuses to mix configurations.

## 12. Reproducing

```sh
# tests (NumPy only)
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest

# benchmark (NumPy only)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/benchmark_features.py --write

# reference fixtures and evaluation (separate env: pip install -e '.[reference]')
python -B tools/generate_feature_reference.py --write
PYTHONPATH=src python -B tools/evaluate_features.py --write
```
