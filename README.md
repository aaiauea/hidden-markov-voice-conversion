# hidden-markov-voice-conversion

HMVC's long-term goal is lightweight, CPU-friendly voice conversion.

- **Phase 0** extracted the existing HMS synthesis engine as a self-contained
  foundation (`hmvc.engine`).
- **Phase 1** adds classical, NumPy-only acoustic feature extraction for
  audio-to-audio conversion (`hmvc.features`).

The repository now also contains a strict 1:1 port of the existing classical voice converter; see [Voice conversion (Phase 1)](#voice-conversion-phase-1).

## Engine foundation

The selected engine from [Hidden Markov Singer](https://github.com/aaiauea/hidden-markov-singer)
(revision `ba1eac18146f017f8cc3cae6acdb7963b8f4149e`) now lives in
**`hmvc.engine`**. HMS is **not** a runtime dependency: an HMS checkout or installed
`hms` package is not needed to install, train, synthesize, or run ordinary tests.

The extraction preserves HMM/GMM training and transitions, duration/pitch models,
MLPG, GV/GM, context/pitch conditioning, model save/load, builtin/MLSA vocoders,
and optional source-aware synthesis. Original algorithms, public class names
(including `HMSModel`), model-format identifiers and legacy backend settings are
retained. The current engine remains score-driven, using HMS's mel-cepstral /
WORLD-style parameters—not an LPC/audio-to-audio conversion engine.

See the [extraction and verification report](docs/hms-engine-extraction.md) for
the exact dependency closure, transferred files, adaptations, actual test results,
and limitations. Attribution: [third-party notices](THIRD_PARTY_NOTICES.md),
[original MIT license](src/hmvc/engine/LICENSE.hms), and
[per-file provenance](src/hmvc/engine/UPSTREAM.json).

## Feature extraction (Phase 1)

`hmvc.features` analyses WAV files or arrays at any common sample rate into
separate, frame-synchronous streams:

- a reconstructible, compact **mel-cepstral** representation of a CheapTrick
  spectral envelope (the primary representation). Finite-order MCEP is an
  approximation of the original envelope, not a lossless encoding: at the
  16 kHz default (order 24) the envelope rebuilt with `mc2sp` differs from the
  original by about 2.2 dB RMS log-spectral distance on speech. Use
  `keep_envelope=True` to keep the full envelope as well;
- **F0**, **voicing** and **periodicity**;
- frame **energy**;
- **band aperiodicity**;
- **spectral descriptors** (centroid, bandwidth, rolloff, flatness, tilt,
  ZCR, flux).

Corpus normalisation is mergeable. It uses no neural model, transcript,
WORLD/SPTK library or new dependency.

```python
from hmvc.features import AnalysisConfig, FeatureExtractor

feats = FeatureExtractor(AnalysisConfig(sample_rate=16000)).extract_file("utt.wav")
feats.mcep, feats.f0, feats.voiced, feats.energy_db, feats.band_aperiodicity
```

CLI: `python -m hmvc.features extract utt.wav utt.npz` and
`python -m hmvc.features info utt.npz`. See the
[feature-extraction report](docs/feature-extraction.md) for the representation
study, the rejected alternatives, agreement with WORLD/SPTK/librosa,
configuration, benchmarks and limitations.

## Voice conversion (Phase 1)

HMVC now includes a **classical, non-neural voice-conversion pipeline** in
`hmvc.conversion`. Phase 1 is a strict 1:1 Python/NumPy port of the existing
retrieval-based converter, preserving its DSP operations, ordering, parameters,
and quirks rather than redesigning or quality-tuning them.

The converter accepts source audio and an optional target-voice recording. It
tracks pitch, optionally builds a target-frame retrieval database, analyzes
spectral features, and synthesizes the converted signal. The three modes are:

- **`voc`** — retrieval-conditioned GMM vocoder (default when a usable target
  database is available);
- **`grain`** — target-grain retrieval and overlap-add synthesis;
- **`env`** — spectral-envelope transfer, also used as the fallback when no
  usable target database is available.

Pitch shifting, automatic target-pitch matching, formant adjustment, blend,
neighbour count, voice-range presets, and named presets are configurable.
Input WAV audio is converted to mono at **22050 Hz** internally; output is
16-bit PCM WAV. A target recording is optional, but retrieval-based modes need
a sufficiently long and audible target recording (at least 30 indexed frames).

### CLI

```sh
python -m hmvc.conversion \
  --src source.wav \
  --tgt target.wav \
  --out converted.wav \
  --mode voc \
  --pitch 0 \
  --form 1.0 \
  --blend 0.8 \
  --neighbours 6 \
  --range standard \
  --progress
```

Omit `--tgt` to run without a target recording. Available modes are
`voc`, `grain`, and `env`; voice ranges are `low`, `standard`, and
`high`. Use `--preset` to select a built-in preset, or `--no-auto` to
disable automatic target-pitch matching. Run
`python -m hmvc.conversion --help` for the complete argument list.

### Python API

```python
from hmvc.conversion import ConversionConfig, convert, load_audio, write_wav

source = load_audio("source.wav")
target = load_audio("target.wav")  # optional; use None to omit it

config = ConversionConfig(
    mode="voc",
    pitch_semitones=0.0,
    formant_factor=1.0,
    envelope_blend=0.8,
    neighbours=6,
)

result = convert(source, target, config)
write_wav("converted.wav", result.samples)
```

The public API also exposes `ConversionResult`, `PRESETS`,
`VOICE_RANGES`, and `preset_params`. See
[the conversion pipeline and fidelity notes](docs/conversion.md) for the
module mapping, parameters, execution order, documented discrepancies, and
known behavior.

**Important architecture distinction:** the conversion pipeline is not an HMM
voice-conversion model. It uses classical nearest-neighbour retrieval, GMM-EM,
and DSP synthesis. The separately extracted `hmvc.engine` is the HMS
score-driven HMM/GMM singing-synthesis engine; it is not currently used by
`hmvc.conversion`. The feature extractor in `hmvc.features` is also a
separate component, not a learned conversion mapper.

### Current limitations

- This is a faithful baseline port, **not a quality-improved or production-ready
  voice-conversion system**. The original envelope, grain, and vocoder artifacts
  are intentionally preserved.
- The parity suite checks deterministic synthetic cases against frozen
  reference outputs. That verifies implementation fidelity for those cases;
  it does not establish perceptual quality, generalization across speakers, or
  performance on a broad real-world evaluation corpus.
- There is no neural conversion model, speaker-embedding system, or trained
  multilingual model.
- The extracted HMS engine, feature-extraction component, and conversion
  pipeline are currently separate subsystems.

## Install

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Only **NumPy and PyYAML** are runtime requirements. There is no mandatory
PyTorch, librosa, SciPy, soundfile, pyworld, or native WORLD dependency. The pure
NumPy `builtin` and `mlsa` backends run without a compiler.

## Use the existing synthesis API

```python
from hmvc.engine.core.labels import load
from hmvc.engine.core.model import HMSModel
from hmvc.engine.core.synthesizer import SynthesisConfig, Synthesizer
from hmvc.engine.data.wavio import write_wav

model = HMSModel.load("path/to/model")
score = load("path/to/score.tsv")
result = Synthesizer(
    model, SynthesisConfig(vocoder="builtin", seed=0)
).synthesize(score)
write_wav("output.wav", result.audio, model.spec.fs)  # PCM16
```

Training uses `hmvc.engine.core.trainer.Trainer` / `TrainingConfig` with labeled
PCM WAV input and a `PhonemeSet`. No CLI, evaluation utility, recorded corpus, or
pretrained voice is included in the installed package. Small **synthetic,
test-only** model/oracle files verify compatibility; they are not voicebanks.

Select `builtin` or `mlsa` explicitly for lightweight operation. For compatibility
with HMS, `auto` retains optional WORLD selection/auto-build behavior; set
`HMS_NO_AUTO_BUILD=1` to prohibit native builds/downloads. Optional backend code
and its package-local build support are retained, but real WORLD execution has
not been verified. WORLD/pyworld would need their own installation/license review.

## Tests

```sh
.venv/bin/python -m pip install -e '.[test]'
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -m pytest
```

The suite blocks imports of `hms` and heavy optional libraries, disables native
auto-build **in tests**, ports the applicable HMS engine regressions, and compares
actual waveforms/parameters against independently generated original-HMS
references. No network access or original HMS repository is needed for this run.

`tests/features/` checks the feature extractor on synthetic signals with known
F0 and envelopes, on unusual input, and against stored outputs of WORLD, SPTK
and librosa (`tests/data/feature-reference/`), without importing those
libraries. `tools/generate_feature_reference.py` and `tools/evaluate_features.py`
need them in a separate environment. `tools/benchmark_features.py` needs only
NumPy.

The port is reproducible/reviewable with the optional maintainer tool
`tools/extract_hms_engine.py --source /separate/pinned/HMS --check`.
`tools/generate_hms_reference.py` regenerates the synthetic baseline from original
HMS only when explicitly requested. Neither tool is used by the installed engine.

**HMVC currently combines an extracted singing-synthesis engine, classical acoustic feature extraction, and a faithful baseline voice-conversion port. These are distinct components; the project is not yet a unified or quality-optimized voice-conversion system.**
