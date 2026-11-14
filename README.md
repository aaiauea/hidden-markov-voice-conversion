# hidden-markov-voice-conversion

HMVC's long-term goal is lightweight, CPU-friendly voice conversion. **Phase 0
extracts the existing HMS synthesis engine as a self-contained foundation; it
does not implement HMVC-specific conversion, LPC features, or synthesis changes.**

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

The port is reproducible/reviewable with the optional maintainer tool
`tools/extract_hms_engine.py --source /separate/pinned/HMS --check`.
`tools/generate_hms_reference.py` regenerates the synthetic baseline from original
HMS only when explicitly requested. Neither tool is used by the installed engine.

**This is an engine bootstrap, not a finished HMVC voice-conversion system.**
