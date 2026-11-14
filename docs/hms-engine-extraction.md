# HMVC Phase 0 — HMS engine extraction and bootstrap

**Date:** 2026-11-14 UTC  
**Source:** [aaiauea/hidden-markov-singer](https://github.com/aaiauea/hidden-markov-singer), **`ba1eac18146f017f8cc3cae6acdb7963b8f4149e`**, engine version 0.1.0  
**Destination:** `src/hmvc/engine`, independently packaged as part of HMVC 0.0.0.

This report **supersedes the earlier compatibility-only investigation**. HMS is
now a source foundation, **not a runtime repository/package dependency**. The
original HMS checkout was not modified.

## Result

The selected engine has been transferred and bootstrapped successfully for the
tested **pure-NumPy training/synthesis paths**, including source-aware synthesis.
This conclusion is supported by execution and waveform/model regressions, not
just imports:

| Level | Evidence |
| --- | --- |
| **Imported** | All **35 engine Python modules** import in fresh isolated interpreters, from unrelated working directories, with `hms` and heavyweight libraries blocked. |
| **Executed** | Local training, state/GMM/HMM fitting, MLPG, duration/pitch generation, GV/GM, source prediction, save/load and audio rendering pass applicable upstream and new regressions. |
| **Integrated** | A standalone HMVC wheel, installed without HMS or source-path injection, trains/renders using local engine code, reads original HMS model formats, and reproduces original-HMS waveform/parameter references. A runtime-only environment also executes save → reload → source-aware MLSA synthesis → PCM16 read/write without pytest installed. |

**Installed-wheel suite: 806 passed, 9 skipped, 0 failures/errors; 815 collected
in 43.233 seconds.** The skips are native-WORLD checks, not engine failures or
unexplained omissions. Real WORLD/pyworld execution remains unverified.

**This is engine extraction/bootstrap only. It is not a finished HMVC
voice-conversion system and introduces no HMVC-specific synthesis improvements.**

Evidence files:

- [Installed-wheel verification, test groups and numerical comparisons](engine-extraction/verification.json)
- [Framework-free runtime-only wheel execution](engine-extraction/runtime-wheel.json)
- [Unchanged original-HMS regression baseline](engine-extraction/upstream-baseline.json)
- [Per-file provenance, hashes, dependency closure and omitted test/helper names](../src/hmvc/engine/UPSTREAM.json)
- [Original-HMS synthetic reference provenance](../tests/data/hms-reference/reference.json)

## 1. Repository inspection and engine boundary

HMVC's original tracked repository was inspected in full: only `README.md` and
`LICENSE` existed at base `eea8f9a6bdd5ae86bf151acc6dcf3579c7844bcd`. There was no
prior package/engine structure to preserve.

The complete HMS tree at the pinned revision contained 95 files / 77 Python
files: 40 runtime Python files, 29 test/support files, and 8 tool/example scripts.
Inspection traced imports, package initializers, deferred/type-checking imports,
model serialization identifiers, resource lookup, native support, packaging,
configuration, tests, and public pipeline entry points.

### Minimum source closure

The extraction tool starts from two functional engine entry points:

- `hms.core.synthesizer`: model/score → state plan → acoustic parameters → audio.
- `hms.core.trainer`: labeled audio → fitted acoustic/duration/pitch/context models.

Following internal imports **including deferred optional branches and parent
initializers** yields **35 Python files**. This is the smallest complete
module-level closure for these entry points **without rewriting package
initializers or removing existing branches**. It is not a claim that a bare
HMM algorithm requires all 35 files.

The optional `source_model=` synthesis path and its eager source initializer
pull in the source/PCA/training components. They are retained rather than
silently deleting existing source-aware behavior. Training is retained so the
foundation can fit new models, not merely play a prebuilt fixture.

### Transferred engine files

Paths below are under `src/hmvc/engine`; all listed stems are `.py` files.

| Directory | Transferred Python files | Purpose |
| --- | --- | --- |
| root (1) | `__init__` | Original lazy model facade and upstream engine version |
| `core/` (16) | `__init__`, `context`, `dsp`, `duration`, `features`, `generation`, `gmm`, `gv`, `hmm`, `labels`, `model`, `phonemes`, `pitch`, `pitch_condition`, `synthesizer`, `trainer` | Complete acoustic/statistical training and synthesis pipeline |
| `data/` (2) | `__init__`, `wavio` | Engine WAV/parameter I/O |
| `source/` (10) | `__init__`, `base`, `cycles`, `generic`, `hmm`, `pca`, `residual`, `synthesis`, `trainer`, `voice` | Existing optional learned-source analysis/prediction/filter integration |
| `vocoder/` (6) | `__init__`, `base`, `builtin`, `mlsa`, `pyworld_backend`, `world_native` | Pure-NumPy waveform generation and optional WORLD adapters |

Additional required/support files:

- `config/phonemes.yaml`: retained **only because `PhonemeSet.default()` uses
  this resource**. This is an inventory/configuration, not a trained voicebank.
- `vocoder/_native/build_world.sh` and `hms_world_capi.cpp`: the two small HMS
  build/C-ABI support files required to retain the optional native-backend path.
  No WORLD implementation, downloaded sources or compiled binary is bundled.
- `LICENSE.hms`: original MIT notice, verbatim.
- `UPSTREAM.json`: generated provenance, original/local SHA-256 hashes and graph.

The installed engine therefore contains **35 Python files plus these five
resource/support/notice/provenance files**. HMVC's new parent `hmvc/__init__.py`
and `pyproject.toml` provide independent packaging. The inspected wheel contains
36 Python files including that parent, and none of HMS's CLI/demo/test utilities.

### Not transferred into the engine

The five runtime modules outside the closure are `hms.cli`, `hms.cli.main`,
`hms.config`, `hms.core.evaluate`, and `hms.data.demo_singer`. No original README,
manual documentation, examples, datasets, benchmark tools, CLI entry points,
voicebanks, trained assets, or Git metadata were bulk-copied. The YAML-to-config
convenience loader is not required by the engine; callers use the retained
`TrainingConfig`, `SynthesisConfig`, `FeatureSpec` and inventory APIs directly.
`parameters.yaml` is therefore not included.

One upstream synthetic generator (`demo_singer.py`) is retained **under
`tests/_support` only**, because the existing regressions depend on its input
fixtures. It is not in `hmvc.engine` or the wheel. No audio dataset is copied.
The small newly generated original-HMS oracle models/arrays are also **test-only,
synthetic regression data**, not production voice assets.

## 2. Resolved dependency graph

```text
core.gmm ----------------------------> NumPy
core.hmm (states, transitions,
          Viterbi/Baum-Welch) --------> core.gmm
core.features -----------------------> NumPy (filterbank/DCT/feature geometry)
core.dsp ----------------------------> core.features
core.generation (MLPG, packed-band
                 Cholesky/solve) ----> core.features
core.gv (GV optimization / GM) -------> core.generation
core.duration, core.pitch -----------> NumPy
core.pitch_condition ----------------> core.labels (MIDI constants)
core.model --------------------------> HMM/features/duration/pitch/context/GV,
                                       phonemes + YAML persistence
core.trainer ------------------------> model + labels/features/HMM/GV,
                                       duration/pitch + vocoder + WAV I/O
core.synthesizer ---------------------> model/state allocation + MLPG/GV/GM,
                                       F0/voicing + feature decoding + vocoder
       optional source_model branch -> source.synthesis / source.hmm
source package ----------------------> source base/cycles/residual/voice/PCA,
                                       source HMM/trainer + common core math
vocoder builtin/mlsa ----------------> common features/DSP + NumPy
vocoder pyworld/native ---------------> optional external bindings/library
```

All internal engine edges now resolve to **`hmvc.engine.*`**, including local
imports in methods and type-checking branches. Private sibling helpers used by
GV/source training remain in the transferred closure. There is no `hms` alias,
namespace shim, symlink, submodule, runtime `sys.path` injection or network fetch
needed to resolve engine imports.

Runtime dependencies remain **`numpy>=1.20` and `PyYAML>=5.4`**. Pytest is a test
extra only. No mandatory Torch, librosa, SciPy, soundfile, pyworld, numba, SPTK or
scikit-learn requirement was added. Build tools are not engine runtime imports.

## 3. Adaptations — no algorithm redesign

1. Import statements changed from `hms...` to `hmvc.engine...`; original class,
   function, parameter and return-type names were preserved, including
   `HMSModel`, `Trainer`, `Synthesizer`, HMM state classes and source APIs.
2. `world_native._try_build()` now resolves the helper beside the installed
   package in `vocoder/_native`, rather than assuming the HMS repository root.
   Diagnostic helper paths were updated. The shell helper's output/source/shim
   locations were adapted to the same package-local directory.
3. The default phoneme YAML is packaged at the relative location already used
   by `PhonemeSet.default()`; its contents and lookup behavior are unchanged.
4. MIT/source attribution headers and provenance were added. The engine facade
   retains HMS's `0.1.0` version for existing `hms_version` model metadata;
   HMVC's bootstrap distribution version is separately `0.0.0`.
5. Applicable upstream tests use the new namespace and test-only generator;
   three monkeypatch import-path strings were adapted and the old checkout
   `sys.path` shim was removed. Core engine assertions were not weakened.

**Non-import strings were not blindly namespace-replaced.** On-disk identifiers
**`hms.source.hmm`** and **`hms.source.pca`**, model version 4 / supported versions
2–4, NPZ keys, YAML layouts, normalization, state statistics, backend names,
legacy C ABI/library symbols and `HMS_NO_AUTO_BUILD` / `HMS_WORLD_SRC` settings
remain compatible. Original-HMS acoustic/source model files load directly in
the port and are regression-tested.

The deterministic maintainer tool reproduces/checks these reviewed changes and
hashes against the pinned, read-only original. It verified **67 selected
engine/notice/test files**, with the original source still pristine. It does not
run at installation, import or synthesis time.

## 4. Actual execution and regression results

Test environment: **CPython 3.11.2, Linux x86_64, NumPy 2.4.6, PyYAML 6.0.3**;
pytest **9.2.0** for regressions. HMS and heavyweight optional libraries were
absent from the fresh wheel environments; import guards additionally reject
them. Native auto-build was disabled in tests, not redesigned in production.

| Run | Passed | Skipped | Failures/errors | Meaning |
| --- | ---: | ---: | ---: | --- |
| Unchanged original HMS baseline | 789 | 9 | 0 | Original suite at audited revision; 798 collected |
| Applicable ported upstream engine regressions | 743 | 9 | 0 | 752 collected; no original HMS code imported |
| New isolation/provenance/model/waveform checks | 63 | 0 | 0 | Includes all 35 fresh-interpreter imports and actual end-to-end execution |
| **Complete installed-wheel suite** | **806** | **9** | **0** | **815 collected, 43.233 s**, run outside the source directory with `python -I -B` |

**46 original collected cases** concerning excluded CLI/config-loader/evaluation/
example surfaces are intentionally outside the ported run, not failures relabeled
as skips or passes. Their omitted test/helper names are recorded in the manifest.
All nine actual skips concern unavailable native WORLD.

The applicable upstream suite checks HMM/GMM estimation and transitions, training
equivalence, Viterbi, banded/dense MLPG agreement, feature transforms, durations,
pitch, contexts, pitch conditioning, GV/GM, persistence/legacy formats, vocoders,
and source training/filter integration.

### Independent original-HMS reference comparisons

`tools/generate_hms_reference.py` ran the **unchanged original** on deterministic
synthetic input and explicitly verified that **HMVC was never imported** during
reference generation. Its original tree remained unchanged. The resulting
fixtures pin four-phone training with delta/delta-delta features, 16 context
units, 18 pitch-conditioned units and GV targets, plus a synthetic source model.
Only non-predictive creation/name/path metadata is normalized.

Ten original-HMS render scenarios are tested:

- ordinary `builtin` and `mlsa`;
- iterative GV and closed-form GM;
- acoustic and state-mean F0 modes;
- model-driven duration;
- authoritative external F0;
- learned-source rendering with both NumPy vocoders.

Each complete pass checks **112 array fields**, including full waveforms, F0,
spectral envelopes, aperiodicity, state IDs/sequences, duration/phone alignment,
and source excitation/support/weights/coverage. Tests compare newly trained
local models, directly loaded original-HMS models, and freshly saved/reloaded
local models. Full acoustic/source model NPZ contents and YAML structure are
also compared; these are not import-only smoke tests.

**Observed on this environment: zero maximum numerical difference and byte-
identical arrays across all ten reference renders.** Regression tolerances are
`rtol=1e-10`, `atol=1e-12`, with NaNs compared consistently; exact byte equality
is an observation here, not a promise across all BLAS/NumPy/platform versions.
The NPZ reference uses lossless deduplication (43 unique arrays), not rounding
or waveform subsampling; it is about 2.6 MB and is never packaged at runtime.

### Independent installed-wheel execution

A **219,179-byte** standalone wheel was built, inspected and installed into
fresh environments. It includes both licensing/provenance and required default
YAML/native support resources; it excludes tests, CLI, demos, source checkouts
and example/benchmark material. `pip check` reported no broken requirements.

A separate environment containing **only the wheel + NumPy + PyYAML** (no HMS,
pytest or heavy libraries) ran `tests/runtime_smoke.py` with `python -I -B` from
an unrelated directory. Original model → save → reload → source-aware MLSA →
PCM16 read/write produced **152 frames / 12,160 samples at 16 kHz**, **194 source
units**. All **16 reference arrays** matched exactly; no `hms` module loaded.
This demonstrates a functioning local engine integration, not just successful
module imports.

## 5. Limits and functionality not established

- **Real WORLD/native and pyworld execution is not verified.** Their wrappers
  and required HMS native support are retained; package-local path dispatch
  and shell syntax were tested without compiling/downloading WORLD. No native
  library or external WORLD source is bundled. All actual synthesis evidence
  here uses the NumPy backends. Do not count the nine native skips as successes.
- `auto` deliberately retains HMS's optional backend selection/auto-build
  behavior. Choose `builtin`/`mlsa` explicitly for lightweight operation, or set
  `HMS_NO_AUTO_BUILD=1` to prevent native build/download attempts.
- **Inherited WAV limitations were preserved, not silently fixed:** named
  PCM16 I/O works; wildcard `wavio` import still encounters its original undefined
  `struct` export, and float32 output does not round-trip correctly. Correctly
  tagged IEEE-float input is not supported by the Python 3.11 `wave` path.
  Tests/report distinguish these known limitations from newly preserved paths.
- CLI, evaluation and the YAML convenience loader are intentionally not
  transferred surfaces, not promised HMVC APIs. Original standalone HMS remains
  independently usable and unmodified.
- Other Python/OS/dependency versions, optional compiled backends, real recorded
  voices, perceptual quality and audio-to-audio conversion have not been
  validated. Metadata retains HMS's declared Python `>=3.9` / dependency lower
  bounds, but this run validates only the environment stated above.
- The engine still expects HMS's score/label, F0, mel-cepstral, AP and frame
  conventions. **No LPC conversion, voice-conversion mapping, new feature
  extraction, speaker adaptation or synthesis improvement was implemented.**

## 6. Licensing and independent maintenance

HMS is MIT, **Copyright (c) 2026 aaiauea**. Its complete notice is preserved in
`hmvc/engine/LICENSE.hms`, the root third-party notice, source headers, and the
installed distribution. HMVC's original MIT license was not changed. Source
paths, revision and original/local hashes are recorded for independent review.

The optional C ABI shim is HMS code; external WORLD is a separate BSD-3-Clause
work. WORLD/pyworld are not included and need their own notice/license review
before any later build/distribution. NumPy/PyYAML retain their own licenses.
No recorded voice/data rights are implied or transferred.

HMS is neither patched nor imported at runtime. Future HMVC work can build on
this local foundation; upstream updates are **explicitly reviewed source ports**,
not implicit dependency upgrades. The extraction checker rejects another HMS
revision. If intentional local engine changes are made later, review/update
provenance and regressions rather than blindly overwriting them or regenerating
oracles to conceal a behavior change.

## 7. Reproduction and next step

Ordinary installation/tests require **no HMS repository**:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -m pytest
```

For a non-editable/wheel-only check, build with `pip wheel --no-deps .`, install
that wheel and the test dependencies into a new venv, then run the suite via
absolute paths from an unrelated CWD. The test suite guards against importing
`hms` and the heavy optional libraries and disables native builds. To repeat the
framework-free check, use an environment containing the wheel + NumPy + PyYAML
only:

```sh
/path/to/runtime-venv/bin/python -I -B /absolute/HMVC/tests/runtime_smoke.py \
  --reference /absolute/HMVC/tests/data/hms-reference \
  --json /absolute/HMVC/.cache/runtime-wheel.json
```

Optional maintainer-only reproduction uses a separate pristine checkout at the
pinned HMS revision; these tools never modify it:

```sh
python -B tools/extract_hms_engine.py --source /separate/pinned/HMS --check
# --write reproduces the selected local files; use only after review.

# Requires NumPy, PyYAML and pytest for original synthetic test fixtures:
python -B tools/generate_hms_reference.py --source /separate/pinned/HMS --write
```

Recommended next step: review/accept this frozen engine boundary and baseline.
Design HMVC-specific contracts/features in a separate phase; do not treat the
bootstrap or import success as completion of the overall HMVC project.
