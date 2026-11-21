# HMVC — Hidden Markov Voice Conversion

HMVC is a lightweight, CPU-friendly voice-conversion project built around classical signal processing. It contains a self-contained HMS synthesis-engine extraction, classical acoustic feature extraction, and a non-neural audio-to-audio conversion pipeline.

## Current capabilities

- **Conversion (hmvc.conversion):** a strict Python port of a working browser voice converter. Its pipeline combines pitch tracking and contour filling, target-frame retrieval with nearest neighbours, TD-PSOLA for the envelope-only path, and grain, smooth-vocoder, or envelope-transfer synthesis.
- **Feature extraction (hmvc.features):** NumPy-only acoustic analysis, including mel-cepstral envelopes, F0, voicing, energy, band aperiodicity, and spectral descriptors.
- **Synthesis engine (hmvc.engine):** the extracted HMS HMM/GMM score-driven singing synthesis engine. This is separate from the audio-to-audio conversion pipeline.

The conversion implementation is **not neural and does not use an HMM**. Target matching uses classical nearest-neighbour retrieval, and the smooth vocoder includes a small GMM-EM model. The HMS engine is not used by the conversion pipeline.

## Browser demo

A client-side demo is available at **[HMVC Web Demo](https://aaiauea.github.io/hidden-markov-voice-conversion/)**.

The demo runs conversion locally in the visitor's browser:
- Audio decoding and resampling use the browser's Web Audio API.
- HMVC's Python conversion code runs in Pyodide with NumPy.
- The small JavaScript layer only handles browser APIs, file selection, audio buffers, progress display, and WAV download. It does not implement conversion DSP.
- Audio is not uploaded to a conversion server.

The first load downloads the Pyodide runtime, NumPy, and the Python source archive, so it can take a while and use substantial memory. The demo is experimental; long recordings may be slow on low-memory devices. Source and target audio are processed as mono at 22050 Hz. The target is limited to 150 seconds, matching the original converter's behavior.

## Install

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Runtime dependencies are **NumPy and PyYAML**. There is no mandatory PyTorch, librosa, SciPy, soundfile, pyworld, or native WORLD dependency.

## Voice conversion API

The conversion pipeline accepts mono float audio at 22050 Hz. Decode and resample audio before calling it.

```python
from hmvc.conversion import ConversionConfig, convert, encode_wav

result = convert(
    source_samples,                 # mono float array, 22050 Hz
    target=target_samples,           # optional target voice, same format
    config=ConversionConfig(
        mode="voc",                  # "voc", "grain", or "env"
        pitch_semitones=0.0,
        formant_factor=1.0,
        envelope_blend=0.8,
        neighbours=6,
    ),
)
wav_bytes = encode_wav(result.samples)
```

Modes:
- `voc`: smooth-vocoder synthesis when a usable target database is available.
- `grain`: target-grain retrieval and resynthesis.
- `env`: envelope transfer. If no usable target database is available, `voc` and `grain` fall back to the envelope path.

The defaults mirror the original interface. Presets and voice-range constants are exposed as `PRESETS` and `VOICE_RANGES`; see `hmvc.conversion.config`.

CLI usage:

```sh
python -m hmvc conversion source.wav --target target.wav --mode voc
```

See [the conversion pipeline notes](docs/conversion.md) for processing order, parameters, module mapping, and known quality limitations. The current smooth-vocoder and envelope modes can sound buzzy or unclear; those characteristics are preserved in this strict port rather than silently redesigned.

## Acoustic feature extraction

hmvc.features analyses WAV files or arrays at common sample rates into frame-synchronous streams:
- compact mel-cepstral representation of the spectral envelope;
- F0, voicing, and periodicity;
- frame energy and band aperiodicity;
- spectral descriptors (centroid, bandwidth, rolloff, flatness, tilt, ZCR, and flux).

Corpus normalisation is mergeable. It uses no neural model, transcript, WORLD/SPTK library, or additional runtime dependency.

```python
from hmvc.features import AnalysisConfig, FeatureExtractor

feats = FeatureExtractor(AnalysisConfig(sample_rate=16000)).extract_file("utt.wav")
feats.mcep, feats.f0, feats.voiced, feats.energy_db
```

CLI:

```sh
python -m hmvc.features extract utt.wav utt.npz
python -m hmvc.features info utt.npz
```

See [the feature-extraction report](docs/feature-extraction.md) for representation choices, comparisons, configuration, benchmarks, and limitations.

## HMS engine

The selected engine from [Hidden Markov Singer](https://github.com/aaiauea/hidden-markov-singer) lives in hmvc.engine. HMS is not a runtime dependency. The extraction preserves HMM/GMM training and transitions, duration/pitch models, MLPG, GV/GM, context/pitch conditioning, model save/load, and builtin/MLSA vocoders.

The engine remains score-driven; it is not the audio-to-audio converter. See the [engine extraction and verification report](docs/hms-engine-extraction.md), [third-party notices](THIRD_PARTY_NOTICES.md), [original MIT license](src/hmvc/engine/LICENSE.hms), and [per-file provenance](src/hmvc/engine/UPSTREAM.json).

## Tests

```sh
.venv/bin/python -m pip install -e '.[test]'
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -m pytest
```

The suite includes engine regressions, feature-extraction checks, and conversion parity fixtures. It blocks imports of HMS and heavy optional libraries and does not require network access or the original source implementation.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
