# Established-implementation reference fixtures for feature extraction

These fixtures hold outputs of **established third-party implementations**,
computed on HMVC's synthetic test signals:

- WORLD via `pyworld` (CheapTrick, Harvest, DIO+StoneMask)
- SPTK via `pysptk` (`sp2mc`, `mc2sp`, `freqt`, `mcepalpha`)
- `librosa` (STFT, RMS, spectral descriptors, YIN, zero-crossing rate)

They let the HMVC NumPy implementations in `src/hmvc/features/` be checked
against those libraries without importing them. The generator asserts that
no `hmvc` module was imported. Ordinary tests import only NumPy, and
`tests/conftest.py` blocks `pyworld`, `pysptk`, `librosa` and `scipy`.

- `reference.npz`: arrays, keyed `<case>/<field>`. The `freqt/*` and
  `mcepalpha/*` keys hold signal-independent checks.
- `reference.json`: per-case geometry (sample rate, hop, frame length, FFT
  size, MCEP order/alpha, frame count), analysis settings, package versions,
  the SHA-256 of the signal recipe `tests/_support/feature_signals.py`, the
  npz SHA-256 and an array manifest.

| case | signal | fs |
|---|---|---|
| `vowel_low` | /a/ gliding 100-130 Hz with vibrato | 16 kHz |
| `vowel_high` | /i/ gliding 260-320 Hz | 16 kHz |
| `mixed` | silence, fricative noise, voiced, silence | 16 kHz |
| `vowel_22k` | /a/ at an odd frame length (551) and a fractional WORLD hop | 22.05 kHz |

The CheapTrick oracle is driven by the **true** F0. This isolates the
envelope algorithm from differences between F0 estimators.

## Known, intentional convention differences (tests account for them)

| field | difference | handling in tests |
|---|---|---|
| `vowel_22k/cheaptrick_log_sp` | WORLD places frame *t* at sample `floor(0.005·t·fs + 0.5)` (hop 110.25 samples). HMVC uses an integer hop of 110. | Run HMVC's CheapTrick on WORLD's origins; it then agrees to ~2e-6 dB. |
| `mixed/cheaptrick_log_sp` | WORLD adds a random floor (~1e-16) to digital silence. | Compare only bins with power > 1e-9. |
| `mixed/sptk_mcep` | HMVC floors power at 1e-12 before taking the log. SPTK takes the log of WORLD's ~1e-16 silent bins. | Compare with `floor=0`, or only on frames that are not silent. |
| `*/librosa_flatness` | librosa floors `\|X\|²` at `amin = 1e-10`. HMVC's power spectrum is normalised by `Σw²` and floored at 1e-12. | Pass `power_floor = 1e-10 / Σw²`. |
| `*/librosa_rms`, `*/librosa_zcr` | librosa centres these frames with `L//2` samples of left padding (its STFT effectively uses `(L+1)//2` for odd `L`). Its ZCR counts the frame's first sample against its predecessor. | Frame with `left=L//2`, and allow a ZCR tolerance of `1/L`. |
| `*/harvest_f0`, `*/dio_f0`, `*/librosa_yin` | Different algorithms, not oracles. | Used only as accuracy baselines against `true_f0`. |

The reference values are float64, except `x`, `cheaptrick_log_sp`,
`librosa_rms` and `librosa_log_stft`, which are float32 to keep the
fixture at ~1.4 MiB. Tests use relative tolerances accordingly.

## Regeneration (opt-in, maintainers only)

You need a separate environment with NumPy, `pyworld`, `pysptk` and `librosa`
(the `reference` extra: `pip install -e '.[reference]'`). The fixture was
produced with pyworld 0.3.5, pysptk 1.0.1, librosa 0.11.0, NumPy 2.4.6 and
Python 3.11.2.

```sh
python -B tools/generate_feature_reference.py           # dry run: summary only
python -B tools/generate_feature_reference.py --write   # rewrite the fixture
```

A changed signal recipe changes the recorded SHA-256, and the tests then
fail until you regenerate. Review regenerated data. Do not regenerate merely
to conceal a regression.

Everything is procedurally synthesised: no recorded voice or downloaded
dataset is included. The fixtures are test-only and are not packaged.
