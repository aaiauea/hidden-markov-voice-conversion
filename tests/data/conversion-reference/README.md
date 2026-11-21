# Conversion reference fixtures

These are frozen reference outputs from the original voice-conversion
implementation. They let the HMVC Python port be checked against the recorded
behavior without retaining the original browser application or its source
code in this repository.

## Files

| file | contents |
|---|---|
| `signals.json` | Fixed synthetic source and target signals. |
| `reference.json` | Recorded output arrays and intermediate values for the four parity cases. |

The test suite compares the Python implementation against these checked-in
fixtures. The fixtures are intentionally static: this repository does not
include the original implementation or a source-language fixture generator.
If conversion behavior changes, update the reference data only through a
separately reviewed process that has access to the original source.

## Cases

| case | mode | target | pitch (st) | formant | blend |
|---|---|---|---|---|---|
| `env-no-target` | `env` | no | 3.0 | 1.0 | 0% |
| `env-with-target` | `env` | yes | 3.0 | 1.15 | 80% |
| `grain` | `grain` | yes | 0.0 | 1.0 | 80% |
| `voc` | `voc` | yes | 0.0 | 1.0 | 80% |

No recorded voice or downloaded dataset is included. All comparison signals
are procedurally synthesized by `tests/_support/conversion_signals.py`.

## Verification

Run the parity test:

```sh
python -m pytest tests/conversion/test_reference.py -q
```

Run the complete suite:

```sh
python -m pytest -q
```
