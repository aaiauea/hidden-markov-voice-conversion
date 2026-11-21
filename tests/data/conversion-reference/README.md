# Conversion reference fixtures (strict 1:1 port of the browser voice converter)

These fixtures hold outputs of the **original** DSP code — the browser app
original DSP script, extracted verbatim and stored at
`reference-source/voice-converter.js` (HTML/UI wrapper omitted) — re-run in Node on fixed synthetic
source/target signals.

They let the HMVC NumPy port in `src/hmvc/conversion/` be checked against the
original without a browser (`tests/conversion/test_reference.py`). The harness
never imports `hmvc`, and the fixture is maintainer-generated.

## Files

| file | contents |
|---|---|
| `reference-source/voice-converter.js` | original DSP JavaScript extracted verbatim from the browser app; HTML and UI are not retained. The harness adds only the documented deterministic RNG and read-only instrumentation. |
| `signals.json` | the fixed synthetic source/target (recipe `tests/_support/conversion_signals.py`, recipe SHA-256 `29d7467697cb4dd6ab781a7ea7ac2c3c1c76b70cd1f9e3ba294fadf2be9525ac`, `signals_sha256` `7015ca983e4d4fec3298c85c8bc774c6c35f4c03085e79759d7a6499a8535514`). Source 23415 samples, target 29169 samples, 22050 Hz. |
| `reference.json` | per-case outputs: `samples` (float32 PCM), `ratio`, `f0`, `vo`, `nfr`, `rs`, `sa`, `sv`, `sf`, `nb`, `db`, `debug`, per-case metadata and the provenance chain back to `signals.json`. |

## Cases

| case | mode | target | pitch (st) | formant | blend | notes |
|---|---|---|---|---|---|---|
| `env-no-target` | `env` | no | 3.0 | 1.0 | 0% | retrieval disabled |
| `env-with-target` | `env` | yes | 3.0 | 1.15 | 80% | retrieval powers the envelope path |
| `grain` | `grain` | yes | 0.0 | 1.0 | 80% | target grains |
| `voc` | `voc` | yes | 0.0 | 1.0 | 80% | seeded `LcgRandom(1234)` noise |

`debug` (vocode only) captures the internal GMM/EM state (`act`, `X`, `Y`,
`mu`, `vr`, `pi`, `G`, `MX`, `MY`, `AA`, `P`, `F0`, `HM`, `PH`, `Am`, `out`,
`noise`) for parity review.

## What the harness changes vs the original (and exactly why)

`tools/generate_conversion_reference.js` is a UI-free driver that re-drives the
same control flow as the original `convert()`. It is *not* a verbatim run of
the browser — it must be, because a browser is not available. Two
modifications are made, and both are documented and deliberately
conservative:

1. **`Math.random` → deterministic 64-bit LCG.** The browser uses an unseeded
   `Math.random`, whose noise is not reproducible by any other implementation
   (and is exactly the failure mode observed earlier: nondeterministic noise
   made the port diverge from the original). The harness monkeypatches
   `Math.random` with a deterministic 64-bit LCG using the same constants as
   `hmvc.conversion.rng` (`MULT 0x5851f42d4c957f2d`, `INC 0x14057b7ef767814f`,
   `float(state >> 11) / 2^53`), seeded with the same `1234` the port uses.
   Everything else in the original JS is untouched.
2. **Vocode instrumentation.** One instrumentation assignment is inserted
   immediately before the original `vocode` function's final `return out;`
   (and one after the per-frame loop, before the phase-fill section) so the
   internals can be captured for the fixture. No DSP arithmetic is changed;
   only variables are read, not written.

In both cases the underlying reference algorithm is unchanged. The original DSP script is stored separately from the HTML/UI wrapper. The
harness patches `Math.random`, inserts instrumentation at fixed markers, then
evaluates the original code as a plain function.

## Regeneration (opt-in, maintainers only)

```sh
node tools/generate_conversion_reference.js
```

Requires Node (>= 18 for `BigInt`; developed with Node v22.22.3). No other
dependencies — the harness uses only Node stdlib. It overwrites
`reference.json`. Do not regenerate merely to conceal a regression.

If `tests/_support/conversion_signals.py` changes, its SHA-256 changes, and
`tests/conversion/test_reference.py::test_fixture_recipe_unchanged` then fails
until you regenerate — review the diff before committing.

## Verification

`tests/conversion/test_reference.py` (the parity test) re-runs the port and
asserts, for each case, that the outputs match the fixture bit-for-bit:

- `samples` — exact equality (`float32`),
- `ratio` — abs 1e-12,
- `pitch_shifted` flag — reproduces the original's TD-PSOLA-or-not decision,
- `f0`, `vo` — exact (`float32`/`uint8`),
- `source_length`/`target_length` — exact,
- `n_frames`, `target_database_active` — exact.

It also asserts the fixture's recipe hash and the synthetic-signal geometry.

Run the complete suite with:

```sh
python -m pytest -q
```

## Provenance and notes

- No recorded voice or downloaded dataset is included; all signals are
  procedurally synthesized (`tests/_support/conversion_signals.py`). Re-running
  the recipe must reproduce `signals.json` byte-for-byte.
- `reference.json` records its own provenance (`generated_by`, `source_file`
  pointing at the standalone JavaScript reference, per-case values, and the
  `signals_provenance` chain back to `signals.json`).
- The fixtures are test-only and are not packaged.
