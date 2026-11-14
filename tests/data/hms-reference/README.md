# Original-HMS synthetic regression oracles

These small test fixtures were generated from the **unchanged original HMS** at
`ba1eac18146f017f8cc3cae6acdb7963b8f4149e`, not from the HMVC port. Generation
asserts that no `hmvc` module was imported and that the original HMS source tree
remained pristine. See `reference.json` for source revision, environment, input
helper hash, per-file hashes and the mapping of 112 case/field comparisons to
43 losslessly deduplicated arrays.

- `acoustic/`: an original-format toy model trained on two procedurally generated
  PCM clips, four phones, context/pitch-conditioned units and GV statistics.
- `source/`: the original upstream synthetic two-component source/PCA test model.
- `renders.npz`: complete original waveforms, F0, spectral envelope, aperiodicity,
  state/duration information, and learned-source fields for ten render modes.
- Only model name, creation date and input-path provenance are normalized;
  predictive parameters and render arrays are not modified, rounded or sampled.

No recorded voice, production voicebank, downloaded dataset or pretrained voice
is included. The fixtures are test-only and are not packaged in the wheel.
Ordinary tests require no HMS checkout. Applicable source attribution is in
`THIRD_PARTY_NOTICES.md` and `src/hmvc/engine/LICENSE.hms`.

Opt-in regeneration, using an environment with NumPy, PyYAML and pytest:

```sh
python -B tools/generate_hms_reference.py --source /separate/pristine/HMS --write
```

Review regenerated data and provenance; do not regenerate merely to conceal a
regression. The generator rejects a different upstream revision. The input
recipe lives in `tests/_support/engine_case.py`; the copied synthetic generator
in `tests/_support/demo_singer.py` separately supports the ported upstream suite.
