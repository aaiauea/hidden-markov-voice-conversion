"""Framework-free installed-wheel execution check (no source-path injection).

Run with python -I -B from a venv containing the wheel, NumPy and PyYAML only.
The synthetic reference directory is test data, not an HMS runtime dependency.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile


class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"hms", "pytest", "torch", "librosa", "scipy", "soundfile", "pyworld", "pysptk", "numba", "sklearn"}:
            raise ModuleNotFoundError(fullname, name=fullname.split(".")[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, Guard())
    os.environ["HMS_NO_AUTO_BUILD"] = "1"
    import numpy as np
    import hmvc.engine
    from hmvc.engine.core.labels import parse
    from hmvc.engine.core.model import HMSModel
    from hmvc.engine.core.phonemes import PhonemeSet
    from hmvc.engine.core.synthesizer import SynthesisConfig, Synthesizer
    from hmvc.engine.data import wavio
    from hmvc.engine.source.hmm import SourceHMMModel

    assert "site-packages" in hmvc.engine.__file__, "Use an installed wheel, not the editable source checkout"
    assert PhonemeSet.default().resolve("a") is not None
    reference = args.reference.resolve()
    meta = json.loads((reference / "reference.json").read_text())
    model, source = HMSModel.load(reference / "acoustic"), SourceHMMModel.load(reference / "source")
    with tempfile.TemporaryDirectory(prefix="hmvc-runtime-only-") as work:
        directory = Path(work)
        model.save(directory / "acoustic")
        source.save(directory / "source")
        model, source = HMSModel.load(directory / "acoustic"), SourceHMMModel.load(directory / "source")
        score = parse("case 0.00 0.04 sil -\ncase 0.04 0.24 a 60\ncase 0.24 0.32 s -\n"
                      "case 0.32 0.52 i 64\ncase 0.52 0.72 a 67\ncase 0.72 0.76 sil -\n")
        result = Synthesizer(model, SynthesisConfig(vocoder="mlsa", seed=11, vibrato=False)).synthesize(score, source_model=source)
        arrays = {"audio": result.audio, "f0": result.params.f0, "sp": result.params.sp,
                  "ap": result.params.ap, "notes": result.notes, "state_ids": result.state_ids,
                  "durations": result.durations, "f0_semitones": result.f0_semitones,
                  "phones": np.asarray(result.phones), "state_sequence": np.asarray(result.state_sequence)}
        for field in ("excitation", "support", "weights", "coverage", "voiced", "f0_hz"):
            arrays["source_" + field] = getattr(result.source, field)
        largest, exact = 0.0, True
        with np.load(reference / "renders.npz", allow_pickle=False) as expected:
            for field, actual in arrays.items():
                actual = np.asarray(actual)
                wanted = expected[meta["array_keys"]["source_mlsa/" + field]]
                assert actual.shape == wanted.shape
                exact &= actual.dtype == wanted.dtype and actual.tobytes() == wanted.tobytes()
                if wanted.dtype.kind in "fci":
                    assert np.allclose(actual, wanted, rtol=1e-10, atol=1e-12, equal_nan=True), field
                    finite = np.isfinite(actual) & np.isfinite(wanted)
                    if finite.any():
                        largest = max(largest, float(np.max(np.abs(actual[finite] - wanted[finite]))))
                else:
                    assert np.array_equal(actual, wanted), field
        assert np.isfinite(result.audio).all() and np.max(np.abs(result.audio)) > 0.01
        assert result.source.n_units > 0
        wavio.write_wav(directory / "result.wav", result.audio, model.spec.fs)
        audio, fs = wavio.read_wav(directory / "result.wav")
        assert len(audio) == len(result.audio) and fs == 16000
    versions = {}
    for name in ("hidden-markov-voice-conversion", "numpy", "PyYAML", "pytest", "hms", "torch", "librosa", "scipy", "pyworld"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    summary = {"status": "executed_and_integrated", "scope": "Extracted engine only; no HMVC conversion features",
               "engine_origin": hmvc.engine.__file__, "distributions": versions,
               "case": "original model -> save -> reload -> MLSA + learned source -> PCM16 read/write",
               "frames": len(result.params.f0), "samples": len(result.audio), "fs": fs,
               "source_units": result.source.n_units, "reference_arrays_compared": len(arrays),
               "max_absolute_numeric_difference": largest, "all_arrays_byte_identical_in_this_environment": bool(exact),
               "hms_modules_loaded": [n for n in sys.modules if n == "hms" or n.startswith("hms.")]}
    assert summary["hms_modules_loaded"] == []
    assert versions["hms"] is None and versions["pytest"] is None
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
