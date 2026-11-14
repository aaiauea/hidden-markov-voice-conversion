#!/usr/bin/env python3
"""Regenerate tiny synthetic regression oracles using ONLY the pinned original HMS.

Maintainer-only, opt-in: ordinary HMVC imports/tests never need an HMS checkout.
Existing fixtures are overwritten only with --write. No recorded voice is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
REVISION = "ba1eac18146f017f8cc3cae6acdb7963b8f4149e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"), timeout=15).strip()


def snapshot(source):
    return {p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(source).parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--write", action="store_true", required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    status_args = ("status", "--porcelain", "--untracked-files=all", "--ignored=matching")
    if git(source, "rev-parse", "HEAD") != REVISION or git(source, *status_args):
        parser.error("Use the pristine pinned original HMS checkout, without caches/build artifacts")
    before = snapshot(source)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HMS_NO_AUTO_BUILD"] = "1"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(source))
    import numpy as np
    from tests._support.engine_case import train, renders, render_arrays, source_model

    output = PROJECT / "tests/data/hms-reference"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hms-original-oracle-") as work:
        model = train("hms", work)
        model.save(output / "acoustic")
        source_tier = source_model("hms")
        source_tier.save(output / "source")
        arrays, array_keys, cases = {}, {}, {}
        for name, result in renders("hms", model, source_tier):
            for field, array in render_arrays(result).items():
                value = np.asarray(array)
                # Lossless content deduplication only, never rounding/subsampling.
                key = hashlib.sha256((value.dtype.str + str(value.shape)).encode() + value.tobytes()).hexdigest()
                arrays[key] = value
                array_keys[name + "/" + field] = key
            cases[name] = {"frames": len(result.params.f0), "samples": len(result.audio),
                           "diagnostics": result.diagnostics,
                           "source_units": result.source.n_units if result.source else 0}
        np.savez_compressed(output / "renders.npz", **arrays)
    if any(name == "hmvc" or name.startswith("hmvc.") for name in sys.modules):
        raise RuntimeError("Oracle contamination: HMVC must NOT be imported during original HMS generation")
    foreign = [name for name, mod in sys.modules.items()
               if (name == "hms" or name.startswith("hms.")) and getattr(mod, "__file__", None)
               and not Path(mod.__file__).resolve().is_relative_to(source)]
    if foreign:
        raise RuntimeError(f"Wrong original HMS origins: {foreign}")
    if before != snapshot(source) or git(source, *status_args):
        raise RuntimeError("Original HMS changed during reference generation")
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
             if p.is_file() and p.suffix in (".npz", ".yaml")}
    metadata = {
        "repository": "https://github.com/aaiauea/hidden-markov-singer", "revision": REVISION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "distributions": {name: importlib.metadata.version(name) for name in ("numpy", "PyYAML", "pytest")},
        "source_unchanged": True, "hmvc_imported_during_generation": False,
        "fixture_code_sha256": sha(PROJECT / "tests/_support/engine_case.py"),
        "normalization": "Only model name, creation date, and input-path metadata normalized; predictive parameters untouched",
        "training": {"phones": sorted(model.hmms), "contexts": len(model.contexts),
                     "pitch_models": len(model.pitch_models), "global_variance": model.gv_stats is not None},
        "cases": cases, "array_keys": array_keys, "files_sha256": files,
    }
    (output / "reference.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {len(cases)} original-HMS render oracles, {len(array_keys)} array comparisons "
          f"({len(arrays)} unique arrays); source unchanged; HMVC not imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
