#!/usr/bin/env python3
"""Maintainer-only: write ``tests/data/conversion-reference/signals.json``.

The JSON is the shared, deterministic comparison input for the reference
fixture.  The Node harness (tools/generate_conversion_reference.js) reads it,
replays the original DSP code, and writes the reference outputs.

Usage:
    python tools/generate_signals.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "_support"))
import conversion_signals as CS  # noqa: E402

REF_DIR = ROOT / "tests" / "data" / "conversion-reference"
SIGNALS = REF_DIR / "signals.json"
RECIPE = ROOT / "tests" / "_support" / "conversion_signals.py"


def main() -> int:
    src, src_f0 = CS.source()
    tgt, tgt_f0 = CS.target()
    recipe_sha256 = hashlib.sha256(RECIPE.read_bytes()).hexdigest()

    out = {
        "generated_by": "tools/generate_signals.py",
        "recipe": "tests/_support/conversion_signals.py",
        "recipe_sha256": recipe_sha256,
        "sample_rate": CS.SR,
        "signals": {
            "source": [float(v) for v in np.ascontiguousarray(src)],
            "target": [float(v) for v in np.ascontiguousarray(tgt)],
        },
        "lengths": {"source": int(len(src)), "target": int(len(tgt))},
    }
    signals_bytes = json.dumps(out["signals"]).encode("utf-8")
    out["signals_sha256"] = hashlib.sha256(signals_bytes).hexdigest()

    REF_DIR.mkdir(parents=True, exist_ok=True)
    SIGNALS.write_text(json.dumps(out))
    print(f"wrote {SIGNALS}")
    print(f"  source {len(src)} samples  target {len(tgt)} samples  sr {CS.SR}")
    print(f"  recipe sha256: {recipe_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
