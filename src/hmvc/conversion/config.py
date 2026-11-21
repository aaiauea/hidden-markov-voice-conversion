"""Configuration for the strict 1:1 voice-converter port.

Mirrors the original browser UI knobs and their defaults exactly:

* pitch slider (semitones, range -12..12, step 0.5, default 0)
* formant slider (× , range 0.6..1.5, step 0.01, default 1.0)
* envelope blend (%, range 0..100, step 1, default 80)
* neighbours (int, range 1..16, step 1, default 6)
* voice range (presets): low 60-300, standard 70-450 (default), high 120-700
* method: "voc" (default, smooth vocoder (trained)), "grain" (target grains), "env" (envelope only)
* auto-pitch checkbox: checked
* presets (pitch, form, ret): Custom (none), Deeper voice, Higher voice, Child,
  Giant, Match target only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

MODES = ("voc", "grain", "env")

#: (label, (fmin, fmax)) voice-range options of the original UI.
VOICE_RANGES: Dict[str, Tuple[float, float]] = {
    "low": (60.0, 300.0),
    "standard": (70.0, 450.0),
    "high": (120.0, 700.0),
}

#: Presets from the original UI ("pitch", "form", "ret") — ret in percent.
#: ``None`` for "Custom" (keeps current values).
PRESETS: Dict[str, Optional[Tuple[float, float, int]]] = {
    "Custom": None,
    "Deeper voice": (-5.0, 0.88, 80),
    "Higher voice": (5.0, 1.15, 80),
    "Child": (8.0, 1.30, 80),
    "Giant": (-10.0, 0.75, 80),
    "Match target only": (0.0, 1.0, 100),
}


@dataclass(frozen=True)
class ConversionConfig:
    """Parameters of one conversion.  All defaults match the original UI."""

    mode: str = "voc"
    pitch_semitones: float = 0.0
    formant_factor: float = 1.0
    envelope_blend: float = 0.8
    neighbours: int = 6
    fmin: float = 70.0
    fmax: float = 450.0
    auto_pitch: bool = True
    target_max_seconds: float = 150.0
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not -12.0 <= self.pitch_semitones <= 12.0:
            raise ValueError(f"pitch_semitones must be within [-12, 12], got {self.pitch_semitones!r}")
        if not 0.6 <= self.formant_factor <= 1.5:
            raise ValueError(f"formant_factor must be within [0.6, 1.5], got {self.formant_factor!r}")
        if not 0.0 <= self.envelope_blend <= 1.0:
            raise ValueError(f"envelope_blend must be within [0, 1], got {self.envelope_blend!r}")
        if not 1 <= self.neighbours <= 16 or int(self.neighbours) != self.neighbours:
            raise ValueError(f"neighbours must be an integer in [1, 16], got {self.neighbours!r}")
        if not (30.0 <= self.fmin < self.fmax <= 1000.0):
            raise ValueError(f"require 30 <= fmin < fmax <= 1000, got [{self.fmin}, {self.fmax}]")
        if self.target_max_seconds <= 0:
            raise ValueError(f"target_max_seconds must be positive, got {self.target_max_seconds!r}")
        if self.rng_seed is not None and not isinstance(self.rng_seed, int):
            raise ValueError(f"rng_seed must be an int, got {self.rng_seed!r}")


def preset_params(name: str):
    """Return ``(pitch, formant_factor, envelope_blend)`` for ``name``.

    Returns ``None`` for "Custom" (the original UI then keeps current values).
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r} (choose from {sorted(PRESETS)})")
    spec = PRESETS[name]
    if spec is None:
        return None
    return spec[0], spec[1], spec[2] / 100.0
