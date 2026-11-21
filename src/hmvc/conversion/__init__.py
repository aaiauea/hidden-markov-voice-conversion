"""HMVC 2.0 — Phase 1: strict 1:1 port of the retrieval voice converter.

This package ports the friend's working, non-neural voice-conversion
implementation (``voice-converter (4).html``) into HMVC.  The HTML file is the
behavioral specification: every mathematical operation, ordering and numerical
parameter is reproduced as faithfully as possible.

Nothing here is redesigned or improved.  Quirky behavior from the original is
preserved and documented in ``docs/conversion.md``.

Audio at 22050 Hz (:data:`hmvc.conversion.constants.SR`) is the pipeline's
internal representation; see :mod:`hmvc.conversion.wavio` for the input/output
path.
"""

from hmvc.conversion.config import PRESETS, VOICE_RANGES, ConversionConfig, preset_params
from hmvc.conversion.pipeline import convert
from hmvc.conversion.types import ConversionResult
from hmvc.conversion.wavio import encode_wav, load_audio, write_wav

__version__ = "0.1.0"

__all__ = [
    "ConversionConfig",
    "ConversionResult",
    "PRESETS",
    "VOICE_RANGES",
    "preset_params",
    "convert",
    "encode_wav",
    "write_wav",
    "load_audio",
]
