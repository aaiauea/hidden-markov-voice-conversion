"""Data container for the conversion result (see ``docs/conversion.md``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConversionResult:
    """Result of a 1:1 voice conversion.

    ``f0``/``voiced`` are the contour after gap-filling (post ``fill_gaps``),
    not the raw track.  ``samples`` is the 22050 Hz, peak-normalized output.
    """

    samples: np.ndarray
    sample_rate: int
    mode: str                     # effective mode: 'grain' | 'voc' | 'env'
    pitch_ratio: float
    pitch_shifted: bool           # TD-PSOLA applied to the source
    direct_mode: bool             # grain/vocoder synthesis was used
    f0: np.ndarray                # smoothed, gap-filled f0 contour
    voiced: np.ndarray            # smoothed, gap-filled voicing flags
    source_length: int            # samples
    target_length: Optional[int]  # samples, or None if no target loaded
    n_frames: int                 # analysis frames
    target_database_active: bool
