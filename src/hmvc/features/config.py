"""Analysis configuration for HMVC feature extraction.

Everything that changes the numerical value of a feature lives in
:class:`AnalysisConfig`.  The configuration is immutable, validated on
construction, JSON-serialisable, and is stored next to every extracted feature
set so that features computed with different settings can never be mixed up
silently.

Quantities that depend on the sample rate (hop in samples, FFT size, the
all-pass warping coefficient, the mel-cepstral order, ...) are resolved for a
concrete rate by :meth:`AnalysisConfig.resolve`, which returns an
:class:`AnalysisGeometry`.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: Window functions supported by :func:`hmvc.features.framing.get_window`.
WINDOWS = ("hann", "hamming", "blackman", "rectangular")

#: Spectral-envelope estimators supported by :mod:`hmvc.features.envelope`.
ENVELOPE_METHODS = ("cheaptrick", "true_envelope", "lifter")


#: Lowest analysis rate accepted as a resampling target.
MIN_SAMPLE_RATE = 4000


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= ``n`` (and >= 2)."""
    return 1 << max(1, int(math.ceil(math.log2(max(int(n), 2)))))


def default_mcep_order(sample_rate: int) -> int:
    """Default mel-cepstral *order* M (the representation has M + 1 values).

    The values follow common practice for statistical voice conversion and
    are supported by the order sweep in ``docs/feature-extraction.md``:
    24 at <= 16 kHz, 34 at <= 24 kHz and 39 above.
    """
    if sample_rate <= 16000:
        return 24
    if sample_rate <= 24000:
        return 34
    return 39


@dataclass(frozen=True)
class AnalysisGeometry:
    """Sample-rate-dependent quantities resolved from an :class:`AnalysisConfig`."""

    sample_rate: int
    hop_length: int
    frame_length: int
    fft_size: int
    mcep_order: int
    mcep_alpha: float
    f0_floor: float
    f0_ceil: float

    @property
    def n_bins(self) -> int:
        return self.fft_size // 2 + 1

    @property
    def frame_period_ms(self) -> float:
        return 1000.0 * self.hop_length / self.sample_rate

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AnalysisConfig:
    """User-facing analysis settings.

    Framing
        Every feature shares one frame grid: frame ``t`` is centred on sample
        ``t * hop_length`` and an ``n``-sample signal has ``1 + n // hop``
        frames (the WORLD convention).  ``hop_ms``/``frame_ms`` are converted
        to samples (rounded) unless ``hop_length``/``frame_length`` are given
        explicitly.  ``frame_ms`` is the *fixed* analysis window used by the
        energy, spectral-descriptor and log-spectrum streams; the spectral
        envelope (CheapTrick) and band aperiodicity use F0-adaptive windows.

    Audio
        ``sample_rate=None`` analyses audio at its native rate; an integer
        resamples every input to that rate first (band-limited, NumPy only),
        which is what makes features from differently sampled corpora
        comparable.

    Cepstrum
        ``mcep_order``/``mcep_alpha`` default to sample-rate dependent values
        (:func:`default_mcep_order`, :func:`hmvc.features.cepstrum.mcep_alpha`).
    """

    # -- audio ------------------------------------------------------------
    sample_rate: Optional[int] = None
    remove_dc: bool = True

    # -- framing ----------------------------------------------------------
    hop_ms: float = 5.0
    hop_length: Optional[int] = None
    frame_ms: float = 25.0
    frame_length: Optional[int] = None
    fft_size: Optional[int] = None
    window: str = "hann"

    # -- F0 / voicing -----------------------------------------------------
    f0_floor: float = 60.0
    f0_ceil: float = 800.0
    yin_threshold: float = 0.15
    voicing_threshold: float = 0.25
    voicing_extension_threshold: float = 0.60
    f0_max_jump: float = 0.20
    f0_jump_cost: float = 2.0
    silence_threshold_db: float = -50.0
    absolute_silence_db: float = -80.0
    f0_median_frames: int = 3
    min_voiced_ms: float = 15.0
    f0_refine_periods: float = 2.0

    # -- spectral envelope / cepstrum -------------------------------------
    envelope_method: str = "cheaptrick"
    cheaptrick_q1: float = -0.15
    mcep_order: Optional[int] = None
    mcep_alpha: Optional[float] = None
    keep_envelope: bool = False

    # -- aperiodicity -----------------------------------------------------
    aperiodicity_bands: Tuple[float, ...] = field(default=())
    n_aperiodicity_bands: int = 5
    aperiodicity_periods: float = 4.0

    # -- descriptors / floors ---------------------------------------------
    tilt_range_hz: Tuple[float, float] = (100.0, 5000.0)
    rolloff_percent: float = 0.85
    energy_floor_db: float = -100.0
    power_floor: float = 1e-12

    # -- implementation ---------------------------------------------------
    block_frames: int = 512

    def __post_init__(self) -> None:
        # Normalise tuple-like fields so JSON round-trips compare equal.
        object.__setattr__(self, "aperiodicity_bands",
                           tuple(float(v) for v in self.aperiodicity_bands))
        object.__setattr__(self, "tilt_range_hz",
                           tuple(float(v) for v in self.tilt_range_hz))
        self._validate()

    # -- validation -------------------------------------------------------

    def _validate(self) -> None:
        def positive(name: str, value, allow_none: bool = False) -> None:
            if value is None and allow_none:
                return
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")

        positive("sample_rate", self.sample_rate, allow_none=True)
        if self.sample_rate is not None and int(self.sample_rate) != self.sample_rate:
            raise ValueError("sample_rate must be an integer number of Hz")
        if self.sample_rate is not None and self.sample_rate < MIN_SAMPLE_RATE:
            raise ValueError(f"sample_rate must be at least {MIN_SAMPLE_RATE} Hz")
        positive("hop_ms", self.hop_ms)
        positive("frame_ms", self.frame_ms)
        positive("hop_length", self.hop_length, allow_none=True)
        positive("frame_length", self.frame_length, allow_none=True)
        positive("fft_size", self.fft_size, allow_none=True)
        for name in ("hop_length", "frame_length", "fft_size"):
            value = getattr(self, name)
            if value is not None and int(value) != value:
                raise ValueError(f"{name} must be an integer")
        if self.fft_size is not None and self.fft_size < 16:
            raise ValueError("fft_size must be at least 16")
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {self.window!r}")
        positive("f0_floor", self.f0_floor)
        positive("f0_ceil", self.f0_ceil)
        if self.f0_ceil <= self.f0_floor:
            raise ValueError("f0_ceil must be greater than f0_floor")
        for name in ("yin_threshold", "voicing_threshold", "voicing_extension_threshold"):
            value = getattr(self, name)
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("silence_threshold_db", "absolute_silence_db", "energy_floor_db"):
            value = getattr(self, name)
            if not math.isfinite(value) or value > 0:
                raise ValueError(f"{name} must be a finite level <= 0 dB")
        if self.f0_median_frames < 1 or int(self.f0_median_frames) != self.f0_median_frames:
            raise ValueError("f0_median_frames must be a positive integer (1 disables)")
        if self.voicing_extension_threshold < self.voicing_threshold:
            raise ValueError("voicing_extension_threshold must be >= voicing_threshold")
        if not (0.0 <= self.f0_max_jump <= 2.0):
            raise ValueError("f0_max_jump must be within [0, 2] (0 disables extension)")
        if not (0.0 <= self.f0_jump_cost <= 100.0):
            raise ValueError("f0_jump_cost must be within [0, 100] (0 disables tracking)")
        if self.min_voiced_ms < 0:
            raise ValueError("min_voiced_ms must be non-negative")
        if self.f0_refine_periods != 0 and not (1.0 <= self.f0_refine_periods <= 8.0):
            raise ValueError("f0_refine_periods must be 0 (off) or within [1, 8]")
        if self.envelope_method not in ENVELOPE_METHODS:
            raise ValueError(f"envelope_method must be one of {ENVELOPE_METHODS}")
        if not (-0.5 <= self.cheaptrick_q1 <= 0.5):
            raise ValueError("cheaptrick_q1 must be within [-0.5, 0.5]")
        if self.mcep_order is not None and (int(self.mcep_order) != self.mcep_order
                                            or self.mcep_order < 1):
            raise ValueError("mcep_order must be a positive integer")
        if self.mcep_alpha is not None and not (-1.0 < self.mcep_alpha < 1.0):
            raise ValueError("mcep_alpha must lie strictly inside (-1, 1)")
        if self.aperiodicity_bands:
            edges = self.aperiodicity_bands
            if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])) \
                    or edges[0] < 0:
                raise ValueError("aperiodicity_bands must be >= 2 increasing, "
                                 "non-negative band edges in Hz")
        elif self.n_aperiodicity_bands < 1:
            raise ValueError("n_aperiodicity_bands must be >= 1")
        if not (1.5 <= self.aperiodicity_periods <= 16.0):
            raise ValueError("aperiodicity_periods must be within [1.5, 16]")
        lo, hi = self.tilt_range_hz if len(self.tilt_range_hz) == 2 else (0.0, -1.0)
        if len(self.tilt_range_hz) != 2 or not (0.0 < lo < hi):
            raise ValueError("tilt_range_hz must be (low, high) with 0 < low < high")
        if not (0.0 < self.rolloff_percent < 1.0):
            raise ValueError("rolloff_percent must be within (0, 1)")
        if not (0.0 < self.power_floor < 1.0):
            raise ValueError("power_floor must be within (0, 1)")
        if self.block_frames < 1:
            raise ValueError("block_frames must be positive")

    # -- resolution -------------------------------------------------------

    def resolve(self, sample_rate: int) -> AnalysisGeometry:
        """Resolve sample-rate dependent settings for ``sample_rate`` Hz."""
        from hmvc.features.cepstrum import mcep_alpha as _alpha

        fs = int(sample_rate)
        if fs <= 0:
            raise ValueError("sample rate must be positive")
        nyquist = fs / 2.0
        if self.f0_ceil >= nyquist / 2.0:
            raise ValueError(f"f0_ceil={self.f0_ceil} Hz is too high for a sample rate "
                             f"of {fs} Hz (must be below fs/4)")
        hop = int(self.hop_length) if self.hop_length else int(round(fs * self.hop_ms / 1000.0))
        frame = int(self.frame_length) if self.frame_length \
            else int(round(fs * self.frame_ms / 1000.0))
        hop = max(1, hop)
        frame = max(2, frame)
        # The CheapTrick window spans three periods of the lowest F0; WORLD's
        # GetFFTSizeForCheapTrick convention guarantees it fits.
        cheaptrick = 1 << (1 + int(math.floor(math.log2(3.0 * fs / self.f0_floor + 1.0))))
        auto_fft = max(next_power_of_two(frame), cheaptrick)
        if self.fft_size is not None:
            fft = int(self.fft_size)
            if fft < frame:
                raise ValueError(f"fft_size={fft} is shorter than the analysis frame "
                                 f"({frame} samples)")
            if fft < 3.0 * fs / self.f0_floor + 3:
                raise ValueError(f"fft_size={fft} cannot hold a 3-period window at "
                                 f"f0_floor={self.f0_floor} Hz; use >= {cheaptrick}")
        else:
            fft = auto_fft
        order = int(self.mcep_order) if self.mcep_order is not None else default_mcep_order(fs)
        if order >= fft // 2:
            raise ValueError(f"mcep_order={order} must be below fft_size/2={fft // 2}")
        alpha = float(self.mcep_alpha) if self.mcep_alpha is not None else _alpha(fs)
        return AnalysisGeometry(sample_rate=fs, hop_length=hop, frame_length=frame,
                                fft_size=fft, mcep_order=order, mcep_alpha=alpha,
                                f0_floor=float(self.f0_floor), f0_ceil=float(self.f0_ceil))

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["aperiodicity_bands"] = list(self.aperiodicity_bands)
        data["tilt_range_hz"] = list(self.tilt_range_hz)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown analysis option(s): {sorted(unknown)}")
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "AnalysisConfig":
        return cls.from_dict(json.loads(text))

    def replace(self, **changes) -> "AnalysisConfig":
        return dataclasses.replace(self, **changes)
