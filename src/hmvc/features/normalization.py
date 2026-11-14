"""Consistent feature normalisation.

:class:`FeatureNormalizer` accumulates per-stream mean / variance over any
number of utterances (one pass, numerically stable pairwise merging of
moments -- Chan, Golub and LeVeque) and applies z-score normalisation.

Frame selection is part of the statistics' definition, so it is fixed per
stream rather than left to callers:

* ``log_f0`` and ``continuous_log_f0`` use **voiced** frames only (statistics
  of interpolated or undefined pitch would be meaningless);
* every other stream uses **speech** frames (level above the silence gate)
  when ``exclude_silence`` is set (default), else all frames.

Unvoiced ``log_f0`` values stay NaN after normalisation.  Dimensions with
(near) zero variance are scaled by ``std_floor`` instead of dividing by zero.
Statistics serialise to JSON (no pickle) together with the analysis
configuration they were computed with, so they can be checked for
compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

DEFAULT_STREAMS = ("mcep", "continuous_log_f0", "energy_db", "band_aperiodicity",
                   "periodicity")
VOICED_STREAMS = ("log_f0", "continuous_log_f0")


def _as_rows(values) -> np.ndarray:
    """``(N,)`` or ``(N, ...)`` -> float ``(N, D)``; also for ``N == 0``."""
    values = np.asarray(values, dtype=np.float64)
    dim = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
    return values.reshape(values.shape[0], dim)


@dataclass
class RunningMoments:
    """Count, mean and sum of squared deviations of a ``(N, D)`` stream."""

    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, dim: int) -> "RunningMoments":
        return cls(0, np.zeros(dim), np.zeros(dim))

    @classmethod
    def from_values(cls, values: np.ndarray) -> "RunningMoments":
        values = _as_rows(values)
        if values.shape[0] == 0:
            return cls.empty(values.shape[1])
        mean = values.mean(axis=0)
        return cls(values.shape[0], mean, np.sum((values - mean) ** 2, axis=0))

    def merge(self, other: "RunningMoments") -> "RunningMoments":
        if other.count == 0:
            return RunningMoments(self.count, self.mean.copy(), self.m2.copy())
        if self.count == 0:
            return RunningMoments(other.count, other.mean.copy(), other.m2.copy())
        n = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.count / n)
        m2 = self.m2 + other.m2 + delta ** 2 * (self.count * other.count / n)
        return RunningMoments(n, mean, m2)

    @property
    def variance(self) -> np.ndarray:
        return self.m2 / self.count if self.count else np.full_like(self.mean, np.nan)


class FeatureNormalizer:
    """Per-stream z-score statistics over a corpus of :class:`AcousticFeatures`."""

    def __init__(self, streams: Sequence[str] = DEFAULT_STREAMS,
                 exclude_silence: bool = True, std_floor: float = 1e-8) -> None:
        self.streams = tuple(streams)
        self.exclude_silence = bool(exclude_silence)
        self.std_floor = float(std_floor)
        self.moments: Dict[str, RunningMoments] = {}
        self.config: Optional[dict] = None

    # -- fitting ------------------------------------------------------------

    def _frames(self, features, name: str) -> np.ndarray:
        values = _as_rows(features.stream(name))
        if name in VOICED_STREAMS:
            select = features.voiced
        elif self.exclude_silence:
            select = features.speech
        else:
            select = np.ones(len(features), dtype=bool)
        values = values[select]
        return values[np.all(np.isfinite(values), axis=1)]

    def partial_fit(self, features) -> "FeatureNormalizer":
        config = features.config
        if self.config is None:
            self.config = config
        elif config != self.config:
            raise ValueError("features were extracted with a different AnalysisConfig")
        for name in self.streams:
            values = self._frames(features, name)
            current = self.moments.get(name, RunningMoments.empty(values.shape[1]))
            if current.mean.shape[0] != values.shape[1]:
                raise ValueError(f"stream {name!r} changed dimension")
            self.moments[name] = current.merge(RunningMoments.from_values(values))
        return self

    def fit(self, corpus: Iterable) -> "FeatureNormalizer":
        for features in corpus:
            self.partial_fit(features)
        return self

    # -- statistics ---------------------------------------------------------

    def mean(self, name: str) -> np.ndarray:
        return self._moments(name).mean

    def std(self, name: str) -> np.ndarray:
        var = self._moments(name).variance
        return np.maximum(np.sqrt(np.maximum(var, 0.0)), self.std_floor)

    def _moments(self, name: str) -> RunningMoments:
        if name not in self.moments or self.moments[name].count == 0:
            raise KeyError(f"no statistics for stream {name!r}")
        return self.moments[name]

    # -- application ----------------------------------------------------------

    def normalize(self, name: str, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        flat = _as_rows(values)
        return ((flat - self.mean(name)) / self.std(name)).reshape(values.shape)

    def denormalize(self, name: str, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        flat = _as_rows(values)
        return (flat * self.std(name) + self.mean(name)).reshape(values.shape)

    def transform(self, features, streams: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Normalised copies of ``streams`` (default: every fitted stream)."""
        if self.config is not None and features.config != self.config:
            raise ValueError("features were extracted with a different AnalysisConfig")
        names = self.streams if streams is None else tuple(streams)
        return {name: self.normalize(name, features.stream(name)) for name in names}

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": "hmvc.features.normalizer",
            "version": 1,
            "streams": list(self.streams),
            "exclude_silence": self.exclude_silence,
            "std_floor": self.std_floor,
            "config": self.config,
            "moments": {name: {"count": m.count, "mean": m.mean.tolist(), "m2": m.m2.tolist()}
                        for name, m in self.moments.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureNormalizer":
        if data.get("format") != "hmvc.features.normalizer":
            raise ValueError("not an HMVC feature-normaliser file")
        obj = cls(data["streams"], data["exclude_silence"], data["std_floor"])
        obj.config = data.get("config")
        obj.moments = {name: RunningMoments(int(m["count"]), np.asarray(m["mean"], float),
                                            np.asarray(m["m2"], float))
                       for name, m in data["moments"].items()}
        return obj

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "FeatureNormalizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def normalize_utterance(features, streams: Sequence[str] = DEFAULT_STREAMS) -> Dict[str, np.ndarray]:
    """Per-utterance mean/variance normalisation (statistics from ``features`` only)."""
    return FeatureNormalizer(streams).partial_fit(features).transform(features)
