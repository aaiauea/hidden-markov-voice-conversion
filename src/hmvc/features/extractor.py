"""Feature extraction pipeline and the :class:`AcousticFeatures` container.

    audio --prepare_signal--> mono float64 @ fs
          |
          +--> YIN F0 / voicing / periodicity ----------------------+
          |                                                        |
          +--> fixed frames --> RMS energy, ZCR                    |
          |          `--> STFT power --> spectral descriptors      |
          |                                                        v
          +--> CheapTrick (F0-adaptive) power envelope --> mel-cepstrum (MCEP)
          |                                                        |
          +--> F0-adaptive band aperiodicity <---------------------+

All streams share one frame grid (frame ``t`` centred on sample
``t * hop_length``; ``1 + n // hop_length`` frames) and are stored as separate
arrays.  Nothing is concatenated unless :meth:`AcousticFeatures.stack` is
called explicitly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from hmvc.features import aperiodicity as _ap
from hmvc.features import envelope as _env
from hmvc.features.audio import prepare_signal, read_wav
from hmvc.features.cepstrum import mc2sp, sp2mc
from hmvc.features.config import AnalysisConfig, AnalysisGeometry
from hmvc.features.descriptors import (DESCRIPTORS, frame_rms, rms_to_db,
                                       spectral_descriptors, zero_crossing_rate)
from hmvc.features.f0 import continuous_log_f0, estimate_f0
from hmvc.features.framing import (blocks, frame_count, frame_signal, frame_times,
                                   get_window, power_spectrum)

FORMAT = "hmvc.features.acoustic"
FORMAT_VERSION = 1

_ARRAY_FIELDS = ("f0", "voiced", "periodicity", "continuous_log_f0", "energy_rms",
                 "energy_db", "speech", "mcep", "band_aperiodicity", "aperiodicity_edges")


@dataclass
class AcousticFeatures:
    """Frame-synchronous acoustic features of one utterance.

    Streams (``T`` frames):

    ==================== ============ =============================================
    ``f0``               (T,)         Hz, 0.0 where unvoiced
    ``voiced``           (T,) bool    voicing decision
    ``periodicity``      (T,)         1 - YIN aperiodicity, in [0, 1]
    ``log_f0``           (T,)         ln F0, NaN where unvoiced (property)
    ``continuous_log_f0`` (T,)        ln F0 interpolated through unvoiced frames
    ``energy_rms``       (T,)         frame RMS (full scale = 1)
    ``energy_db``        (T,)         20 log10 RMS, floored at ``energy_floor_db``
    ``speech``           (T,) bool    level above the silence gate
    ``mcep``             (T, M+1)     mel-cepstrum c~0..c~M of the CheapTrick
                                      power envelope (SPTK convention, alpha
                                      ``mcep_alpha``)
    ``band_aperiodicity`` (T, B)      noise power fraction per band, in [0, 1]
    ``spectral``         dict of (T,) centroid, bandwidth, rolloff, flatness,
                                      tilt, zcr, flux
    ``spectral_envelope`` (T, N/2+1)  linear power envelope (only when
                                      ``keep_envelope``; see :meth:`envelope`)
    ==================== ============ =============================================
    """

    sample_rate: int
    hop_length: int
    frame_length: int
    fft_size: int
    mcep_alpha: float
    n_samples: int
    config: dict
    f0: np.ndarray
    voiced: np.ndarray
    periodicity: np.ndarray
    continuous_log_f0: np.ndarray
    energy_rms: np.ndarray
    energy_db: np.ndarray
    speech: np.ndarray
    mcep: np.ndarray
    band_aperiodicity: np.ndarray
    aperiodicity_edges: np.ndarray
    spectral: Dict[str, np.ndarray]
    spectral_envelope: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)

    # -- geometry ---------------------------------------------------------

    def __len__(self) -> int:
        return int(self.f0.shape[0])

    @property
    def n_frames(self) -> int:
        return len(self)

    @property
    def times(self) -> np.ndarray:
        return frame_times(len(self), self.hop_length, self.sample_rate)

    @property
    def frame_period_ms(self) -> float:
        return 1000.0 * self.hop_length / self.sample_rate

    @property
    def duration(self) -> float:
        return self.n_samples / float(self.sample_rate)

    @property
    def mcep_order(self) -> int:
        return int(self.mcep.shape[1]) - 1

    @property
    def log_f0(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(self.voiced & (self.f0 > 0), np.log(np.maximum(self.f0, 1e-300)),
                            np.nan)

    # -- access -------------------------------------------------------------

    def stream(self, name: str) -> np.ndarray:
        """Any stream by name; spectral descriptors as ``"centroid"`` or
        ``"spectral_centroid"``.  Returns ``(T,)`` or ``(T, D)``."""
        key = name[len("spectral_"):] if name.startswith("spectral_") \
            and name != "spectral_envelope" else name
        if key in self.spectral:
            return self.spectral[key]
        if name == "log_f0":
            return self.log_f0
        if name == "spectral_envelope":
            return self.envelope()
        if name in _ARRAY_FIELDS:
            return getattr(self, name)
        raise KeyError(f"unknown feature stream {name!r}")

    def stack(self, names: Sequence[str]) -> np.ndarray:
        """Concatenate selected streams into one ``(T, D)`` float matrix."""
        parts = []
        for name in names:
            values = np.asarray(self.stream(name), dtype=np.float64)
            dim = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
            parts.append(values.reshape(len(self), dim))
        if not parts:
            return np.zeros((len(self), 0))
        return np.concatenate(parts, axis=1)

    def envelope(self) -> np.ndarray:
        """Power envelope: the kept full-resolution one, or rebuilt from MCEP."""
        if self.spectral_envelope is not None:
            return self.spectral_envelope
        return mc2sp(self.mcep, self.mcep_alpha, self.fft_size)

    # -- validation / persistence -------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if shapes or values are inconsistent."""
        t = len(self)
        for name in _ARRAY_FIELDS[:-1]:
            arr = getattr(self, name)
            if arr.shape[0] != t:
                raise ValueError(f"{name} has {arr.shape[0]} frames, expected {t}")
            if arr.dtype != bool and not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        for name, arr in self.spectral.items():
            if arr.shape != (t,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"spectral descriptor {name} is malformed")
        if np.any(self.f0[~self.voiced] != 0) or np.any(self.f0[self.voiced] <= 0):
            raise ValueError("f0 and voiced disagree")
        if np.any((self.band_aperiodicity < 0) | (self.band_aperiodicity > 1)):
            raise ValueError("band aperiodicity outside [0, 1]")
        if t != frame_count(self.n_samples, self.hop_length):
            raise ValueError("frame count does not match the signal length")

    def _meta(self) -> dict:
        return {"format": FORMAT, "version": FORMAT_VERSION, "sample_rate": self.sample_rate,
                "hop_length": self.hop_length, "frame_length": self.frame_length,
                "fft_size": self.fft_size, "mcep_alpha": self.mcep_alpha,
                "n_samples": self.n_samples, "config": self.config,
                "spectral": sorted(self.spectral), "diagnostics": self.diagnostics}

    def save(self, path) -> None:
        """Save to a compressed ``.npz`` (arrays + JSON metadata, no pickle)."""
        arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        arrays.update({f"spectral_{k}": v for k, v in self.spectral.items()})
        if self.spectral_envelope is not None:
            arrays["spectral_envelope"] = self.spectral_envelope
        arrays["meta"] = np.array(json.dumps(self._meta(), default=float))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "AcousticFeatures":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError("not an HMVC acoustic-feature file")
            if meta.get("version") != FORMAT_VERSION:
                raise ValueError(f"unsupported feature file version {meta.get('version')}")
            arrays = {name: np.array(data[name]) for name in _ARRAY_FIELDS}
            spectral = {k: np.array(data[f"spectral_{k}"]) for k in meta["spectral"]}
            envelope = np.array(data["spectral_envelope"]) if "spectral_envelope" in data else None
        return cls(sample_rate=meta["sample_rate"], hop_length=meta["hop_length"],
                   frame_length=meta["frame_length"], fft_size=meta["fft_size"],
                   mcep_alpha=meta["mcep_alpha"], n_samples=meta["n_samples"],
                   config=meta["config"], spectral=spectral, spectral_envelope=envelope,
                   diagnostics=meta.get("diagnostics", {}), **arrays)


class FeatureExtractor:
    """Classical acoustic feature extractor (NumPy only)."""

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config if config is not None else AnalysisConfig()

    def geometry(self, sample_rate: int) -> AnalysisGeometry:
        return self.config.resolve(sample_rate)

    def aperiodicity_edges(self, sample_rate: int) -> np.ndarray:
        cfg = self.config
        if cfg.aperiodicity_bands:
            edges = np.asarray(cfg.aperiodicity_bands, dtype=np.float64)
            if edges[-1] > sample_rate / 2.0 + 1e-6:
                raise ValueError(f"aperiodicity band edge {edges[-1]} Hz exceeds the "
                                 f"Nyquist frequency of {sample_rate} Hz audio")
            return edges
        return _ap.band_edges(cfg.n_aperiodicity_bands, sample_rate)

    def extract_file(self, path) -> AcousticFeatures:
        audio = read_wav(path, mono=True)
        features = self.extract(audio.samples, audio.sample_rate)
        features.diagnostics["source"] = {"path": str(path), "format": audio.sample_format,
                                          "channels": audio.channels,
                                          "truncated": bool(audio.info.get("truncated"))}
        return features

    def extract(self, signal: np.ndarray, sample_rate: int) -> AcousticFeatures:
        cfg = self.config
        x, fs, report = prepare_signal(signal, sample_rate, cfg.sample_rate, cfg.remove_dc)
        geom = cfg.resolve(fs)
        hop, n_fft = geom.hop_length, geom.fft_size
        edges = self.aperiodicity_edges(fs)
        n_frames = frame_count(x.size, hop)
        n_bins = n_fft // 2 + 1
        order = geom.mcep_order

        min_voiced = int(math.ceil(cfg.min_voiced_ms / geom.frame_period_ms)) \
            if cfg.min_voiced_ms > 0 else 0
        pitch = estimate_f0(x, fs, hop, cfg.f0_floor, cfg.f0_ceil, cfg.yin_threshold,
                            cfg.voicing_threshold, cfg.silence_threshold_db,
                            cfg.absolute_silence_db, cfg.f0_median_frames, min_voiced,
                            cfg.f0_refine_periods, cfg.voicing_extension_threshold,
                            cfg.f0_max_jump, cfg.f0_jump_cost, block=cfg.block_frames)

        energy = np.zeros(n_frames)
        zcr = np.zeros(n_frames)
        mcep = np.zeros((n_frames, order + 1))
        keep = np.zeros((n_frames, n_bins)) if cfg.keep_envelope else None
        spectral = {name: np.zeros(n_frames) for name in DESCRIPTORS}
        window = get_window(cfg.window, geom.frame_length)
        previous = None
        for start, stop in blocks(n_frames, cfg.block_frames):
            frames = frame_signal(x, geom.frame_length, hop, stop - start, start)
            energy[start:stop] = frame_rms(frames)
            zcr[start:stop] = zero_crossing_rate(frames)
            power = power_spectrum(frames, window, n_fft)
            desc, previous = spectral_descriptors(power, fs, cfg.rolloff_percent,
                                                  cfg.tilt_range_hz, cfg.power_floor, previous)
            for name, values in desc.items():
                spectral[name][start:stop] = values
            f0_block = pitch.f0[start:stop]
            if cfg.envelope_method == "cheaptrick":
                env = _env.cheaptrick_frames(x, fs, f0_block, start, hop, n_fft,
                                             cfg.cheaptrick_q1, cfg.power_floor)
            elif cfg.envelope_method == "true_envelope":
                env = _env.true_envelope(power, fs, f0_block, cfg.power_floor)
            else:
                env = _env.lifter_envelope(power, fs, f0_block, cfg.power_floor)
            env = np.maximum(env, cfg.power_floor)
            mcep[start:stop] = sp2mc(env, order, geom.mcep_alpha, floor=cfg.power_floor)
            if keep is not None:
                keep[start:stop] = env
        spectral["zcr"] = zcr

        band_ap = _ap.band_aperiodicity(x, fs, pitch.f0, hop, edges, cfg.aperiodicity_periods,
                                        block=cfg.block_frames)
        energy_db = rms_to_db(energy, cfg.energy_floor_db)
        if n_frames:
            gate = max(cfg.absolute_silence_db, float(energy_db.max()) + cfg.silence_threshold_db)
            speech = energy_db > gate
        else:
            speech = np.zeros(0, dtype=bool)
        fill = math.sqrt(cfg.f0_floor * cfg.f0_ceil)
        report.update({"n_frames": int(n_frames), "voiced_frames": int(pitch.voiced.sum()),
                       "speech_frames": int(speech.sum()), "geometry": geom.to_dict()})
        features = AcousticFeatures(
            sample_rate=fs, hop_length=hop, frame_length=geom.frame_length, fft_size=n_fft,
            mcep_alpha=geom.mcep_alpha, n_samples=int(x.size), config=cfg.to_dict(),
            f0=pitch.f0, voiced=pitch.voiced, periodicity=pitch.periodicity,
            continuous_log_f0=continuous_log_f0(pitch.f0, pitch.voiced, fill),
            energy_rms=energy, energy_db=energy_db, speech=speech, mcep=mcep,
            band_aperiodicity=band_ap, aperiodicity_edges=edges, spectral=spectral,
            spectral_envelope=keep, diagnostics=report)
        return features


def extract(signal: np.ndarray, sample_rate: int,
            config: Optional[AnalysisConfig] = None) -> AcousticFeatures:
    """Convenience wrapper: ``FeatureExtractor(config).extract(signal, sample_rate)``."""
    return FeatureExtractor(config).extract(signal, sample_rate)


def extract_file(path, config: Optional[AnalysisConfig] = None) -> AcousticFeatures:
    return FeatureExtractor(config).extract_file(path)
