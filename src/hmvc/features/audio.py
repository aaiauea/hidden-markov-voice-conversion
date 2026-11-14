"""Audio input for feature extraction: WAV I/O, resampling and sanitisation.

The engine's :mod:`hmvc.engine.data.wavio` is kept byte-for-byte as extracted
from HMS (its tests pin its behaviour, including reading IEEE-float files as
integer PCM).  Feature extraction needs a stricter reader, so this module
parses RIFF/WAVE itself:

* PCM 8 (unsigned), 16, 24 and 32 bit; IEEE float 32 and 64 bit;
* ``WAVE_FORMAT_EXTENSIBLE`` headers (the sub-format GUID decides PCM/float);
* odd-sized chunks, unknown chunks, and truncated ``data`` chunks (the
  available whole sample frames are returned instead of failing).

Everything is converted to float64; integer PCM is scaled to [-1, 1).

:func:`resample` is a band-limited (Kaiser-windowed sinc) rational-ratio
resampler written with NumPy only.  :func:`prepare_signal` turns arbitrary
caller input into a clean mono float64 signal and reports what it changed.
"""

from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

_FORMAT_PCM = 0x0001
_FORMAT_FLOAT = 0x0003
_FORMAT_EXTENSIBLE = 0xFFFE


@dataclass
class AudioData:
    """Decoded WAV payload."""

    samples: np.ndarray          # (n,) mono or (n, channels), float64
    sample_rate: int
    channels: int
    sample_format: str           # e.g. "pcm16", "float32"
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.samples.shape[0] / float(self.sample_rate)


def _read_chunks(handle: BinaryIO) -> Tuple[bytes, bytes, Dict[str, object]]:
    header = handle.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    fmt = None
    data = None
    info: Dict[str, object] = {"truncated": False}
    while True:
        chunk_header = handle.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id, size = chunk_header[:4], struct.unpack("<I", chunk_header[4:])[0]
        if chunk_id == b"data":
            payload = handle.read(size)
            if len(payload) < size:
                info["truncated"] = True
            data = payload
            if fmt is not None:
                break
        else:
            payload = handle.read(size)
            if chunk_id == b"fmt ":
                fmt = payload
        if size % 2:
            handle.read(1)  # RIFF chunks are word aligned
    if fmt is None:
        raise ValueError("WAV file has no 'fmt ' chunk")
    if data is None:
        raise ValueError("WAV file has no 'data' chunk")
    return fmt, data, info


def _decode(fmt: bytes, data: bytes) -> Tuple[np.ndarray, int, int, str]:
    if len(fmt) < 16:
        raise ValueError("malformed 'fmt ' chunk")
    tag, channels, rate, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _FORMAT_EXTENSIBLE:
        if len(fmt) < 40:
            raise ValueError("malformed WAVE_FORMAT_EXTENSIBLE header")
        tag = struct.unpack("<H", fmt[24:26])[0]
    if channels < 1 or rate < 1:
        raise ValueError("WAV header declares no channels or a zero sample rate")
    width = bits // 8
    if bits % 8 or width < 1 or block_align < width * channels:
        raise ValueError(f"unsupported WAV sample layout ({bits} bit, align {block_align})")
    usable = (len(data) // block_align) * block_align
    raw = np.frombuffer(data[:usable], dtype=np.uint8).reshape(-1, block_align)
    raw = raw[:, :width * channels].reshape(-1, channels, width)
    flat = np.ascontiguousarray(raw).reshape(-1)
    if tag == _FORMAT_PCM:
        if width == 1:
            values = (flat.astype(np.float64) - 128.0) / 128.0
        elif width == 2:
            values = flat.view("<i2").astype(np.float64) / 32768.0
        elif width == 3:
            b = flat.reshape(-1, 3).astype(np.int32)
            ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
            values = ints.astype(np.float64) / float(1 << 23)
        elif width == 4:
            values = flat.view("<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"unsupported PCM width: {bits} bit")
        name = f"pcm{bits}"
    elif tag == _FORMAT_FLOAT:
        if width == 4:
            values = flat.view("<f4").astype(np.float64)
        elif width == 8:
            values = flat.view("<f8").astype(np.float64)
        else:
            raise ValueError(f"unsupported float width: {bits} bit")
        name = f"float{bits}"
    else:
        raise ValueError(f"unsupported WAV format tag 0x{tag:04x}")
    return values.reshape(-1, channels), int(rate), int(channels), name


def read_wav(source: Union[PathLike, BinaryIO, bytes], mono: bool = True) -> AudioData:
    """Read a WAV file (path, binary file object, or bytes).

    ``mono=True`` averages channels; otherwise ``samples`` is ``(n, channels)``.
    """
    if isinstance(source, (bytes, bytearray)):
        handle: BinaryIO = io.BytesIO(source)
        fmt, data, info = _read_chunks(handle)
    elif hasattr(source, "read"):
        fmt, data, info = _read_chunks(source)  # type: ignore[arg-type]
    else:
        with open(Path(source), "rb") as handle:
            fmt, data, info = _read_chunks(handle)
    values, rate, channels, name = _decode(fmt, data)
    samples = values.mean(axis=1) if mono else values
    if channels == 1 and not mono:
        samples = values
    return AudioData(samples=np.ascontiguousarray(samples), sample_rate=rate,
                     channels=channels, sample_format=name, info=info)


def write_wav(path: PathLike, samples: np.ndarray, sample_rate: int,
              sample_format: str = "pcm16") -> None:
    """Write ``samples`` ((n,) or (n, channels), float in [-1, 1]) as WAV.

    ``sample_format`` is ``"pcm16"``, ``"pcm24"``, ``"float32"`` or
    ``"float64"``.  PCM output is clipped (not normalised) to full scale.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("samples must be 1-D or (n, channels)")
    channels = x.shape[1]
    if sample_format == "pcm16":
        tag, bits = _FORMAT_PCM, 16
        payload = np.round(np.clip(x, -1.0, 32767 / 32768) * 32768.0).astype("<i2").tobytes()
    elif sample_format == "pcm24":
        tag, bits = _FORMAT_PCM, 24
        ints = np.round(np.clip(x, -1.0, (2 ** 23 - 1) / 2 ** 23) * 2 ** 23).astype(np.int32)
        b = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        payload = np.ascontiguousarray(b).tobytes()
    elif sample_format == "float32":
        tag, bits = _FORMAT_FLOAT, 32
        payload = x.astype("<f4").tobytes()
    elif sample_format == "float64":
        tag, bits = _FORMAT_FLOAT, 64
        payload = x.astype("<f8").tobytes()
    else:
        raise ValueError(f"unsupported sample_format {sample_format!r}")
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, int(sample_rate),
                      int(sample_rate) * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        body += b"\x00"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

#: Zero crossings of the sinc kernel on each side (quality/speed trade-off).
RESAMPLE_ZERO_CROSSINGS = 16
#: Kaiser window shape parameter (~ -90 dB stop band side lobes).
RESAMPLE_KAISER_BETA = 8.6
#: Pass-band edge as a fraction of the lower Nyquist frequency.
RESAMPLE_ROLLOFF = 0.94


def _kernel(offsets: np.ndarray, cutoff: float, half_width: float) -> np.ndarray:
    """Windowed-sinc kernel evaluated at ``offsets`` (input-sample units)."""
    ratio = np.clip(offsets / half_width, -1.0, 1.0)
    window = np.i0(RESAMPLE_KAISER_BETA * np.sqrt(1.0 - ratio ** 2)) \
        / np.i0(RESAMPLE_KAISER_BETA)
    window = np.where(np.abs(offsets) <= half_width, window, 0.0)
    return cutoff * np.sinc(cutoff * offsets) * window


def resample(x: np.ndarray, orig_sr: int, target_sr: int,
             block: int = 16384) -> np.ndarray:
    """Band-limited resampling of a 1-D signal from ``orig_sr`` to ``target_sr``.

    Output sample ``n`` is evaluated at input time ``n * orig_sr / target_sr``
    with a Kaiser-windowed sinc whose cut-off is ``RESAMPLE_ROLLOFF`` times the
    lower of the two Nyquist frequencies (anti-aliasing when downsampling).
    Samples outside the signal are treated as zeros.  The output has
    ``ceil(len(x) * target_sr / orig_sr)`` samples.  For rational ratios with a
    small numerator the kernel is tabulated once per phase (polyphase).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    orig_sr, target_sr = int(orig_sr), int(target_sr)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sr == target_sr or x.size == 0:
        return x.copy()
    g = math.gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    n_out = -(-x.size * up // down)
    cutoff = min(1.0, target_sr / orig_sr) * RESAMPLE_ROLLOFF
    half_width = RESAMPLE_ZERO_CROSSINGS / cutoff
    taps = int(math.ceil(half_width))
    span = np.arange(-taps + 1, taps + 1)                  # 2 * taps offsets
    padded = np.concatenate([np.zeros(taps), x, np.zeros(taps + 1)])
    bank = None
    if up <= 4096:
        phases = np.arange(up) / up                         # fractional positions
        bank = _kernel(phases[:, None] - span[None, :], cutoff, half_width)
    out = np.empty(n_out, dtype=np.float64)
    for start in range(0, n_out, block):
        n = np.arange(start, min(start + block, n_out))
        num = n * down
        base = num // up                                    # integer input index
        phase = num - base * up
        idx = base[:, None] + span[None, :] + taps
        if bank is not None:
            weights = bank[phase]
        else:
            frac = phase / up
            weights = _kernel(frac[:, None] - span[None, :], cutoff, half_width)
        out[n] = np.einsum("ij,ij->i", padded[idx], weights)
    return out


# --------------------------------------------------------------------------
# Signal preparation
# --------------------------------------------------------------------------


def prepare_signal(signal: np.ndarray, sample_rate: int,
                   target_rate: int | None = None,
                   remove_dc: bool = True) -> Tuple[np.ndarray, int, Dict[str, object]]:
    """Coerce arbitrary input into a finite mono float64 signal.

    * integer arrays are scaled by their dtype's full scale;
    * ``(n, channels)`` input is averaged to mono (at most 64 channels);
    * NaN/inf samples are replaced by 0 (counted in the report);
    * the DC offset is removed when ``remove_dc`` is set;
    * the signal is resampled when ``target_rate`` differs from the input rate.

    Returns ``(signal, rate, report)``.  Never raises for unusual *content*;
    it raises ``ValueError`` only for malformed arguments.
    """
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    x = np.asarray(signal)
    report: Dict[str, object] = {"input_sample_rate": sample_rate, "input_dtype": str(x.dtype)}
    if x.dtype.kind in "iu":
        info = np.iinfo(x.dtype)
        scale = float(max(abs(info.min), info.max + 1))
        offset = 0.0 if x.dtype.kind == "i" else scale / 2.0
        x = (x.astype(np.float64) - offset) / (scale if x.dtype.kind == "i" else offset)
    elif x.dtype.kind == "b":
        x = x.astype(np.float64)
    elif x.dtype.kind not in "f":
        raise ValueError(f"unsupported audio dtype {x.dtype}")
    x = x.astype(np.float64, copy=False)
    if x.ndim == 2:
        if x.shape[1] > 64:
            raise ValueError(f"2-D audio must be (n_samples, n_channels); got shape {x.shape}")
        report["input_channels"] = int(x.shape[1])
        x = x.mean(axis=1) if x.shape[1] else np.zeros(x.shape[0])
    elif x.ndim > 2:
        raise ValueError("audio must be 1-D or (n_samples, n_channels)")
    x = x.reshape(-1)
    bad = ~np.isfinite(x)
    report["nonfinite_samples"] = int(bad.sum())
    if bad.any():
        x = np.where(bad, 0.0, x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    report["peak"] = peak
    report["clipped_fraction"] = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    if remove_dc and x.size:
        dc = float(np.mean(x))
        report["dc_offset"] = dc
        x = x - dc
    else:
        x = x.copy()
    rate = sample_rate
    if target_rate is not None and int(target_rate) != sample_rate:
        x = resample(x, sample_rate, int(target_rate))
        rate = int(target_rate)
        report["resampled_to"] = rate
    report["n_samples"] = int(x.size)
    return x, rate, report
