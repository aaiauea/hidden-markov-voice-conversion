"""Output WAV writer — a strict 1:1 port of the original ``wav()``.

16-bit PCM mono at 22050 Hz; clamp to [-1, 1], scale by 32767, truncate to
integer (matching the original's ``DataView.setInt16`` semantics).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from hmvc.conversion.constants import SR


def encode_wav(samples) -> bytes:
    """Encode ``samples`` as the original 16-bit PCM WAV bytes."""
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    n = samples.size
    b = bytearray(44 + n * 2)
    b[0:4] = b"RIFF"
    struct.pack_into("<I", b, 4, 36 + n * 2)
    b[8:12] = b"WAVE"
    b[12:16] = b"fmt "
    struct.pack_into("<I", b, 16, 16)
    struct.pack_into("<H", b, 20, 1)    # PCM
    struct.pack_into("<H", b, 22, 1)    # mono
    struct.pack_into("<I", b, 24, SR)
    struct.pack_into("<I", b, 28, SR * 2)
    struct.pack_into("<H", b, 32, 2)    # block align
    struct.pack_into("<H", b, 34, 16)   # bits
    b[36:40] = b"data"
    struct.pack_into("<I", b, 40, n * 2)
    for i in range(n):
        v = max(-1.0, min(1.0, float(samples[i]))) * 32767.0
        # DataView.setInt16 truncates toward zero, as int() does.
        struct.pack_into("<h", b, 44 + i * 2, int(v))
    return bytes(b)


def write_wav(path, samples) -> Path:
    """Encode ``samples`` as WAV and write it to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_wav(samples))
    return path


def load_audio(path):
    """Load a WAV file (any rate/channels) as 22050 Hz mono.

    Resampling uses the repo's existing band-limited resampler; WAV files
    only (the original browser accepts any decodable audio format — that
    capability is browser-specific and not ported).
    """
    from hmvc.features.audio import read_wav, resample

    wav = read_wav(path, mono=True)
    if wav.sample_rate != SR:
        return np.asarray(resample(wav.samples, wav.sample_rate, SR), dtype=np.float32)
    return np.asarray(wav.samples, dtype=np.float32)
