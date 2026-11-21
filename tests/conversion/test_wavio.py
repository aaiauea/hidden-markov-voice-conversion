"""WAV encoding (PCM16 @ 22050) — bytes and round trip."""
from __future__ import annotations

import struct

import numpy as np

from hmvc.conversion import encode_wav, write_wav


def _decode_wav(data: bytes):
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert data[12:16] == b"fmt "
    assert data[20:22] == struct.pack("<H", 1)        # PCM
    assert data[22:24] == struct.pack("<H", 1)        # mono
    sr = struct.unpack("<I", data[24:28])[0]
    assert data[28:32] == struct.pack("<I", sr * 2)   # byte rate
    assert struct.unpack("<H", data[32:34])[0] == 2   # block align
    assert struct.unpack("<H", data[34:36])[0] == 16  # bits
    assert data[36:40] == b"data"
    data_len = struct.unpack("<I", data[40:44])[0]
    assert data_len == len(data) - 44
    samples = np.frombuffer(data[44:], dtype=np.int16)
    assert len(samples) == data_len // 2
    return sr, samples


def test_encode_wav_header_and_data_length():
    samples = np.zeros(100, dtype=np.float32)
    data = encode_wav(samples)
    sr, decoded = _decode_wav(data)
    assert sr == 22050
    assert len(decoded) == 100


def test_encode_wav_clamps_and_scales():
    samples = np.array([1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float32)
    data = encode_wav(samples)
    _, decoded = _decode_wav(data)
    # int16(1.0 * 32767) = 32767; int16(-1.0 * 32767) = -32767; 2.0 -> clamped 32767
    assert decoded[0] == 32767
    assert decoded[1] == -32767
    assert decoded[2] == 32767
    assert decoded[3] == -32767
    assert decoded[4] == int(0.5 * 32767)


def test_write_wav_round_trip(tmp_path):
    samples = np.linspace(-0.9, 0.9, 500, dtype=np.float32)
    path = write_wav(tmp_path / "out.wav", samples)
    assert path.exists()
    sr, decoded = _decode_wav(path.read_bytes())
    assert sr == 22050
    assert len(decoded) == 500
