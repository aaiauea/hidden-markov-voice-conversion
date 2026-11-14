"""WAV I/O, resampling and input sanitising."""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from hmvc.features import prepare_signal, read_wav, resample, write_wav


def _wav_bytes(payload: bytes, tag: int, channels: int, rate: int, bits: int,
               extensible: bool = False, extra_chunks: bytes = b"",
               declared_data_size: int | None = None) -> bytes:
    align = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, channels, rate, rate * align, align, bits,
                          22, bits, 0) + struct.pack("<H", tag) + b"\x00" * 14
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * align, align, bits)
    size = len(payload) if declared_data_size is None else declared_data_size
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra_chunks \
        + b"data" + struct.pack("<I", size) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _tone(n=800, fs=8000, f=440.0, amp=0.5):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / fs)


@pytest.mark.parametrize("fmt, tol", [("pcm16", 1 / 32768), ("pcm24", 1 / 2 ** 23),
                                      ("float32", 1e-7), ("float64", 0.0)])
def test_write_read_round_trip(tmp_path, fmt, tol):
    x = _tone()
    path = tmp_path / f"t_{fmt}.wav"
    write_wav(path, x, 8000, fmt)
    audio = read_wav(path)
    assert audio.sample_rate == 8000 and audio.channels == 1 and audio.sample_format == fmt
    assert audio.samples.shape == x.shape
    assert np.max(np.abs(audio.samples - x)) <= tol + 1e-12
    assert audio.duration == pytest.approx(0.1)


def test_float32_wav_is_read_correctly():
    # the HMS engine's reader misinterprets IEEE-float WAV; ours must not
    x = np.array([0.0, 0.25, -0.5, 0.999, -1.0], dtype="<f4")
    audio = read_wav(_wav_bytes(x.tobytes(), 3, 1, 16000, 32))
    np.testing.assert_allclose(audio.samples, x.astype(np.float64))


def test_pcm8_and_pcm32_and_extensible():
    u8 = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    a = read_wav(_wav_bytes(u8.tobytes(), 1, 1, 8000, 8))
    np.testing.assert_allclose(a.samples, (u8.astype(float) - 128) / 128)
    i32 = np.array([0, 2 ** 30, -2 ** 31], dtype="<i4")
    b = read_wav(_wav_bytes(i32.tobytes(), 1, 1, 8000, 32))
    np.testing.assert_allclose(b.samples, [0.0, 0.5, -1.0])
    i16 = np.array([1000, -1000, 16384], dtype="<i2")
    c = read_wav(_wav_bytes(i16.tobytes(), 1, 1, 22050, 16, extensible=True))
    assert c.sample_format == "pcm16" and c.sample_rate == 22050
    np.testing.assert_allclose(c.samples, i16 / 32768.0)


def test_stereo_mixdown_and_multichannel_access(tmp_path):
    left, right = _tone(f=300.0), _tone(f=500.0, amp=0.25)
    path = tmp_path / "stereo.wav"
    write_wav(path, np.stack([left, right], axis=1), 8000, "float32")
    mono = read_wav(path)
    both = read_wav(path, mono=False)
    assert mono.channels == 2 and both.samples.shape == (800, 2)
    np.testing.assert_allclose(mono.samples, (left + right) / 2, atol=1e-7)


def test_unknown_chunks_odd_sizes_and_truncation():
    pcm = (np.arange(10, dtype="<i2") * 100).tobytes()
    junk = b"LIST" + struct.pack("<I", 3) + b"abc" + b"\x00"         # odd chunk + pad
    a = read_wav(_wav_bytes(pcm, 1, 1, 8000, 16, extra_chunks=junk))
    assert a.samples.size == 10 and not a.info["truncated"]
    b = read_wav(_wav_bytes(pcm, 1, 1, 8000, 16, declared_data_size=100))
    assert b.samples.size == 10 and b.info["truncated"]
    c = read_wav(_wav_bytes(pcm[:-1], 1, 1, 8000, 16))                # partial sample
    assert c.samples.size == 9


def test_file_object_and_empty_data():
    a = read_wav(io.BytesIO(_wav_bytes(b"", 1, 1, 8000, 16)))
    assert a.samples.size == 0


@pytest.mark.parametrize("blob", [b"", b"RIFX0000WAVE", b"RIFF\x04\x00\x00\x00WAVE",
                                  _wav_bytes(b"\x00\x00", 2, 1, 8000, 16),     # ADPCM tag
                                  _wav_bytes(b"\x00\x00", 1, 1, 8000, 12),     # 12-bit
                                  _wav_bytes(b"\x00\x00", 1, 0, 8000, 16)])
def test_malformed_files_raise_value_error(blob):
    with pytest.raises(ValueError):
        read_wav(blob)


def test_pcm_writing_clips_instead_of_wrapping(tmp_path):
    path = tmp_path / "clip.wav"
    write_wav(path, np.array([2.0, -3.0, 0.5]), 8000, "pcm16")
    np.testing.assert_allclose(read_wav(path).samples, [32767 / 32768, -1.0, 0.5])


# -- resampling ----------------------------------------------------------------


@pytest.mark.parametrize("src, dst", [(16000, 8000), (8000, 16000), (44100, 16000),
                                      (16000, 22050), (48000, 44100)])
def test_resampler_preserves_in_band_tones(src, dst):
    n = src // 2
    f = 1000.0
    y = resample(np.sin(2 * np.pi * f * np.arange(n) / src), src, dst)
    assert y.size == -(-n * dst // src)
    t = np.arange(y.size) / dst
    inner = slice(y.size // 8, -y.size // 8)                       # avoid edge transients
    err = y[inner] - np.sin(2 * np.pi * f * t[inner])
    assert np.sqrt(np.mean(err ** 2)) < 2e-3


def test_resampler_rejects_aliases_when_downsampling():
    src, dst = 48000, 16000
    t = np.arange(src) / src
    y = resample(np.sin(2 * np.pi * 12000.0 * t), src, dst)        # above the new Nyquist
    inner = y[2000:-2000]
    assert np.sqrt(np.mean(inner ** 2)) < 1e-3                     # > 57 dB rejection


def test_resampler_identity_empty_and_errors():
    x = np.random.default_rng(0).standard_normal(100)
    np.testing.assert_array_equal(resample(x, 16000, 16000), x)
    assert resample(np.zeros(0), 16000, 8000).size == 0
    with pytest.raises(ValueError):
        resample(x, 0, 16000)


def test_resampler_is_block_invariant():
    x = np.random.default_rng(1).standard_normal(5000)
    np.testing.assert_allclose(resample(x, 44100, 16000, block=97),
                               resample(x, 44100, 16000), atol=1e-12)


# -- input sanitising ----------------------------------------------------------


def test_prepare_signal_handles_dtypes_channels_nonfinite_and_dc():
    i16 = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    x, fs, rep = prepare_signal(i16, 8000, remove_dc=False)
    np.testing.assert_allclose(x, [0.0, 0.5, -1.0, 32767 / 32768])
    u8 = np.array([128, 255, 0], dtype=np.uint8)
    x, _, _ = prepare_signal(u8, 8000, remove_dc=False)
    np.testing.assert_allclose(x, [0.0, 127 / 128, -1.0])
    st = np.stack([np.ones(10), -np.ones(10) * 0.5], axis=1)
    x, _, rep = prepare_signal(st, 8000, remove_dc=False)
    assert rep["input_channels"] == 2 and np.allclose(x, 0.25)
    bad = np.array([0.1, np.nan, np.inf, -np.inf, 0.3])
    x, _, rep = prepare_signal(bad, 8000)
    assert rep["nonfinite_samples"] == 3 and np.all(np.isfinite(x))
    assert abs(x.mean()) < 1e-15 and rep["dc_offset"] == pytest.approx(0.08)


def test_prepare_signal_resamples_and_rejects_bad_arguments():
    x, fs, rep = prepare_signal(np.zeros(1600), 16000, target_rate=8000)
    assert fs == 8000 and x.size == 800 and rep["resampled_to"] == 8000
    for args in [(np.zeros(4), 0), (np.zeros((4, 2, 2)), 8000),
                 (np.zeros((2, 100)), 8000), (np.array(["a"]), 8000)]:
        with pytest.raises(ValueError):
            prepare_signal(*args)
