"""Framing, windows, power spectra, cepstral transforms and descriptors."""
from __future__ import annotations

import numpy as np
import pytest

from hmvc.features import cepstrum as C
from hmvc.features import descriptors as D
from hmvc.features.framing import (blocks, frame_count, frame_signal, frame_times,
                                   get_window, log_magnitude, power_spectrum)

RNG = np.random.default_rng(1234)


# -- framing --------------------------------------------------------------------


@pytest.mark.parametrize("n, hop, expected", [(0, 80, 0), (1, 80, 1), (79, 80, 1),
                                              (80, 80, 2), (16000, 80, 201)])
def test_frame_count_follows_world_convention(n, hop, expected):
    assert frame_count(n, hop) == expected


def test_frame_count_rejects_bad_hop():
    with pytest.raises(ValueError):
        frame_count(100, 0)


@pytest.mark.parametrize("length", [400, 401, 551])
def test_frames_are_centred_and_zero_padded(length):
    x = np.arange(1, 2001, dtype=float)
    hop = 80
    frames = frame_signal(x, length, hop)
    left = (length + 1) // 2
    assert frames.shape == (frame_count(x.size, hop), length)
    for t in (0, 3, 10, len(frames) - 1):
        centre = t * hop
        if centre < x.size:
            assert frames[t, left] == x[centre]
    assert np.all(frames[0, :left] == 0)                     # zero padding before t=0
    custom = frame_signal(x, length, hop, left=length // 2)
    assert custom[5, length // 2] == x[5 * hop]


def test_frame_ranges_match_full_framing():
    x = RNG.standard_normal(3000)
    full = frame_signal(x, 400, 80)
    for start, stop in blocks(len(full), 7):
        np.testing.assert_array_equal(frame_signal(x, 400, 80, stop - start, start),
                                      full[start:stop])
    assert frame_signal(x, 400, 80, 0).shape == (0, 400)
    assert frame_signal(np.zeros(0), 400, 80).shape == (0, 400)
    assert list(blocks(10, 4)) == [(0, 4), (4, 8), (8, 10)]
    np.testing.assert_allclose(frame_times(3, 80, 16000), [0.0, 0.005, 0.01])


def test_frames_are_independent_copies():
    x = np.ones(1000)
    frames = frame_signal(x, 200, 50)
    frames[1, :] = 5.0
    assert np.all(frames[2] <= 1.0) and np.all(x == 1.0)


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "rectangular"])
def test_windows_are_periodic(name):
    w = get_window(name, 64)
    longer = get_window(name, 65) if name == "rectangular" else None
    assert w.shape == (64,) and np.all(w >= -1e-12) and w.max() <= 1.0 + 1e-12
    if name != "rectangular":
        # DFT-even: w[n] == w[N - n] and the periodic extension is smooth
        np.testing.assert_allclose(w[1:], w[1:][::-1], atol=1e-12)
    else:
        assert np.all(longer == 1.0)
    with pytest.raises(ValueError):
        get_window("kaiser", 64)


def test_power_spectrum_is_a_frame_length_independent_density():
    sigma = 0.3
    x = sigma * RNG.standard_normal(200000)
    for length in (256, 400, 1024):
        frames = frame_signal(x, length, length)[1:-1]
        p = power_spectrum(frames, get_window("hann", length), 2048)
        assert np.mean(p[:, 5:-5]) == pytest.approx(sigma ** 2, rel=0.03)


def test_power_spectrum_locates_a_sine():
    fs, n_fft, f = 16000, 1024, 1000.0
    x = np.sin(2 * np.pi * f * np.arange(4000) / fs)
    p = power_spectrum(frame_signal(x, 400, 80)[10:20], get_window("hann", 400), n_fft)
    assert np.all(np.argmax(p, axis=1) == round(f * n_fft / fs))
    assert np.all(np.isfinite(log_magnitude(np.zeros((2, 5)))))


# -- cepstrum ---------------------------------------------------------------------


def _smooth_envelope(n_bins=513, seed=0):
    rng = np.random.default_rng(seed)
    cep = np.zeros(2 * (n_bins - 1))
    cep[0] = -2.0
    cep[1:20] = rng.standard_normal(19) * 0.6 / np.arange(1, 20)
    log_amp = C.cepstrum_to_log_amplitude(cep[None, :20], n_bins)[0]
    return np.exp(2.0 * log_amp)


def test_real_cepstrum_round_trip_is_exact():
    power = np.exp(RNG.standard_normal((3, 257)))
    cep = C.real_cepstrum(power)
    log_amp = C.cepstrum_to_log_amplitude(cep, 257)
    np.testing.assert_allclose(log_amp, 0.5 * np.log(power), atol=1e-10)


def test_cepstrum_of_flat_spectrum_is_an_impulse():
    cep = C.real_cepstrum(np.full((1, 129), np.e ** 2))
    assert cep[0, 0] == pytest.approx(1.0)                     # 0.5 * ln(e^2)
    np.testing.assert_allclose(cep[0, 1:], 0.0, atol=1e-12)


def test_lifter_kinds_and_per_frame_orders():
    cep = np.ones((2, 10))
    rect = C.lifter(cep, np.array([2, 5]))
    assert rect[0].tolist() == [1, 1, 1] + [0] * 7 and rect[1, :6].sum() == 6
    hann = C.lifter(cep, 4, kind="hann")
    assert hann[0, 0] == 1.0 and np.all(np.diff(hann[0, :5]) < 0) and np.all(hann[0, 5:] == 0)
    with pytest.raises(ValueError):
        C.lifter(cep, 3, kind="gaussian")


@pytest.mark.parametrize("alpha", [-0.42, 0.0, 0.31, 0.55])
def test_freqt_matrix_equals_recursion(alpha):
    c = RNG.standard_normal((4, 40)) / np.arange(1, 41)
    np.testing.assert_allclose(C.freqt(c, 30, alpha), C.freqt_recursive(c, 30, alpha),
                               atol=1e-13)


def test_freqt_identity_and_inverse():
    c = RNG.standard_normal((2, 25)) / np.arange(1, 26) ** 2
    np.testing.assert_allclose(C.freqt(c, 24, 0.0), c, atol=1e-15)
    warped = C.freqt(c, 200, 0.41)        # order 200: truncation error below the tolerance
    np.testing.assert_allclose(C.freqt(warped, 24, -0.41), c, atol=1e-9)
    np.testing.assert_allclose(C.mc2c(C.freqt(c, 200, 0.41), 0.41, 25), c, atol=1e-9)


@pytest.mark.parametrize("alpha", [0.0, 0.41, 0.55])
def test_sp2mc_mc2sp_inverts_a_representable_envelope(alpha):
    mc = np.zeros((2, 25))
    mc[:, 0] = [-3.0, 1.0]
    mc[:, 1:8] = np.random.default_rng(3).standard_normal((2, 7)) * 0.3
    sp = C.mc2sp(mc, alpha, 1024)
    np.testing.assert_allclose(C.sp2mc(sp, 24, alpha), mc, atol=1e-8)


def test_mcep_c0_is_mean_log_amplitude_and_roundtrip_improves_with_order():
    sp = _smooth_envelope()
    mc = C.sp2mc(sp[None], 24, 0.0)
    assert mc[0, 0] == pytest.approx(np.mean(0.5 * np.log(sp)), abs=1e-3)
    errs = [C.log_spectral_distance(C.mc2sp(C.sp2mc(sp[None], m, 0.41), 0.41, 1024),
                                    sp[None])[0] for m in (4, 12, 24, 48)]
    assert all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 0.05


def test_mcep_alpha_values_and_monotonicity():
    rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
    values = [C.mcep_alpha(fs) for fs in rates]
    assert values[2] == pytest.approx(0.41) and all(np.diff(values) > 0)


def test_distortion_measures():
    a = np.zeros((1, 25))
    b = a.copy()
    b[0, 1] = 0.1
    assert C.mel_cepstral_distortion(a, a)[0] == 0.0
    assert C.mel_cepstral_distortion(a, b)[0] == pytest.approx(10 / np.log(10) * np.sqrt(0.02))
    b0 = a.copy()
    b0[0, 0] = 1.0
    assert C.mel_cepstral_distortion(a, b0)[0] == 0.0
    assert C.mel_cepstral_distortion(a, b0, include_c0=True)[0] > 0
    p = np.ones((1, 10))
    assert C.log_spectral_distance(p * 10, p)[0] == pytest.approx(10.0)
    assert C.log_spectral_distance(p * 10, p, remove_gain=True)[0] == pytest.approx(0.0)
    w = C.mel_weights(513, 16000)
    assert w.shape == (513,) and np.all(np.diff(w) < 0)


# -- descriptors --------------------------------------------------------------------


def test_descriptors_of_a_sine_white_noise_and_silence():
    fs, n_fft = 16000, 1024
    win = get_window("hann", 400)
    sine = np.sin(2 * np.pi * 2000.0 * np.arange(8000) / fs)
    d, _ = D.spectral_descriptors(power_spectrum(frame_signal(sine, 400, 80)[5:-5], win,
                                                 n_fft), fs)
    assert np.allclose(d["centroid"], 2000.0, rtol=0.01)
    assert np.all(d["bandwidth"] < 100.0) and np.all(d["flatness"] < 1e-3)
    noise = RNG.standard_normal(80000)
    d, _ = D.spectral_descriptors(power_spectrum(frame_signal(noise, 400, 80)[5:-5], win,
                                                 n_fft), fs)
    assert np.median(d["centroid"]) == pytest.approx(4000.0, rel=0.05)
    assert np.median(d["rolloff"]) == pytest.approx(0.85 * 8000, rel=0.05)
    assert 0.4 < np.median(d["flatness"]) < 0.7                    # chi-square: e^-gamma
    assert abs(np.median(d["tilt"])) < 1.0
    d, _ = D.spectral_descriptors(np.zeros((3, 513)), fs)
    assert np.all(d["centroid"] == 0) and np.allclose(d["flatness"], 1.0)
    assert all(np.all(np.isfinite(v)) for v in d.values())


def test_spectral_tilt_measures_db_per_octave():
    fs = 16000
    f = np.linspace(0, fs / 2, 513)
    power = np.where(f > 0, (np.maximum(f, 1.0) / 1000.0) ** -2.0, 1.0)  # -6.02 dB/oct
    d, _ = D.spectral_descriptors(power[None], fs)
    assert d["tilt"][0] == pytest.approx(-6.02, abs=0.05)


def test_flux_is_continuous_across_blocks():
    x = RNG.standard_normal(16000)
    p = power_spectrum(frame_signal(x, 400, 80), get_window("hann", 400), 1024)
    full, _ = D.spectral_descriptors(p, 16000)
    first, prev = D.spectral_descriptors(p[:50], 16000)
    second, _ = D.spectral_descriptors(p[50:], 16000, previous_magnitude=prev)
    np.testing.assert_allclose(np.concatenate([first["flux"], second["flux"]]), full["flux"])
    assert full["flux"][0] == 0.0


def test_rms_zcr_and_db():
    frames = np.array([[1.0, -1.0, 1.0, -1.0], [0.5, 0.5, 0.5, 0.5], [0.0] * 4])
    np.testing.assert_allclose(D.frame_rms(frames), [1.0, 0.5, 0.0])
    np.testing.assert_allclose(D.zero_crossing_rate(frames), [0.75, 0.0, 0.0])
    np.testing.assert_allclose(D.rms_to_db(np.array([1.0, 0.1, 0.0])), [0.0, -20.0, -100.0])
