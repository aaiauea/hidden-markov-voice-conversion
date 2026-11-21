"""Constants of the original voice converter (see ``voice-converter (4).html``).

All values are reproduced *exactly* as the browser code defines them.  They
are plain floats/ints; only the array buffers carry a dtype (see
:mod:`hmvc.conversion.envelope` and :mod:`hmvc.conversion.fft`).
"""

#: Internal sample rate used by the whole pipeline.
SR = 22050
#: FFT / window size for spectral analysis and synthesis.
N = 1024
#: Hann window definition used everywhere (0.5 - 0.5 * cos(2 pi i / N)).
#: ``H`` is the analysis hop in samples between features frames.
H = 256
#: Number of spectral bins (N/2 + 1).
NB = N // 2 + 1
#: Hop size for the pitch tracker, in samples.
HOP = 320
#: Number of DCT features (mel-cepstra) in the retrieval/GMM space.
D = 12
#: Number of mel bands feeding the DCT.
NM = 34
#: Number of cepstral coefficients kept (lifter order).
LIF = 36
#: RMS floor for the silence gate.
SIL = 0.004
#: Frame hop used by the target-voice retrieval index.
DB_HOP = 512
#: RMS normaliser of the Hann window (mean of win^2 ~= 3/8 -> sqrt(3/8)).
WINDOW_NORM = 0.375
#: Envelope-path overlap-add normalization.
ENVELOPE_NORM = 1.5
#: Harmonic overlap-add normalization.
HARMONIC_NORM = 1.2247
#: Grain-phase harmonic synthesis constant (2 / (N * N * 0.375)).
STFT_SCALE = 2.0 / (N * N * WINDOW_NORM)
#: Length of the spectral envelope in the grain/vocoder synthesis path.
NB_BINS = NB
#: Number of output harmonics (NH).
NH = 220
#: Grain phase random walk (.6180339 = 1/phi; the pseudo-random PH phases).
PHASE_FILL_RATE = 0.6
PHASE_RETURN_COEFF = 0.6180339
#: Unvoiced grain duration.
UNVOICED_GRAIN = 110
#: Default time constant in seconds for the grain's temporal gain EMA.
GAIN_EMA = 0.3
#: Minimum frames required in the retrieval database before it is trusted.
DB_MIN_FRAMES = 30
#: Maximum grain amplitude clamp [0.25, 4].
GAIN_MIN = 0.25
GAIN_MAX = 4.0
#: Smoothing constants for the grain-retrieval gain normalisation.
GAIN_NORMALISER = 0.35  # maximum(ws, 0.35)
#: Peak level the output is normalised to.
PEAK_TARGET = 0.92
#: Output 16-bit scale factor (clamped to [-1, 1]).
WAV_SCALE = 32767.0
#: Target audio is truncated to 150 s in the UI.
TARGET_MAX_SECONDS = 150
