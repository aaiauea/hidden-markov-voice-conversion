"""HMVC Phase 1: classical acoustic feature extraction (NumPy only).

Quick start::

    from hmvc.features import AnalysisConfig, FeatureExtractor

    extractor = FeatureExtractor(AnalysisConfig(sample_rate=16000, hop_ms=5.0))
    feats = extractor.extract_file("input.wav")
    feats.mcep            # (T, M+1) mel-cepstrum (primary spectral representation)
    feats.f0, feats.voiced, feats.periodicity
    feats.energy_db, feats.band_aperiodicity, feats.spectral["centroid"]
    feats.envelope()      # (T, N/2+1) power envelope (kept, or approximated from MCEP)

Modules
-------
``config``          :class:`AnalysisConfig` / :class:`AnalysisGeometry`
``audio``           WAV reader/writer, band-limited resampler, input sanitising
``framing``         shared frame grid, windows, STFT power spectrum
``f0``              YIN F0, voicing decision, periodicity, continuous log-F0
``envelope``        CheapTrick (default), true-envelope and lifter envelopes
``cepstrum``        real cepstrum, liftering, freqt, sp2mc/mc2sp, distortions
``aperiodicity``    band aperiodicity (window-corrected band autocorrelation)
``descriptors``     energy, ZCR and spectral descriptors
``normalization``   corpus/utterance z-score statistics
``extractor``       :class:`FeatureExtractor`, :class:`AcousticFeatures`
``interop``         optional conversion to the HMS engine's frame sequence

No neural networks, pretrained models, transcripts or external WORLD/SPTK
libraries are used.  See ``docs/feature-extraction.md``.
"""

from hmvc.features.audio import AudioData, prepare_signal, read_wav, resample, write_wav
from hmvc.features.cepstrum import (log_spectral_distance, mc2sp, mcep_alpha,
                                    mel_cepstral_distortion, real_cepstrum, sp2mc)
from hmvc.features.config import AnalysisConfig, AnalysisGeometry, default_mcep_order
from hmvc.features.extractor import AcousticFeatures, FeatureExtractor, extract, extract_file
from hmvc.features.normalization import FeatureNormalizer, normalize_utterance

__all__ = [
    "AnalysisConfig", "AnalysisGeometry", "default_mcep_order",
    "FeatureExtractor", "AcousticFeatures", "extract", "extract_file",
    "FeatureNormalizer", "normalize_utterance",
    "AudioData", "read_wav", "write_wav", "resample", "prepare_signal",
    "sp2mc", "mc2sp", "mcep_alpha", "real_cepstrum",
    "mel_cepstral_distortion", "log_spectral_distance",
]
