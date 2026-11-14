"""Small deterministic synthetic case shared by the oracle tool and local tests.

This is test input/orchestration, not a synthesis implementation. All model,
feature, source, and waveform generation is delegated to the selected engine.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import wave

import numpy as np

FS = 16000


def module(namespace, name):
    return import_module(namespace + "." + name)


def corpus(directory):
    directory = Path(directory)
    wav_dir = directory / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    lines = ["# utt_id\tonset\toffset\tphone\tnote"]
    for index, notes in enumerate(((60, 64, 67), (67, 60, 64))):
        name = f"synthetic-{index}"
        script = (("sil", 0.04, None), ("a", 0.20, notes[0]),
                  ("s", 0.08, None), ("i", 0.20, notes[1]),
                  ("a", 0.20, notes[2]), ("sil", 0.04, None))
        chunks, onset = [], 0.0
        for phone, duration, note in script:
            count = round(duration * FS)
            time = np.arange(count, dtype=np.float64) / FS
            if phone == "sil":
                signal = np.zeros(count)
            elif phone == "s":
                signal = 0.04 * rng.normal(size=count)
            else:
                frequency = 440.0 * 2.0 ** ((note - 69.0) / 12.0)
                phase = 2 * np.pi * frequency * time
                harmonic = 0.45 if phone == "a" else 0.15
                signal = 0.25 * (np.sin(phase) + harmonic * np.sin(2 * phase) + 0.1 * np.sin(3 * phase))
                ramp = np.minimum(np.arange(count), np.arange(count)[::-1]) / 160.0
                signal *= np.clip(ramp, 0.0, 1.0)
                signal += 0.0005 * rng.normal(size=count)
            chunks.append(signal)
            lines.append(f"{name}\t{onset:.6f}\t{onset + duration:.6f}\t{phone}\t{'-' if note is None else note}")
            onset += duration
        audio = np.concatenate(chunks)
        payload = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        with wave.open(str(wav_dir / f"{name}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(FS)
            handle.writeframes(payload)
    labels = directory / "labels.tsv"
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return labels, wav_dir


def train(namespace, directory):
    trainer = module(namespace, "core.trainer")
    phonemes = module(namespace, "core.phonemes")
    labels, wav_dir = corpus(directory)
    inventory = phonemes.PhonemeSet.from_dict({
        "defaults": {
            "vowel": {"n_states": 2, "n_components": 2, "voiced": True},
            "unvoiced_consonant": {"n_states": 1, "n_components": 1, "voiced": False},
            "silence": {"n_states": 1, "n_components": 1, "voiced": False},
        },
        "phonemes": {"sil": {"type": "silence"}, "a": {"type": "vowel"},
                     "i": {"type": "vowel"}, "s": {"type": "unvoiced_consonant"}},
    })
    config = trainer.TrainingConfig(
        label_file=str(labels), wav_dir=str(wav_dir), fs=FS, fft_size=512,
        n_mcep=6, n_band=3, use_delta=True, use_delta2=True,
        n_iterations=2, min_phoneme_frames=4, vocoder="builtin", seed=7,
        context_enabled=True, context_min_frames=4, context_min_occurrences=1,
        context_max_models=16, context_global_backoff=True,
        pitch_conditioning_enabled=True, pitch_conditioning_bin_size=6,
        vibrato_estimate_from_data=False)
    model = trainer.Trainer(config, inventory).train()
    # Only non-predictive fixture provenance is normalized for a portable oracle.
    model.name = "synthetic-phase0"
    model.stats.created = ""
    model.metadata["label_file"] = "synthetic/labels.tsv"
    model.metadata["wav_dir"] = "synthetic/wav"
    model.metadata["training_config"]["label_file"] = "synthetic/labels.tsv"
    model.metadata["training_config"]["wav_dir"] = "synthetic/wav"
    return model


def score(namespace):
    return module(namespace, "core.labels").parse(
        "case 0.00 0.04 sil -\ncase 0.04 0.24 a 60\ncase 0.24 0.32 s -\n"
        "case 0.32 0.52 i 64\ncase 0.52 0.72 a 67\ncase 0.72 0.76 sil -\n")


def source_model(namespace):
    # This is the original upstream synthetic test fixture, mechanically ported
    # with its tests; its classes always come from the requested engine.
    case = import_module("hms.tests.test_source_synthesis" if namespace == "hms"
                         else "tests.upstream.test_source_synthesis")
    return case._tiny_source_model(fs=FS)


def renders(namespace, model, source):
    synthesis = module(namespace, "core.synthesizer")
    label_score = score(namespace)
    options = {
        "builtin": {},
        "mlsa": {"vocoder": "mlsa"},
        "gv": {"gv_enabled": True, "gv_weight": 0.2, "gv_iterations": 4},
        "gm": {"gm_enabled": True, "gm_strength": 0.5},
        "acoustic_f0": {"f0_source": "acoustic"},
        "state_f0": {"f0_source": "state_means"},
        "model_duration": {"duration_mode": "model", "tempo": 1.1},
        "external_f0": {},
        "source_builtin": {},
        "source_mlsa": {"vocoder": "mlsa"},
    }
    for name, overrides in options.items():
        config = synthesis.SynthesisConfig(**dict(
            {"vocoder": "builtin", "seed": 11, "vibrato": False}, **overrides))
        engine = synthesis.Synthesizer(model, config)
        kwargs = {}
        if name == "external_f0":
            count = len(engine.plan(label_score)[0])
            trajectory = np.linspace(180.0, 390.0, count)
            trajectory[:8] = trajectory[48:64] = trajectory[-8:] = 0.0
            kwargs["f0"] = trajectory
        if name.startswith("source_"):
            kwargs["source_model"] = source
        yield name, engine.synthesize(label_score, **kwargs)


def render_arrays(result):
    arrays = {"audio": result.audio, "f0": result.params.f0, "sp": result.params.sp,
              "ap": result.params.ap, "notes": result.notes, "state_ids": result.state_ids,
              "durations": result.durations, "f0_semitones": result.f0_semitones,
              "phones": np.asarray(result.phones), "state_sequence": np.asarray(result.state_sequence)}
    if result.source is not None:
        for field in ("excitation", "support", "weights", "coverage", "voiced", "f0_hz"):
            arrays["source_" + field] = getattr(result.source, field)
    return arrays
