"""Command-line interface for the strict 1:1 voice converter.

Reads WAV files (any sample rate/channels — resampled to 22050 Hz mono) and
writes a 16-bit PCM WAV output.  Example::

    python -m hmvc.conversion --src source.wav --tgt target.wav \
        --out converted.wav --mode voc --blend 0.8 --neighbours 6 \
        --range standard --auto --seed 42
"""

from __future__ import annotations

import argparse
import sys

from hmvc.conversion import (
    PRESETS,
    VOICE_RANGES,
    ConversionConfig,
    convert,
    load_audio,
    preset_params,
    write_wav,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hmvc.conversion",
        description="Strict 1:1 port of the browser retrieval voice converter.",
    )
    p.add_argument("--src", required=True, help="source voice WAV (required)")
    p.add_argument("--tgt", help="target voice WAV (optional)")
    p.add_argument("--out", default="converted.wav", help="output WAV (default converted.wav)")
    p.add_argument("--mode", default="voc", choices=("voc", "grain", "env"),
                   help="conversion method (default: voc, smooth vocoder (trained))")
    p.add_argument("--range", default="standard", choices=sorted(VOICE_RANGES),
                   help="voice range preset (default: standard)")
    p.add_argument("--preset", default=None, choices=list(PRESETS),
                   help="preset that sets pitch/formant/blend (overrides --pitch/--form/--blend)")
    p.add_argument("--pitch", type=float, default=0.0,
                   help="pitch shift in semitones (default 0)")
    p.add_argument("--form", type=float, default=1.0,
                   help="formant warping factor (default 1.0)")
    p.add_argument("--blend", type=float, default=None, help="envelope blend 0..1 (default 0.8)")
    p.add_argument("--neighbours", type=int, default=6, help="retrieval neighbours (default 6)")
    p.add_argument("--no-auto", action="store_true", help="disable auto target-pitch matching")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the vocoder's noise (default: nondeterministic, as the original)")
    p.add_argument("--progress", action="store_true", help="print progress (one line per stage)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    fmin, fmax = VOICE_RANGES[args.range]
    pitch, form, blend = args.pitch, args.form, args.blend
    if args.preset is not None:
        params = preset_params(args.preset)
        if params is not None:
            pitch, form, blend = params
    if blend is None:
        blend = 0.8
    config = ConversionConfig(
        mode=args.mode,
        pitch_semitones=pitch,
        formant_factor=form,
        envelope_blend=blend,
        neighbours=args.neighbours,
        fmin=fmin,
        fmax=fmax,
        auto_pitch=not args.no_auto,
        rng_seed=args.seed,
    )

    def progress(frac, message):
        if args.progress:
            print(f"[{frac:5.1%}] {message}", file=sys.stderr, flush=True)

    x = load_audio(args.src)
    t = load_audio(args.tgt) if args.tgt else None
    result = convert(x, t, config, progress)
    write_wav(args.out, result.samples)
    print(
        f"Done. mode={result.mode} src={result.source_length / result.sample_rate:.1f}s "
        f"ratio={result.pitch_ratio:.4f} db_active={result.target_database_active} "
        f"shifted={result.pitch_shifted} -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
