"""Minimal command line: ``python -m hmvc.features {extract,info} ...``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from hmvc.features.config import AnalysisConfig
from hmvc.features.extractor import AcousticFeatures, FeatureExtractor


def _summary(features: AcousticFeatures) -> dict:
    voiced = features.f0[features.voiced]
    return {
        "frames": len(features), "duration_s": round(features.duration, 4),
        "sample_rate": features.sample_rate, "hop_length": features.hop_length,
        "fft_size": features.fft_size, "mcep_order": features.mcep_order,
        "mcep_alpha": features.mcep_alpha,
        "voiced_fraction": round(float(np.mean(features.voiced)) if len(features) else 0.0, 4),
        "median_f0_hz": round(float(np.median(voiced)), 2) if voiced.size else None,
        "band_aperiodicity_edges_hz": [round(float(e), 1) for e in features.aperiodicity_edges],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hmvc.features",
                                     description="HMVC classical feature extraction")
    sub = parser.add_subparsers(dest="command", required=True)
    ext = sub.add_parser("extract", help="extract features from a WAV file")
    ext.add_argument("wav")
    ext.add_argument("output", help="output .npz path")
    ext.add_argument("--config", help="JSON file with AnalysisConfig options")
    ext.add_argument("--sample-rate", type=int)
    ext.add_argument("--hop-ms", type=float)
    ext.add_argument("--f0-floor", type=float)
    ext.add_argument("--f0-ceil", type=float)
    ext.add_argument("--mcep-order", type=int)
    ext.add_argument("--keep-envelope", action="store_true")
    info = sub.add_parser("info", help="summarise a saved feature file")
    info.add_argument("features")
    args = parser.parse_args(argv)

    if args.command == "info":
        print(json.dumps(_summary(AcousticFeatures.load(args.features)), indent=2))
        return 0
    options = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            options.update(json.load(fh))
    for key in ("sample_rate", "hop_ms", "f0_floor", "f0_ceil", "mcep_order"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    if args.keep_envelope:
        options["keep_envelope"] = True
    features = FeatureExtractor(AnalysisConfig.from_dict(options)).extract_file(args.wav)
    features.save(args.output)
    print(json.dumps(_summary(features), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
