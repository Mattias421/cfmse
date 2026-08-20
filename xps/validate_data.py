#!/usr/bin/env python3
"""Preflight the paired 16 kHz VB+DMD tree and experiment speaker files."""

import argparse
import json
import wave
from pathlib import Path


EXPECTED_VALID_SPEAKERS = {"p226", "p287"}
EXPECTED_TEST_SPEAKERS = {"p232", "p257"}
EXPECTED_PAIR_COUNTS = {"train": 10802, "valid": 770, "test": 824}
SPLIT_ORDER = ("single", "very_low", "low", "medium", "full")


def speaker_id(path):
    return path.stem.split("_", 1)[0]


def wav_map(root):
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise ValueError(f"No WAV files found below {root}")
    return {str(path.relative_to(root)): path for path in files}


def read_speakers(path):
    with path.open() as handle:
        result = {
            line.split("#", 1)[0].strip()
            for line in handle
            if line.split("#", 1)[0].strip()
        }
    if not result:
        raise ValueError(f"Speaker file is empty: {path}")
    return result


def validate_subset(base_dir, subset, expected_rate):
    clean = wav_map(base_dir / subset / "clean")
    noisy = wav_map(base_dir / subset / "noisy")
    if clean.keys() != noisy.keys():
        missing_noisy = sorted(clean.keys() - noisy.keys())[:10]
        missing_clean = sorted(noisy.keys() - clean.keys())[:10]
        raise ValueError(
            f"{subset} clean/noisy names differ; missing noisy={missing_noisy}, "
            f"missing clean={missing_clean}"
        )

    for relative_name in clean:
        for path in (clean[relative_name], noisy[relative_name]):
            with wave.open(str(path), "rb") as handle:
                if handle.getframerate() != expected_rate:
                    raise ValueError(
                        f"Expected {expected_rate} Hz, got {handle.getframerate()} "
                        f"for {path}"
                    )
                if handle.getnchannels() != 1:
                    raise ValueError(f"Expected mono audio, got {path}")
    return {
        "pairs": len(clean),
        "speakers": sorted({speaker_id(path) for path in clean.values()}),
    }


def validate_speaker_files(speaker_dir, train_speakers):
    selections = {
        name: read_speakers(speaker_dir / f"{name}.txt") for name in SPLIT_ORDER
    }
    for name, speakers in selections.items():
        unknown = speakers - train_speakers
        if unknown:
            raise ValueError(
                f"{name}.txt has unknown training speakers: {sorted(unknown)}"
            )
    for smaller, larger in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        if not selections[smaller] < selections[larger]:
            raise ValueError(
                f"Expected {smaller}.txt to be a strict subset of {larger}.txt"
            )
    if selections["full"] != train_speakers:
        raise ValueError("full.txt must contain every training speaker exactly once")
    return {name: sorted(speakers) for name, speakers in selections.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument(
        "--speaker-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "splits",
    )
    parser.add_argument("--expected-rate", type=int, default=16000)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    summary = {
        subset: validate_subset(args.base_dir, subset, args.expected_rate)
        for subset in ("train", "valid", "test")
    }
    for subset, expected_count in EXPECTED_PAIR_COUNTS.items():
        if summary[subset]["pairs"] != expected_count:
            raise ValueError(
                f"Expected {expected_count} {subset} pairs for VB+DMD, got "
                f"{summary[subset]['pairs']}"
            )
    train_speakers = set(summary["train"]["speakers"])
    valid_speakers = set(summary["valid"]["speakers"])
    test_speakers = set(summary["test"]["speakers"])
    if valid_speakers != EXPECTED_VALID_SPEAKERS:
        raise ValueError(
            f"Validation speakers must be {sorted(EXPECTED_VALID_SPEAKERS)}, "
            f"got {sorted(valid_speakers)}"
        )
    if train_speakers & valid_speakers:
        raise ValueError("Training and validation speaker sets overlap")
    if test_speakers != EXPECTED_TEST_SPEAKERS:
        raise ValueError(
            f"Test speakers must be {sorted(EXPECTED_TEST_SPEAKERS)}, "
            f"got {sorted(test_speakers)}"
        )
    if test_speakers & (train_speakers | valid_speakers):
        raise ValueError("Test speakers overlap with training or validation")
    summary["experiment_splits"] = validate_speaker_files(
        args.speaker_dir, train_speakers
    )

    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for subset in ("train", "valid", "test"):
            print(
                f"{subset}: {summary[subset]['pairs']} pairs, "
                f"{len(summary[subset]['speakers'])} speakers"
            )
        print(
            "speaker splits: "
            + ", ".join(
                f"{name}={len(summary['experiment_splits'][name])}"
                for name in reversed(SPLIT_ORDER)
            )
        )
        print("data preflight passed")


if __name__ == "__main__":
    main()
