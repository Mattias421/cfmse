#!/usr/bin/env python3
"""Check the runtime imports and pinned versions used by experiment jobs."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "requirements_version.txt"
REQUIRED_DISTRIBUTIONS = {
    "librosa": "librosa",
    "ninja": "ninja",
    "numpy": "numpy",
    "openai-whisper": "whisper",
    "pandas": "pandas",
    "pesq": "pesq",
    "protobuf": "google.protobuf",
    "pystoi": "pystoi",
    "pytorch-lightning": "pytorch_lightning",
    "scipy": "scipy",
    "setuptools": "setuptools",
    "soundfile": "soundfile",
    "torch": "torch",
    "torch-ema": "torch_ema",
    "torch-pesq": "torch_pesq",
    "torchaudio": "torchaudio",
    "torchcfm": "torchcfm",
    "torchsde": "torchsde",
    "tqdm": "tqdm",
    "wandb": "wandb",
}


def locked_versions(path):
    versions = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"Expected an exact version pin in {path}: {raw_line}")
        package, version = line.split("==", 1)
        versions[package.lower()] = version
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda-build", action="store_true")
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="Report exact-version differences without failing the preflight.",
    )
    args = parser.parse_args()

    pins = locked_versions(LOCK_FILE)
    versions = {}
    errors = []
    for distribution, module in REQUIRED_DISTRIBUTIONS.items():
        expected = pins.get(distribution)
        if expected is None:
            errors.append(f"{distribution} has no pin in {LOCK_FILE}")
            continue
        try:
            actual = importlib.metadata.version(distribution)
            importlib.import_module(module)
        except (importlib.metadata.PackageNotFoundError, ImportError) as error:
            errors.append(f"{distribution} is unavailable: {error}")
            continue
        versions[distribution] = actual
        if actual != expected:
            message = f"{distribution}: expected {expected}, found {actual}"
            if args.allow_version_drift:
                print(f"warning: {message}")
            else:
                errors.append(message)

    ninja = shutil.which("ninja")
    if ninja is None:
        errors.append("ninja executable is not on PATH")

    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = shutil.which("nvcc")
    if args.require_cuda_build and nvcc is None:
        errors.append(
            "nvcc is not on PATH; uv supplies Python dependencies, but the "
            "NCSN++ extensions also require an external CUDA toolkit"
        )

    torch = importlib.import_module("torch") if "torch" in versions else None
    cuda_build = torch.version.cuda if torch is not None else None
    if args.require_cuda_build and cuda_build is None:
        errors.append("the configured PyTorch build has no CUDA support")

    if errors:
        raise SystemExit("environment preflight failed:\n- " + "\n- ".join(errors))

    print(
        json.dumps(
            {
                "cuda_build": cuda_build,
                "cuda_home": cuda_home,
                "ninja": ninja,
                "nvcc": nvcc,
                "packages": versions,
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("environment preflight passed")


if __name__ == "__main__":
    main()
