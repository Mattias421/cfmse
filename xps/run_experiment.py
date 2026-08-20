#!/usr/bin/env python3
"""Validate, display, train, or evaluate one entry in the experiment manifest."""

import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "experiments.json"
VALID_SERVERS = {"stanage", "mimas", "phoebe"}
VALID_INIT_PROFILES = {"icfm_uniform_flow_matching"}
PROVENANCE_PACKAGES = (
    "librosa",
    "numpy",
    "openai-whisper",
    "pandas",
    "pesq",
    "pystoi",
    "pytorch-lightning",
    "soundfile",
    "torch",
    "torch-ema",
    "torch-pesq",
    "torchaudio",
    "torchcfm",
    "torchsde",
    "wandb",
)


def load_manifest(path):
    with path.open() as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing manifest schema_version")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Manifest must contain a non-empty experiments list")

    required = {"id", "family", "model", "split", "server", "train_args", "eval_args"}
    ids = []
    for experiment in experiments:
        missing = required - experiment.keys()
        if missing:
            raise ValueError(
                f"Experiment is missing fields {sorted(missing)}: {experiment}"
            )
        if experiment["server"] not in VALID_SERVERS:
            raise ValueError(
                f"Invalid server {experiment['server']} for {experiment['id']}"
            )
        for field in ("train_args", "eval_args"):
            if not isinstance(experiment[field], list) or not all(
                isinstance(value, str) for value in experiment[field]
            ):
                raise ValueError(f"{experiment['id']}.{field} must be a string list")
        init_profile = experiment.get("init_checkpoint_profile")
        if init_profile and init_profile not in VALID_INIT_PROFILES:
            raise ValueError(
                f"Invalid initialization profile {init_profile} for {experiment['id']}"
            )
        if init_profile and not experiment.get("init_checkpoint_env"):
            raise ValueError(
                f"{experiment['id']} has an init profile but no checkpoint env"
            )
        split_file = ROOT / "xps" / "splits" / f"{experiment['split']}.txt"
        if not split_file.is_file():
            raise ValueError(f"Missing split file for {experiment['id']}: {split_file}")
        ids.append(experiment["id"])

    duplicates = sorted(
        {experiment_id for experiment_id in ids if ids.count(experiment_id) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate experiment IDs: {duplicates}")
    known_ids = set(ids)
    dependencies = {}
    for experiment in experiments:
        dependency = experiment.get("depends_on")
        if dependency and dependency not in known_ids:
            raise ValueError(f"Unknown dependency {dependency} for {experiment['id']}")
        dependencies[experiment["id"]] = dependency

    visiting = set()
    visited = set()

    def visit(experiment_id):
        if experiment_id in visiting:
            raise ValueError(f"Dependency cycle includes {experiment_id}")
        if experiment_id in visited:
            return
        visiting.add(experiment_id)
        dependency = dependencies[experiment_id]
        if dependency:
            visit(dependency)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in dependencies:
        visit(experiment_id)


def experiments_for_server(manifest, server):
    return [
        experiment
        for experiment in manifest["experiments"]
        if server is None or experiment["server"] == server
    ]


def select_experiment(manifest, server, experiment_id, index):
    candidates = experiments_for_server(manifest, server)
    if experiment_id is not None:
        matches = [item for item in candidates if item["id"] == experiment_id]
        if not matches:
            raise ValueError(f"No experiment {experiment_id!r} assigned to {server!r}")
        return matches[0]
    if index is None:
        raise ValueError("Select an experiment with --id or --index")
    if index < 0 or index >= len(candidates):
        raise ValueError(
            f"Index {index} is outside the {server} range 0..{len(candidates) - 1}"
        )
    return candidates[index]


def data_root_from_environment():
    configured = os.environ.get("CFMSE_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    data_parent = os.environ.get("DATA")
    if data_parent:
        return Path(data_parent).expanduser().resolve() / "VB+DMD"
    raise ValueError("Set CFMSE_DATA_ROOT (or DATA with VB+DMD below it)")


def output_root_from_environment():
    configured = os.environ.get("CFMSE_LOG_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else ROOT / "logs" / "experiments"
    )


def python_from_environment():
    return os.environ.get("CFMSE_PYTHON", sys.executable)


def parse_flag_value(arguments, flag, default=None):
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def train_command(manifest, experiment, args, run_dir, data_root):
    defaults = manifest.get("defaults", {})
    if args.global_batch_size % args.devices:
        raise ValueError("--global-batch-size must be divisible by --devices")
    per_device_batch = args.global_batch_size // args.devices
    strategy = "auto" if args.devices == 1 else "ddp_find_unused_parameters_false"
    workers = args.num_workers
    if workers is None:
        workers = 8 if experiment["server"] == "stanage" else 4

    command = [
        python_from_environment(),
        "-u",
        str(ROOT / "train.py"),
        "--base_dir",
        str(data_root),
        "--train_speaker_file",
        str(ROOT / "xps" / "splits" / f"{experiment['split']}.txt"),
        "--run_dir",
        str(run_dir),
        "--wandb_name",
        experiment["id"],
        "--seed",
        str(defaults.get("seed", 42)),
        "--devices",
        str(args.devices),
        "--strategy",
        strategy,
        "--batch_size",
        str(per_device_batch),
        "--num_workers",
        str(workers),
    ]
    command.extend(defaults.get("train_args", []))
    command.extend(experiment["train_args"])

    init_env = experiment.get("init_checkpoint_env")
    if init_env and not args.resume:
        checkpoint = os.environ.get(init_env)
        if not checkpoint:
            if args.dry_run:
                checkpoint = f"<set-{init_env}>"
            else:
                raise ValueError(
                    f"{experiment['id']} requires {init_env} to name the weights-only "
                    "initialization checkpoint"
                )
        if not args.dry_run and not Path(checkpoint).is_file():
            raise ValueError(f"Initialization checkpoint does not exist: {checkpoint}")
        command.extend(["--init_ckpt", checkpoint])
        init_profile = experiment.get("init_checkpoint_profile")
        if init_profile:
            command.extend(["--init_ckpt_profile", init_profile])

    last_checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if args.resume:
        if not last_checkpoint.is_file():
            raise ValueError(f"Cannot resume; checkpoint is missing: {last_checkpoint}")
        command.extend(["--ckpt", str(last_checkpoint)])
    elif last_checkpoint.exists():
        raise ValueError(
            f"Run already has a checkpoint: {last_checkpoint}. Use --resume explicitly."
        )

    if os.environ.get("CFMSE_NOLOG", "").lower() in {"1", "true", "yes"}:
        command.append("--nolog")
    extra = os.environ.get("CFMSE_TRAIN_EXTRA_ARGS")
    if extra:
        command.extend(shlex.split(extra))
    return command


def checkpoint_for_evaluation(run_dir, explicit_checkpoint, dry_run):
    if explicit_checkpoint:
        checkpoint = Path(explicit_checkpoint).expanduser().resolve()
        if not dry_run and not checkpoint.is_file():
            raise ValueError(f"Evaluation checkpoint does not exist: {checkpoint}")
        return checkpoint
    checkpoint_dir = run_dir / "checkpoints"
    pesq_checkpoints = sorted(checkpoint_dir.glob("best-pesq-*.ckpt"))
    if pesq_checkpoints:
        return pesq_checkpoints[-1]
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.is_file() or dry_run:
        return last_checkpoint
    raise ValueError(f"No best-PESQ or last checkpoint found below {checkpoint_dir}")


def evaluation_commands(experiment, args, run_dir, data_root):
    checkpoint = checkpoint_for_evaluation(
        run_dir, args.evaluation_checkpoint, args.dry_run
    )
    sampler = parse_flag_value(experiment["eval_args"], "--sampler_type", "default")
    steps = parse_flag_value(experiment["eval_args"], "--N", "default")
    enhanced_dir = run_dir / "evaluation" / f"{sampler}_N{steps}"
    results_file = enhanced_dir / "_results.csv"
    if results_file.exists() and not args.force_evaluation:
        raise ValueError(
            f"Evaluation already exists: {results_file}. Use --force-evaluation to rerun."
        )
    enhance = [
        python_from_environment(),
        "-u",
        str(ROOT / "enhancement.py"),
        "--test_dir",
        str(data_root / "test" / "noisy"),
        "--enhanced_dir",
        str(enhanced_dir),
        "--ckpt",
        str(checkpoint),
        "--device",
        "cuda",
    ] + experiment["eval_args"]
    metrics = [
        python_from_environment(),
        "-u",
        str(ROOT / "calc_metrics.py"),
        "--clean_dir",
        str(data_root / "test" / "clean"),
        "--noisy_dir",
        str(data_root / "test" / "noisy"),
        "--enhanced_dir",
        str(enhanced_dir),
    ]
    return [enhance, metrics]


def show_list(manifest, server):
    print("index\tid\tfamily\tmodel\tsplit\tserver\tdependency")
    for index, experiment in enumerate(experiments_for_server(manifest, server)):
        print(
            "\t".join(
                [
                    str(index),
                    experiment["id"],
                    experiment["family"],
                    experiment["model"],
                    experiment["split"],
                    experiment["server"],
                    experiment.get("depends_on", "-"),
                ]
            )
        )


def runtime_provenance():
    package_versions = {}
    for package in PROVENANCE_PACKAGES:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
        status = []

    environment_names = (
        "CFMSE_GPU_TYPE",
        "CFMSE_NOLOG",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
        "TORCH_CUDA_ARCH_LIST",
        "WANDB_MODE",
    )
    return {
        "executable": sys.executable,
        "git_commit": commit,
        "git_status": status,
        "hostname": platform.node(),
        "packages": package_versions,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "selected_environment": {
            name: os.environ[name] for name in environment_names if name in os.environ
        },
    }


def record_run(run_dir, experiment, commands, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": experiment,
        "phase": args.phase,
        "devices": args.devices,
        "global_batch_size": args.global_batch_size,
        "commands": commands,
        "runtime": runtime_provenance(),
    }
    record_name = (
        "resolved_evaluation.json" if args.phase == "evaluate" else "resolved_run.json"
    )
    (run_dir / record_name).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "launch_history.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def acquire_run_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    handle = lock_path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ValueError(
            f"Another launcher holds the experiment lock: {lock_path}"
        ) from None
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--server", choices=sorted(VALID_SERVERS))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--id")
    selection.add_argument("--index", type=int)
    parser.add_argument("--list", action="store_true", dest="show_experiments")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument(
        "--phase", choices=("train", "evaluate", "all"), default="train"
    )
    parser.add_argument("--devices", type=int, choices=(1, 2), default=1)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-checkpoint")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.validate:
        print(f"manifest valid: {len(manifest['experiments'])} experiments")
        return
    if args.show_experiments:
        show_list(manifest, args.server)
        return
    if args.count:
        print(len(experiments_for_server(manifest, args.server)))
        return
    if args.server is None:
        parser.error("--server is required when selecting or running an experiment")

    try:
        experiment = select_experiment(manifest, args.server, args.id, args.index)
        data_root = data_root_from_environment()
        if not args.dry_run and not data_root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {data_root}")
        run_dir = output_root_from_environment() / experiment["id"]
        train_commands = []
        if args.phase in ("train", "all"):
            train_commands.append(
                train_command(manifest, experiment, args, run_dir, data_root)
            )
        eval_commands = []
        if args.phase == "evaluate" or (args.phase == "all" and args.dry_run):
            eval_commands.extend(
                evaluation_commands(experiment, args, run_dir, data_root)
            )
    except ValueError as error:
        parser.error(str(error))

    print(f"experiment: {experiment['id']} ({experiment['server']})", flush=True)
    for command in train_commands + eval_commands:
        print("command: " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    try:
        lock_handle = acquire_run_lock(run_dir)
    except ValueError as error:
        parser.error(str(error))
    try:
        record_run(run_dir, experiment, train_commands + eval_commands, args)
        for command in train_commands:
            subprocess.run(command, cwd=ROOT, check=True)
        if args.phase == "all":
            eval_commands = evaluation_commands(experiment, args, run_dir, data_root)
            record_run(run_dir, experiment, train_commands + eval_commands, args)
            for command in eval_commands:
                print("command: " + shlex.join(command), flush=True)
        for command in eval_commands:
            subprocess.run(command, cwd=ROOT, check=True)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    main()
