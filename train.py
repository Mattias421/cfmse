import torch
import argparse
import datetime as dt
import pytorch_lightning as pl
import os
import shlex
import sys
import uuid

from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel

set_torch_cuda_arch_list()
torch.set_float32_matmul_precision("high")


def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


def initialize_weights(model, checkpoint_path, profile=None):
    """Load model and EMA weights without restoring optimizer or loop state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema" not in checkpoint:
        raise ValueError(
            f"Initialization checkpoint has no EMA state: {checkpoint_path}"
        )
    if profile == "icfm_uniform_flow_matching":
        expected = {
            "backbone": "ncsnpp_v2",
            "sde": "icfm",
            "loss_type": "flow_matching",
            "time_sampling": "uniform",
            "condition_on_noisy": True,
            "condition_on_time": True,
            "path_noise_scale": 1.0,
            "sampler_type": "ode",
            "sigma": 0.1,
        }
        hyperparameters = checkpoint.get("hyper_parameters", {})
        mismatches = {
            name: (expected_value, hyperparameters.get(name))
            for name, expected_value in expected.items()
            if hyperparameters.get(name) != expected_value
        }
        train_speaker_file = hyperparameters.get("train_speaker_file")
        if not train_speaker_file or Path(train_speaker_file).name != "full.txt":
            mismatches["train_speaker_file"] = ("full.txt", train_speaker_file)
        if mismatches:
            details = ", ".join(
                f"{name}=expected {expected_value!r}, got {actual_value!r}"
                for name, (expected_value, actual_value) in mismatches.items()
            )
            raise ValueError(
                f"Initialization checkpoint does not match {profile}: {details}"
            )
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.ema.load_state_dict(checkpoint["ema"])


def deterministic_callbacks(run_dir, save_ckpt_interval, num_eval_files):
    checkpoint_dir = Path(run_dir, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="last",
            save_last=True,
            save_top_k=0,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        )
    ]
    if save_ckpt_interval > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="step={step}",
                save_top_k=-1,
                every_n_train_steps=save_ckpt_interval,
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )
    if num_eval_files:
        callbacks.extend(
            [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="best-pesq-{pesq:.2f}",
                    save_top_k=1,
                    monitor="pesq",
                    mode="max",
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                ),
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="best-si-sdr-{si_sdr:.2f}",
                    save_top_k=1,
                    monitor="si_sdr",
                    mode="max",
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=False,
                    enable_version_counter=False,
                ),
            ]
        )
    return callbacks


if __name__ == "__main__":
    # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument(
            "--backbone",
            type=str,
            choices=BackboneRegistry.get_all_names(),
            default="ncsnpp",
        )
        parser_.add_argument(
            "--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve"
        )
        parser_.add_argument("--nolog", action="store_true", help="Turn off logging.")
        parser_.add_argument(
            "--wandb_name",
            type=str,
            default=None,
            help="Name for wandb logger. If not set, a random name is generated.",
        )
        parser_.add_argument(
            "--ckpt", type=str, default=None, help="Resume training from checkpoint."
        )
        parser_.add_argument(
            "--init_ckpt",
            type=str,
            default=None,
            help=(
                "Initialize model/EMA weights from a checkpoint while starting a new "
                "optimizer and training loop."
            ),
        )
        parser_.add_argument(
            "--init_ckpt_profile",
            choices=("icfm_uniform_flow_matching",),
            default=None,
            help="Validate initialization checkpoint provenance and objective.",
        )
        parser_.add_argument(
            "--log_dir", type=str, default="logs", help="Directory to save logs."
        )
        parser_.add_argument(
            "--run_dir",
            type=str,
            default=None,
            help=(
                "Deterministic output directory. Checkpoints are written to "
                "<run_dir>/checkpoints even when W&B is disabled."
            ),
        )
        parser_.add_argument(
            "--seed", type=int, default=42, help="Global experiment seed."
        )
        parser_.add_argument(
            "--save_ckpt_interval",
            type=int,
            default=50000,
            help="Save checkpoint interval.",
        )

    temp_args, _ = base_parser.parse_known_args()

    # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)
    trainer_parser = parser.add_argument_group(
        "Trainer", description="Lightning Trainer"
    )
    trainer_parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Supports passing different accelerator types.",
    )
    trainer_parser.add_argument(
        "--devices", default="auto", help="How many gpus to use."
    )
    trainer_parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients."
    )
    trainer_parser.add_argument(
        "--max_epochs", type=int, default=-1, help="Number of epochs to train."
    )
    trainer_parser.add_argument(
        "--max_steps", type=int, default=-1, help="Maximum optimizer steps."
    )
    trainer_parser.add_argument(
        "--strategy",
        type=str,
        default="ddp_find_unused_parameters_false",
        help="Lightning distributed strategy.",
    )
    trainer_parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )

    ScoreModel.add_argparse_args(
        parser.add_argument_group("ScoreModel", description=ScoreModel.__name__)
    )
    sde_class.add_argparse_args(
        parser.add_argument_group("SDE", description=sde_class.__name__)
    )
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__)
    )
    # Add data module args
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description=data_module_cls.__name__)
    )
    # Parse args and separate into groups
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)
    if args.ckpt and args.init_ckpt:
        parser.error("--ckpt and --init_ckpt are mutually exclusive")
    if args.init_ckpt_profile and not args.init_ckpt:
        parser.error("--init_ckpt_profile requires --init_ckpt")

    pl.seed_everything(args.seed, workers=True)

    # Initialize logger, trainer, model, datamodule
    model = ScoreModel(
        backbone=args.backbone,
        sde=args.sde,
        data_module_cls=data_module_cls,
        **{
            **vars(arg_groups["ScoreModel"]),
            **vars(arg_groups["SDE"]),
            **vars(arg_groups["Backbone"]),
            **vars(arg_groups["DataModule"]),
        },
    )
    if args.init_ckpt:
        initialize_weights(model, args.init_ckpt, args.init_ckpt_profile)

    # Set up logger configuration
    if args.nolog and args.run_dir:
        # Lightning's CSV writer cannot safely reuse a prior header when a
        # resumed launch logs a different initial set of metric keys.
        logger = CSVLogger(save_dir=args.run_dir, name="csv")
    elif args.nolog:
        logger = None
    else:
        wandb_kwargs = {}
        if args.run_dir:
            run_path = Path(args.run_dir)
            run_path.mkdir(parents=True, exist_ok=True)
            wandb_id_path = run_path / "wandb_id.txt"
            try:
                with wandb_id_path.open("x") as handle:
                    handle.write(uuid.uuid4().hex[:16] + "\n")
            except FileExistsError:
                pass
            wandb_id = wandb_id_path.read_text().strip()
            if not wandb_id:
                raise ValueError(f"Empty W&B run ID file: {wandb_id_path}")
            wandb_kwargs.update(id=wandb_id, resume="allow")
        logger = WandbLogger(
            project="sgmse",
            log_model=False,
            save_dir=args.run_dir or args.log_dir,
            name=args.wandb_name,
            **wandb_kwargs,
        )
        logger.experiment.log_code(".")

    # Set up callbacks for logger
    if args.run_dir:
        run_path = Path(args.run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        callbacks = deterministic_callbacks(
            run_path, args.save_ckpt_interval, args.num_eval_files
        )
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            command = shlex.join(sys.argv)
            original_command = Path(run_path, "command.txt")
            try:
                with original_command.open("x") as handle:
                    handle.write(command + "\n")
            except FileExistsError:
                pass
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            with Path(run_path, "command_history.tsv").open("a") as handle:
                handle.write(f"{timestamp}\t{command}\n")
    elif logger is not None:
        callbacks = [
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_last=True,
                filename="{epoch}-last",
            )
        ]
        callbacks += [
            ModelCheckpoint(
                dirpath=join(args.log_dir, f"{str(logger.version)}-{args.wandb_name}"),
                filename="{step}",
                save_top_k=-1,
                every_n_train_steps=args.save_ckpt_interval,
            )
        ]
        if args.num_eval_files:
            checkpoint_callback_pesq = ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_top_k=1,
                monitor="pesq",
                mode="max",
                filename="{epoch}-{pesq:.2f}",
            )
            checkpoint_callback_si_sdr = ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_top_k=1,
                monitor="si_sdr",
                mode="max",
                filename="{epoch}-{si_sdr:.2f}",
            )
            callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]
    else:
        callbacks = None

    # Initialize the Trainer and the DataModule
    trainer = pl.Trainer(
        **vars(arg_groups["Trainer"]),
        logger=logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )

    # Train model
    trainer.fit(model, ckpt_path=args.ckpt)
