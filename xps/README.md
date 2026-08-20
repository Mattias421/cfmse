# Experiment execution

`experiments.json` is the single source of truth for the 22 runs configured from
the examples and alternatives in `xp_plan.md`. `run_experiment.py` converts one
manifest entry into an explicit training or evaluation command. The server
wrappers only provide resource and environment details; they do not duplicate
model hyperparameters.

## Allocation

| Server | Resources per job | Assigned experiments | Why |
| --- | --- | --- | --- |
| Stanage | 1 available A100, H100, or H100 NVL | 15 data-efficiency runs + 3 conditioning ablations | A generic GRES lets each array task use the first available Stanage GPU type. |
| Phoebe | GPUs 0,1 | 2 direct baselines, sequentially | These new baselines are easiest to inspect on the single interactive GPU pair. |
| Mimas | GPUs 0,1 and 2,3 | annealed DDP + ICFM-to-DDP fine-tune | The two independent jobs can run concurrently once the base ICFM checkpoint is available. |

The conditioned full-data jobs in the data-efficiency family are reused as the
controls for the conditioning ablation. The manifest uses one seed (`42`) and a
global batch of 16: per-device batch 16 on one Stanage GPU and 8 on each of two
small GPUs. Training is FP32 on every server.

## One-time setup on each server

1. Copy `xps/env/<server>.env.example` to `xps/env/<server>.env` and fill in the
   data, output, Python, and activation paths. These local `.env` files should not
   be committed.
2. Activate/build a portable environment for all three Stanage node types. A100
   uses `sm80`; H100 and H100 NVL use `sm90`, and the H100-NVL nodes use a
   different CPU architecture. The batch script detects the allocated GPU before
   compiling CUDA extensions. GPU nodes now run EL9, so check `module avail`
   before accepting the example GCC/CUDA module names.
   `requirements_version.txt` is the cross-host version contract; install a
   CUDA-enabled PyTorch build matching those pins and the host driver/toolkit.
3. Validate everything without launching training:

   ```bash
   source xps/lib/runtime.sh stanage  # use mimas or phoebe on those hosts
   "${CFMSE_PYTHON}" xps/check_environment.py --require-cuda-build
   "${CFMSE_PYTHON}" xps/run_experiment.py --validate
   "${CFMSE_PYTHON}" xps/validate_data.py --base-dir "${CFMSE_DATA_ROOT}"
   "${CFMSE_PYTHON}" xps/run_experiment.py --server stanage --list
   "${CFMSE_PYTHON}" xps/run_experiment.py --server stanage --index 0 --dry-run
   ```

The checked dataset contains 10,802 training pairs from 26 speakers, 770
validation pairs from exactly `p226` and `p287`, and 824 test pairs from `p232`
and `p257`; all are paired, mono, and 16 kHz. The preflight repeats these checks
on each server. The 20 enhancement files used for checkpoint metrics are sampled
deterministically across each validation speaker (10 per speaker), rather than
being taken from the beginning of the sorted file list.

## Launching

Stanage:

```bash
xps/stanage/submit.sh
```

The submit helper performs a login-side check of the configured Python/module
stack, runs the data/manifest preflight, and creates `logs/slurm` before calling
`sbatch`. Use
`CFMSE_SKIP_DATA_PREFLIGHT=1 xps/stanage/submit.sh` only after an unchanged
dataset has already passed; the full header scan is metadata-heavy. The array is
`0-17%6`: one generic `gpu:1` request per task, six tasks at once. Slurm may place
each task on the first available A100, H100, or H100 NVL; the resolved run record
captures the actual type. Stanage currently allows at most 12 GPUs per user, so
`%6` leaves capacity for other work. The script deliberately omits `--account`
for normal/free access. A special allocation or future high-priority service
needs site-issued account/partition directives.

The login-side check cannot verify worker GPU/driver compatibility. Before
releasing the full array, run one reduced task into a separate output root:

```bash
CFMSE_LOG_ROOT=/mnt/parscratch/users/USER/cfmse-smoke \
CFMSE_TRAIN_EXTRA_ARGS='--dummy --max_steps 1 --num_eval_files 0 --nolog' \
xps/stanage/submit.sh --array=0
```

Each array task stages the 7.5 GB/~25k-file dataset to its node-local `$TMPDIR`
before training, avoiding repeated small-file reads from Lustre. Set
`CFMSE_STAGE_DATA=0` only if `CFMSE_DATA_ROOT` is already node-local or site
profiling shows staging is undesirable. Checkpoints and metrics always remain
under `CFMSE_LOG_ROOT`, which should point to job-persistent parscratch storage.
Parscratch is temporary and unbacked, not archival storage; copy final results to
an appropriate backed-up store from a login node. `/shared` is unavailable on
Stanage workers, so the repository, data, environment, and live outputs must all
be on worker-visible storage.

Phoebe:

```bash
xps/phoebe.sh
```

Mimas (annealing can start before the base checkpoint is transferred):

```bash
xps/mimas.sh anneal
export CFMSE_ICFM_FULL_CKPT=/path/to/data_icfm_full/checkpoints/best-pesq-....ckpt
xps/mimas.sh finetune
# Or, once the checkpoint exists, run both GPU pairs concurrently:
xps/mimas.sh all
```

The default phase is training. To evaluate completed runs using the best-PESQ
checkpoint (falling back to `last.ckpt`), use for example
`CFMSE_PHASE=evaluate xps/phoebe.sh` (or the corresponding server command).
`CFMSE_PHASE=all` trains and evaluates within one allocation, but
may need more than the requested Stanage wall time. Core evaluation writes
enhanced audio plus PESQ, ESTOI, SI-SDR, SI-SIR, and SI-SAR results. WhiSQA and
DNSMOS are not included because their external repositories/model paths are not
specified.

Set `CFMSE_NOLOG=1` for local CSV logging and deterministic checkpoints without
W&B. `WANDB_MODE=offline` retains W&B metadata without network access.
Launches enforce the exact core-package pins in `requirements_version.txt`; set
`CFMSE_ALLOW_VERSION_DRIFT=1` only for an intentional, separately validated
environment change. The resolved run record captures the actual core package
versions, host, Git state, and selected scheduler/CUDA environment.
`CFMSE_TRAIN_EXTRA_ARGS` may contain temporary training overrides, for example
`--dummy --max_epochs 1 --num_eval_files 0 --nolog` for a smoke test. Every
resolved training command and manifest row is stored in
`<run>/resolved_run.json`; evaluation writes `resolved_evaluation.json`, and
every launch is appended to `launch_history.jsonl`. Original and resumed command
lines are retained in `command.txt` and `command_history.tsv`. W&B keeps a
persistent ID from `wandb_id.txt` across explicit resumes; local CSV metrics use
one safely versioned directory per launch. To
resume selected interrupted jobs from their deterministic `last.ckpt`, use an
inline exported control such as
`CFMSE_RESUME=1 xps/stanage/submit.sh --array=3,7` or
`CFMSE_RESUME=1 xps/phoebe.sh`.

## Operational assumptions requiring confirmation

- Data subsets are nested seeded speaker selections: 26 (`full`), 13
  (`medium`), 7 (`low`), 3 (`very_low`), and `p243` alone (`single`). The plan did
  not specify exact counts or IDs.
- `p226` and `p287` are held out from all training and the existing `test` split
  is immutable.
- "ICFM c=0.1" is interpreted as `--sigma 0.1`. ICFM has no `c` argument.
- SGMSE is interpreted as OUVE with repository defaults `theta=1.5`,
  `sigma_min=0.05`, and `sigma_max=0.5`. OUVE has no `k`/`c`, so the request for
  "a good k and c for SGMSE" needs correction or a tuning grid.
- Both augmented-DDP alternatives are included even though the plan says "or."
- For ICFM, DDP is interpreted as one-step *direct prediction* of the clean-minus-
  noisy flow residual: direct/augmented ICFM retains `loss_type=flow_matching`,
  and the DP sampler adds the noisy input. This makes fine-tuning
  function-preserving and lets annealing change only the time distribution. The
  plain U-Net instead uses `loss_type=data_prediction` to regress clean speech.
- Fixed-`t=1` ICFM retains the specified `sigma=0.1` path perturbation, so its
  training input is noisy speech plus path noise while DP inference starts from
  noisy speech exactly. The plain regressor uses `path_noise_scale=0`. Confirm
  whether fixed-`t=1` ICFM should also remove its perturbation.
- Annealing linearly replaces uniform times with `t=1` over 250 epochs while
  holding the flow-matching objective and stochastic path fixed.
- The "unconditioned" ablation removes only the network's explicit noisy `y`
  channel. The bridge dynamics and sampler still start from/use `y`; removing it
  there would define a different generative method.
- The simple regressor reuses NCSN++'s U-Net topology, zeros the extra noisy
  conditioning channel, supplies the noisy sample as `x_t`, and removes the time
  embedding/projections from its residual blocks. A wholly separate plain U-Net
  topology would require a more specific design.
- Runs use 300 epochs, validation every 10 epochs, one seed, and a global batch of
  16. Fixed epochs mean smaller data splits receive fewer optimizer updates. If
  the intended comparison holds optimizer steps constant, set a common
  `--max_steps` instead and revise the manifest.
- The function-preserving fine-tune uses 50 new epochs with a fresh optimizer and
  initializes both network and EMA weights from `data_icfm_full`. Learning rate
  and duration were not specified in the plan. Launch-time checkpoint validation
  rejects a source that is not conditioned, uniform-time, flow-matching ICFM.
- Main ICFM evaluations use a 50-step ODE; direct/augmented DDP uses one-step
  direct prediction; SGMSE uses 30-step PC and SB-VE a 50-step ODE. Confirm these
  samplers before final reporting.
- Stanage deliberately allows A100, H100, or H100 NVL scheduling for queue
  throughput. It requests the common-safe baseline of 96 GB host RAM and 12 CPUs,
  an 80-hour wall time, normal/free access, and array concurrency 6. GPU type is
  therefore a nuisance variable that should be included in runtime/result
  analysis. Special accounts or a different measured wall time require changes.

Stanage directives follow Sheffield's current guidance:

- <https://docs.hpc.shef.ac.uk/en/latest/stanage/GPUComputingStanage.html#submitting-gpu-batch-jobs>
- <https://docs.hpc.shef.ac.uk/en/latest/stanage/GPUComputingStanage.html#choosing-appropriate-gpu-compute-resources>
- <https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/advanced/advanced_job_submission_and_control.html#job-or-task-arrays>
