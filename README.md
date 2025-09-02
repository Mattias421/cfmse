## CFMSE
This repository contains code for running experiments for: [Flowing Straighter with Conditional Flow Matching for Accurate Speech Enhancement](https://arxiv.org/abs/2508.20584)

To train an ICFM model with flow-matching loss, do `--backbone ncsnpp_v2 --sde icfm --sigma 0.1 --loss_type flow_matching`
For the SB-SV, we build off the SB-VE, use the c parameter to set sigma e.g. `--backbone ncsnpp_v2 --sde sbve --loss_type data_prediction --variance_type stationary --c 0.1`
For our novel one-step sampler, set `--sampler_type dp`
To use the `xps/eval.sh` script, make sure WhiSQA and DNSMOS are set up within the parent directory. Other scripts in the xps folder show examples of the settings used and how to run slurm job arrays.
Audio samples can be heard [here](https://mattias421.github.io/cfmse/)

## Installation

- Create a new virtual environment with Python 3.11 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
  - Let pip resolve the dependencies for you. If you encounter any issues, please check `requirements_version.txt` for the exact versions we used.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--nolog` to `train.py`.
    - Your logs will be stored as local CSVLogger logs in `lightning_logs/`.

## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper [2]) can be run with

```bash
python train.py --base_dir <your_base_dir>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.

## Evaluation

To evaluate on a test set, run
```bash
python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
```

to generate the enhanced .wav files, and subsequently run

```bash
python calc_metrics.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir>
```

to calculate and output the instrumental metrics.

Both scripts should receive the same `--test_dir` and `--enhanced_dir` parameters. The `--cpkt` parameter of `enhancement.py` should be the path to a trained model checkpoint, as stored by the logger in `logs/`.



## Citations / References
This work (code, readme, experimental setup) is based off [SGMSE](https://github.com/sp-uhh/sgmse) by `sp-uhh`, please check them out!
