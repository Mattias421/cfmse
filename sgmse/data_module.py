from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
from pathlib import Path


def get_window(window_type, window_length):
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(
        self,
        data_dir,
        subset,
        dummy,
        shuffle_spec,
        num_frames,
        format="default",
        normalize="noisy",
        spec_transform=None,
        stft_kwargs=None,
        unpaired=False,
        speaker_file=None,
        expected_sample_rate=16000,
        **ignored_kwargs,
    ):
        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = []
            self.clean_files += sorted(glob(join(data_dir, subset, "clean", "*.wav")))
            self.clean_files += sorted(
                glob(join(data_dir, subset, "clean", "**", "*.wav"))
            )
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
            self.noisy_files += sorted(
                glob(join(data_dir, subset, "noisy", "**", "*.wav"))
            )
        elif format == "reverb":
            self.clean_files = []
            self.clean_files += sorted(
                glob(join(data_dir, subset, "anechoic", "*.wav"))
            )
            self.clean_files += sorted(
                glob(join(data_dir, subset, "anechoic", "**", "*.wav"))
            )
            self.noisy_files = []
            self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "*.wav")))
            self.noisy_files += sorted(
                glob(join(data_dir, subset, "reverb", "**", "*.wav"))
            )
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        if len(self.clean_files) != len(self.noisy_files):
            raise ValueError(
                f"Mismatched clean/noisy file counts in {data_dir}/{subset}: "
                f"{len(self.clean_files)} != {len(self.noisy_files)}"
            )

        clean_root_name = "anechoic" if format == "reverb" else "clean"
        noisy_root_name = "reverb" if format == "reverb" else "noisy"
        clean_root = Path(data_dir, subset, clean_root_name)
        noisy_root = Path(data_dir, subset, noisy_root_name)
        clean_by_name = {
            str(Path(path).relative_to(clean_root)): path for path in self.clean_files
        }
        noisy_by_name = {
            str(Path(path).relative_to(noisy_root)): path for path in self.noisy_files
        }
        if clean_by_name.keys() != noisy_by_name.keys():
            missing_noisy = sorted(clean_by_name.keys() - noisy_by_name.keys())[:5]
            missing_clean = sorted(noisy_by_name.keys() - clean_by_name.keys())[:5]
            raise ValueError(
                f"Clean/noisy filenames do not match in {data_dir}/{subset}; "
                f"missing noisy={missing_noisy}, missing clean={missing_clean}"
            )

        paired_names = sorted(clean_by_name)
        if speaker_file is not None:
            speaker_path = Path(speaker_file)
            with speaker_path.open() as handle:
                speakers = {
                    line.split("#", 1)[0].strip()
                    for line in handle
                    if line.split("#", 1)[0].strip()
                }
            if not speakers:
                raise ValueError(f"Speaker file is empty: {speaker_path}")
            available = {Path(name).stem.split("_", 1)[0] for name in paired_names}
            unknown = sorted(speakers - available)
            if unknown:
                raise ValueError(
                    f"Speaker file {speaker_path} contains speakers absent from "
                    f"{subset}: {unknown}"
                )
            paired_names = [
                name
                for name in paired_names
                if Path(name).stem.split("_", 1)[0] in speakers
            ]

        if not paired_names:
            raise ValueError(f"No paired audio files found in {data_dir}/{subset}")
        self.clean_files = [clean_by_name[name] for name in paired_names]
        self.noisy_files = [noisy_by_name[name] for name in paired_names]

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.unpaired = unpaired
        self.expected_sample_rate = expected_sample_rate

        assert all(
            k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]
        ), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) is True, (
            "'center' must be True for current implementation"
        )

    def __getitem__(self, i):
        if self.unpaired:
            x, sr_x = load(self.clean_files[i])
            y, sr_y = load(
                self.noisy_files[torch.randint(0, self.__len__(), (1,)).item()]
            )
        else:
            x, sr_x = load(self.clean_files[i])
            y, sr_y = load(self.noisy_files[i])

        if sr_x != sr_y:
            raise ValueError(
                f"Sample rates differ for pair {self.clean_files[i]} and "
                f"{self.noisy_files[i]}: {sr_x} != {sr_y}"
            )
        if self.expected_sample_rate and sr_x != self.expected_sample_rate:
            raise ValueError(
                f"Expected {self.expected_sample_rate} Hz audio, got {sr_x} Hz in "
                f"{self.clean_files[i]}"
            )

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            x = x[..., start : start + target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

        if self.unpaired:
            current_len = y.size(-1)
            pad = max(target_len - current_len, 0)
            if pad == 0:
                # extract random part of the audio file
                if self.shuffle_spec:
                    start = int(np.random.uniform(0, current_len - target_len))
                else:
                    start = int((current_len - target_len) / 2)
                y = y[..., start : start + target_len]
            else:
                # pad audio if the length T is smaller than num_frames
                y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode="constant")
        else:
            if pad == 0:
                y = y[..., start : start + target_len]
            else:
                # pad audio if the length T is smaller than num_frames
                y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac, _ = y.abs().max(dim=1)
        elif self.normalize == "clean":
            normfac, _ = x.abs().max(dim=1)
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files) / 200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--base_dir",
            type=str,
            required=True,
            help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=("default", "reverb"),
            default="default",
            help="Read file paths according to file naming format.",
        )
        parser.add_argument(
            "--batch_size", type=int, default=8, help="The batch size. 8 by default."
        )
        parser.add_argument(
            "--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default."
        )  # to assure 256 freq bins
        parser.add_argument(
            "--hop_length",
            type=int,
            default=128,
            help="Window hop length. 128 by default.",
        )
        parser.add_argument(
            "--num_frames",
            type=int,
            default=256,
            help="Number of frames for the dataset. 256 by default.",
        )
        parser.add_argument(
            "--window",
            type=str,
            choices=("sqrthann", "hann"),
            default="hann",
            help="The window function to use for the STFT. 'hann' by default.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers to use for DataLoaders. 4 by default.",
        )
        parser.add_argument(
            "--dummy",
            action="store_true",
            help="Use reduced dummy dataset for prototyping.",
        )
        parser.add_argument(
            "--unpaired",
            action="store_true",
            help="Whether to use unpaired data or not",
        )
        parser.add_argument(
            "--spec_factor",
            type=float,
            default=0.15,
            help="Factor to multiply complex STFT coefficients by. 0.15 by default.",
        )
        parser.add_argument(
            "--spec_abs_exponent",
            type=float,
            default=0.5,
            help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.",
        )
        parser.add_argument(
            "--normalize",
            type=str,
            choices=("clean", "noisy", "not"),
            default="noisy",
            help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.",
        )
        parser.add_argument(
            "--transform_type",
            type=str,
            choices=("exponent", "log", "none"),
            default="exponent",
            help="Spectogram transformation for input representation.",
        )
        parser.add_argument(
            "--train_speaker_file",
            type=str,
            default=None,
            help=(
                "Optional text file containing one training speaker ID per line. "
                "Validation and test data are never filtered."
            ),
        )
        parser.add_argument(
            "--expected_sample_rate",
            type=int,
            default=16000,
            help="Fail if a loaded training pair is not at this sample rate.",
        )
        return parser

    def __init__(
        self,
        base_dir,
        format="default",
        batch_size=8,
        n_fft=510,
        hop_length=128,
        num_frames=256,
        window="hann",
        num_workers=4,
        dummy=False,
        unpaired=False,
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        gpu=True,
        normalize="noisy",
        transform_type="exponent",
        train_speaker_file=None,
        expected_sample_rate=16000,
        **kwargs,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs
        self.unpaired = unpaired
        self.train_speaker_file = train_speaker_file
        self.expected_sample_rate = expected_sample_rate

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs,
            num_frames=self.num_frames,
            spec_transform=self.spec_fwd,
            **self.kwargs,
        )
        if stage == "fit" or stage is None:
            self.train_set = Specs(
                data_dir=self.base_dir,
                subset="train",
                dummy=self.dummy,
                shuffle_spec=True,
                format=self.format,
                normalize=self.normalize,
                unpaired=self.unpaired,
                speaker_file=self.train_speaker_file,
                expected_sample_rate=self.expected_sample_rate,
                **specs_kwargs,
            )
            self.valid_set = Specs(
                data_dir=self.base_dir,
                subset="valid",
                dummy=self.dummy,
                shuffle_spec=False,
                format=self.format,
                normalize=self.normalize,
                expected_sample_rate=self.expected_sample_rate,
                **specs_kwargs,
            )
        if stage == "test" or stage is None:
            self.test_set = Specs(
                data_dir=self.base_dir,
                subset="test",
                dummy=self.dummy,
                shuffle_spec=False,
                format=self.format,
                normalize=self.normalize,
                expected_sample_rate=self.expected_sample_rate,
                **specs_kwargs,
            )

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(
            spec, **{**self.istft_kwargs, "window": window, "length": length}
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
        )
