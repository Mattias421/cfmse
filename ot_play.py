from sgmse.data_module import SpecsDataModule
import whisper
from jiwer import cer
from torchcfm.optimal_transport import OTPlanSampler
import torch
import torch.nn.functional as F

model = whisper.load_model("base.en").cuda()
decoding_options = whisper.DecodingOptions(language="english")

dm = SpecsDataModule(base_dir="/store/store4/data/VB+DMD", unpaired=True, batch_size=16)
dm.setup()
dl = dm.train_dataloader()

target_len = (dm.num_frames - 1) * dm.hop_length


def to_audio(spec, length=None):
    return dm.istft(dm.spec_back(spec), length)


for x, y in dl:
    x_td = to_audio(x.squeeze(), target_len).cuda()
    y_td = to_audio(y.squeeze(), target_len).cuda()

    x_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(x_td), n_mels=model.dims.n_mels
    )
    y_mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(y_td), n_mels=model.dims.n_mels
    )
    x_wsp = whisper.decode(model, x_mel, decoding_options)
    y_wsp = whisper.decode(model, y_mel, decoding_options)

    x_text = [i.text for i in x_wsp]
    y_text = [i.text for i in y_wsp]

    x_enc = [i.audio_features for i in x_wsp]
    y_enc = [i.audio_features for i in y_wsp]

    print(f"target wer is {cer(y_text, x_text)}")
    # print(f"mse before OT is {(torch.stack(x_enc) - torch.stack(y_enc)).square().mean()}")
    #
    # X = list(zip(x_text, x_enc))
    # shuffle(X)
    # x_text = [i[0] for i in X]
    # x_enc = [i[1] for i in X]
    #
    # print(f"unpaired wer is {cer(y_text, x_text)}")

    ot_plan = OTPlanSampler("exact")
    x0 = torch.stack(x_enc).to(dtype=torch.float32)
    x1 = torch.stack(y_enc).to(dtype=torch.float32)
    print(f"mse before OT is {F.cosine_similarity(x0, x1).mean()}")
    x0, x1, y0, y1 = ot_plan.sample_plan_with_labels(
        x0 / torch.norm(x0, p=2, dim=-1, keepdim=True),
        x1 / torch.norm(x1, p=2, dim=-1, keepdim=True),
        torch.arange(0, 16),
        torch.arange(0, 16),
        replace=False,
    )
    print(f"mse after OT is {F.cosine_similarity(x0, x1).mean()}")
    ot_wer = cer([x_text[i] for i in y0], [y_text[i] for i in y1])
    print(f"ot wer is {ot_wer}")
    breakpoint()
