import json
import unittest
from pathlib import Path

import torch

from sgmse.backbones.ncsnpp_v2 import NCSNpp_v2
from sgmse.model import ScoreModel, stratified_file_pairs


ROOT = Path(__file__).resolve().parents[2]


class DummySDE:
    T = 1.0


class TimePolicy:
    _sample_time = ScoreModel._sample_time
    _t1_probability = ScoreModel._t1_probability

    def __init__(self, policy, epoch=0):
        self.time_sampling = policy
        self.fixed_time = 1.0
        self.t_eps = 0.03
        self.sde = DummySDE()
        self.current_epoch = epoch
        self.t1_probability_start = 0.0
        self.t1_probability_end = 1.0
        self.t1_anneal_epochs = 10


class CaptureDNN(torch.nn.Module):
    def forward(self, x, y, t):
        self.inputs = (x, y, t)
        return x


class ExperimentInfrastructureTest(unittest.TestCase):
    def test_manifest_inventory_and_allocation(self):
        manifest = json.loads((ROOT / "xps" / "experiments.json").read_text())
        experiments = manifest["experiments"]
        self.assertEqual(len(experiments), 22)
        self.assertEqual(len({item["id"] for item in experiments}), 22)
        counts = {
            server: sum(item["server"] == server for item in experiments)
            for server in ("stanage", "mimas", "phoebe")
        }
        self.assertEqual(counts, {"stanage": 18, "mimas": 2, "phoebe": 2})

    def test_fixed_time_policy(self):
        policy = TimePolicy("fixed")
        actual = policy._sample_time(32, torch.device("cpu"))
        self.assertTrue(torch.equal(actual, torch.ones(32)))

    def test_annealed_time_policy_reaches_endpoint(self):
        policy = TimePolicy("annealed_t1", epoch=10)
        actual = policy._sample_time(32, torch.device("cpu"))
        self.assertTrue(torch.equal(actual, torch.ones(32)))
        self.assertEqual(policy._t1_probability(), 1.0)

    def test_network_conditioning_can_be_removed(self):
        model = ScoreModel.__new__(ScoreModel)
        torch.nn.Module.__init__(model)
        model.backbone = "ncsnpp_v2"
        model.dnn = CaptureDNN()
        model.condition_on_noisy = False
        model.condition_on_time = False
        model.network_scaling = None
        model.loss_type = "flow_matching"
        model.c_in = "1"

        x = torch.complex(torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4))
        y = torch.complex(torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4))
        t = torch.tensor([0.2, 0.8])
        model(x, y, t)
        _, network_y, network_t = model.dnn.inputs
        self.assertTrue(torch.equal(network_y, torch.zeros_like(y)))
        self.assertTrue(torch.equal(network_t, torch.ones_like(t)))

    def test_time_free_backbone_removes_embedding_parameters(self):
        kwargs = {
            "nf": 8,
            "ch_mult": (1, 1),
            "num_res_blocks": 1,
            "attn_resolutions": (),
        }
        timed = NCSNpp_v2(condition_on_time=True, **kwargs)
        time_free = NCSNpp_v2(condition_on_time=False, **kwargs)
        timed_parameters = sum(parameter.numel() for parameter in timed.parameters())
        time_free_parameters = sum(
            parameter.numel() for parameter in time_free.parameters()
        )
        self.assertTrue(timed.condition_on_time)
        self.assertFalse(time_free.condition_on_time)
        self.assertLess(time_free_parameters, timed_parameters)

    def test_validation_selection_is_speaker_balanced(self):
        clean = [f"/clean/p226_{index:03}.wav" for index in range(30)]
        clean += [f"/clean/p287_{index:03}.wav" for index in range(40)]
        noisy = [path.replace("/clean/", "/noisy/") for path in clean]
        selected = stratified_file_pairs(clean, noisy, 20)
        speakers = [
            Path(clean_file).stem.split("_", 1)[0] for clean_file, _ in selected
        ]
        self.assertEqual(speakers.count("p226"), 10)
        self.assertEqual(speakers.count("p287"), 10)
        self.assertGreater(int(Path(selected[-2][0]).stem.split("_")[1]), 20)


if __name__ == "__main__":
    unittest.main()
