from sgmse.backbones.marginal_path_nn import MarginalPathNN

import pytest
import torch


@pytest.fixture
def marginal_path_nn_instance():
    return MarginalPathNN()


def test_forward(marginal_path_nn_instance):
    t = 0.5 * torch.ones(2)
    a, b, s = marginal_path_nn_instance(t)

    assert a.shape[0] == 2
    assert b.shape[0] == 2
    assert s.shape[0] == 2
