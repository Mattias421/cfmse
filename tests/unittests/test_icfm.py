import pytest
import torch
from argparse import ArgumentParser

# Assuming ICFM, SDE, ConditionalFlowMatcher, pad_t_like_x are defined in a module named 'your_module'
from sgmse.sdes import ICFM


# Mock SDE class for inheritance testing
class SDE:
    def __init__(self, N):
        self.N = N


@pytest.fixture
def icfm_instance():
    return ICFM(sigma=0.05, N=30, sampler_type="ode")


def test_add_argparse_args():
    parser = ArgumentParser()
    parser = ICFM.add_argparse_args(parser)
    args = parser.parse_args(["--sigma", "0.1", "--N", "50", "--sampler_type", "pc"])
    assert args.sigma == 0.1
    assert args.N == 50
    assert args.sampler_type == "pc"


def test_init(icfm_instance):
    assert icfm_instance.cfm.sigma == 0.05
    assert icfm_instance.N == 30
    assert icfm_instance.sampler_type == "ode"


def test_copy(icfm_instance):
    copied_icfm = icfm_instance.copy()
    assert copied_icfm.cfm.sigma == icfm_instance.cfm.sigma
    assert copied_icfm.N == icfm_instance.N
    assert copied_icfm.sampler_type == icfm_instance.sampler_type
    assert copied_icfm is not icfm_instance  # Ensure it's a new object


def test_T(icfm_instance):
    assert icfm_instance.T == 1


def test_mean(icfm_instance):
    batch_size = 10
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randn(batch_size, 3, 32, 32)

    # Test t = 1
    t1 = torch.ones(batch_size)
    mean1 = icfm_instance._mean(x, y, t1)
    assert mean1.shape == x.shape
    assert torch.allclose(mean1, x, atol=1e-3)  # Check if mean is close to y when t=1

    # Test t = 0
    t0 = torch.zeros(batch_size)
    mean0 = icfm_instance._mean(x, y, t0)
    assert mean0.shape == x.shape
    assert torch.allclose(mean0, y, atol=1e-3)  # Check if mean is close to x0 when t=0

    # Test intermediate t
    t_intermediate = 0.5 * torch.ones(batch_size)
    mean_intermediate = icfm_instance._mean(x, y, t_intermediate)

    assert mean_intermediate.shape == x.shape
    assert not torch.allclose(
        mean_intermediate, y, atol=1e-3
    )  # Should not be close to y
    assert not torch.allclose(
        mean_intermediate, x, atol=1e-3
    )  # Should not be close to x0
    # Should be in the middle of x0 and y
    assert torch.allclose(mean_intermediate, 0.5 * (x + y), atol=1e-3)


def test_std(icfm_instance):
    t = torch.rand(10)
    std = icfm_instance._std(t)
    assert std.shape == t.shape
    # Check that sigma is computed as expected and that padding works
    sigma_t = icfm_instance.cfm.compute_sigma_t(1 - t)
    assert torch.all(std == sigma_t)


def test_marginal_prob(icfm_instance):
    x0 = torch.randn(10, 3, 32, 32)
    y = torch.randn(10, 3, 32, 32)
    t = torch.rand(10)
    mean, std = icfm_instance.marginal_prob(x0, y, t)
    assert mean.shape == x0.shape
    assert std.shape == t.shape


def test_prior_sampling(icfm_instance):
    shape = (10, 3, 32, 32)
    y = torch.randn(10, 3, 32, 32)
    x_T = icfm_instance.prior_sampling(shape, y)
    assert x_T.shape == shape


def test_prior_sampling_warning(icfm_instance):
    shape = (5, 3, 32, 32)  # different shape
    y = torch.randn(10, 3, 32, 32)
    with pytest.warns(
        UserWarning, match="Target shape .* does not match shape of y .*"
    ):
        x_T = icfm_instance.prior_sampling(shape, y)
    assert x_T.shape == y.shape
