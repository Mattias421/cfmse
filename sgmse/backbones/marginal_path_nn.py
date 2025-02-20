from .ncsnpp_utils import layers, layerspp
import torch.nn as nn
import torch

from .shared import BackboneRegistry

get_act = layers.get_act
default_initializer = layers.default_init


@BackboneRegistry.register("marginal_path_nn")
class MarginalPathNN(nn.Module):
    """NN to predict weights and sigma of marginal path"""

    def __init__(
        self,
        nf=128,
        nonlinearity="swish",
        fourier_scale=16,
        embedding_type="fourier",
        two_weights=False,
        **unused_kwargs,
    ):
        super().__init__()
        self.act = get_act(nonlinearity)

        self.nf = nf = nf
        self.embedding_type = embedding_type = embedding_type.lower()

        assert embedding_type in ["fourier", "positional"]

        modules = []
        # timestep/noise_level embedding
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf, scale=fourier_scale
                )
            )
            embed_dim = 2 * nf
        elif embedding_type == "positional":
            embed_dim = nf
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        modules.append(nn.Linear(embed_dim, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(nf * 4, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        modules.append(nn.Linear(nf * 4, 1))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        modules.append(nn.Linear(nf * 4, 1))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        self.two_weights = two_weights

        if self.two_weights:
            modules.append(nn.Linear(nf * 4, 1))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        self.all_modules = nn.ModuleList(modules)

    def forward(self, t):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        weight_1 = torch.sigmoid(modules[m_idx](temb))
        m_idx += 1

        sigma_t = torch.sigmoid(modules[m_idx](temb))
        m_idx += 1

        if self.two_weights:
            weight_2 = torch.sigmoid(modules[m_idx](temb))
            m_idx += 1

            return weight_1[:, 0], weight_2[:, 0], sigma_t[:, 0]
        else:
            return weight_1[:, 0], sigma_t[:, 0]
