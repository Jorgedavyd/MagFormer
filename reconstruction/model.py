import torch
from torch import nn, Tensor
from typing import Dict, Tuple, Sequence
from lightorch.training.supervised import Module
from lightorch.nn.functional import residual_connection
from .utils import batch_data

class SingleLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: bool,
        pooling: int | None = None,
        dropout: float | None = None
    ) -> None:
        super().__init__(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        )
        if activation:
            self.add_module('activation', nn.ReLU())

        if pooling is not None:
            assert (pooling > 0)
            self.add_module('pooling_layer', nn.MaxPool1d(pooling))

        self.add_module('batch_norm', nn.BatchNorm1d(out_channels))

        if dropout is not None:
            self.add_module('dropout', nn.Dropout(dropout))

class ResidualLayer(SingleLayer):
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, activation: bool, pooling: int | None = None, dropout: float | None = None) -> None:
        super().__init__(channels, channels, kernel_size, stride, padding, activation, pooling, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return residual_connection(x, lambda x: super().forward(x))

class Encoder(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            lambd: int = 2,
            architecture: Sequence[int] = [0, 0, 1, 1],
            pooling_layers: Sequence[int] = [1],
            dropout: float | None = None
    ) -> None:
        super().__init__()
        assert (len(architecture) == len(pooling_layers)), "Not valid neither pooling nor architecture"

        idx: int = 0
        while idx < len(architecture):
            pooling: int = pooling_layers[idx]
            flag: int = architecture[idx]
            if flag:
                self.add_module(
                    f'single_layer_{idx}',
                    SingleLayer(in_channels*(lambd**idx), in_channels*(lambd**(idx+1)), 3, 1, 1, True, pooling, dropout)
                )
                idx += 1
            else:
                self.add_module(
                    f'res_layer_{idx}',
                    ResidualLayer(in_channels*(lambd**idx), 3, 1, 1, True, pooling, dropout)
                )
            idx += 1

class Decoder(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            lambd: int = 2,
            architecture: Sequence[int] = [0, 0, 1, 1],
            pooling_layers: Sequence[int] = [1],
            dropout: float | None = None
    ) -> None:
        super().__init__()
        assert (len(architecture) == len(pooling_layers)), "Not valid neither pooling nor architecture"

        idx: int = len(architecture) - 1
        while idx >= 0:
            pooling: int = pooling_layers[idx]
            flag: int = architecture[idx]
            if flag:
                self.add_module(
                    f'single_layer_{idx}',
                    SingleLayer(in_channels*(lambd**(idx + 1)), in_channels*(lambd**idx), 3, 1, 1, True, pooling, dropout)
                )
            else:
                self.add_module(
                    f'res_layer_{idx}',
                    ResidualLayer(in_channels*(lambd**idx), 3, 1, 1, True, pooling, dropout)
                )
            idx -= 1

class VAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        channels: int,
        lambd: int,
        architecture: Sequence[int] = [0, 0, 1, 1],
        pooling_layers: Sequence[int] = [1, 1, 1, 1],
        dropout: float | None = None
    ) -> None:
        super().__init__()
        self.channels: int = channels
        self.input_size: int = input_size
        self.encoder = Encoder(channels, lambd, architecture, pooling_layers, dropout)
        self.decoder = Decoder(channels, lambd, architecture, pooling_layers, dropout)
        out_channel_size: int = channels*(lambd**(len(architecture)))
        self.fc = nn.Linear(out_channel_size, 2*out_channel_size)

    def reparametrization(self, x: Tensor) -> Tuple[Tensor, ...]:
        b, c, l = x.shape
        out = self.fc(x.view(b, -1))
        mu, log_variance = torch.chunk(out, 2, -1)
        log_variance = torch.clamp(log_variance, -30, 20)
        std = log_variance.exp().sqrt()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z.view(b, c, l), mu, log_variance

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        encoded = self.encoder(x)
        z, mu, log_variance = self.reparametrization(encoded)
        decoded = self.decoder(z)
        return decoded, mu, log_variance

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = VAE(
            hparams['input_size'],
            hparams['channels'],
            hparams['lambd'],
            hparams['architecture'],
            hparams['pooling_layers'],
            hparams['dropout'],
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.model(x)

    def loss_forward(self, batch: Tensor, idx: int) -> Dict[str, Tensor]:
        out, mu, log = self(batch[0])
        return {
            "input": out,
            "target": batch[1],
            "mu": mu,
            "log": log,
            **batch_data(out)
        }
