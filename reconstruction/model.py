from torch import nn, Tensor
from typing import Dict, List, Tuple, Union
import torch
from lightorch.nn.sequntial.residual import *
from lightorch.nn.criterions import LightorchLoss
from lightorch.training.supervised import Module

DecoderParams = Dict[str, int | float | List[int|float]]
EncoderParams = Dict[str, int | float | List[int|float]]

class M1(nn.Module):
    def __init__(self,*args,**kwargs ) -> None:
        super().__init__()
        self.res_lstm = ResidualLSTM(*args, **kwargs)
    @torch.jit.script
    def forward(self, x: Tensor) -> Tensor:
        return self.res_lstm(x)

class M2(nn.Module):
    def __init__(self,*args,**kwargs ) -> None:
        super().__init__()
        self.res_lstm = ResidualGRU(*args, **kwargs)
    @torch.jit.script
    def forward(self, x: Tensor) -> Tensor:
        return self.res_lstm(x)

class Encoder(nn.Module):
    def __init__(self, encoderParams: EncoderParams) -> None:
        super().__init__()
        self.args = encoderParams
        self.init_params()
    def init_params(self) -> None:
        # TODO
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Decoder(nn.Sequential):
    def __init__(self, decoderParams: DecoderParams) -> None:
        super().__init__()
        self.args = decoderParams
        self.init_params()
    def init_params(self) -> None:
        #TODO
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class VAE(nn.Module):
    def __init__(self, encoderParams: EncoderParams, decoderParams: DecoderParams) -> None:
        super().__init__()
        self.encoder = Encoder(encoderParams)
        self.decoder = Decoder(decoderParams)

  def reparametrization(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        out = self.fc(x)
        mu, log_variance = torch.chunk(out, 2, -1)
        log_variance = torch.clamp(log_variance, -30, 20)
        std = log_variance.exp().sqrt()
        out = mu + torch.randn(b, self.channels[-1], self.last_height, self.last_width) * std
        return out, mu, log_variance

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        out = self.encoder(x)
        out, mu, log_variance = self.reparametrization(out)
        out = self.decoder(out)
        return out, mu, log_variance

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.model(x)

    def loss_forward(self, batch: Tensor, idx: int) -> Dict[str, Union[Tensor, float]]:
        out, mu, log = self(batch[0])
        return dict(
            input = out,
            target = batch[1],
            mu = mu,
            log = log
        )


