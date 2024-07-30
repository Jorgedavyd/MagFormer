from torch import nn, Tensor
import torch
from lightorch.nn.sequential import *
from lightorch.nn.criterions import LightorchLoss
from lightorch.training.supervised import Module


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




