from torch import nn, Tensor
import torch
from lightorch.nn.sequential import *
from lightorch.training.supervised import Module

class Model1(Module):
    def __init__(self, input_size: int, layers: int, **hparams) -> None:
        super().__init__(**hparams)
        self.model = ResidualLSTM(input_size, layers)
        self.criterion = #define criterion
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Model2(Module):
    def __init__(self, input_size: int, layers: int, **hparams) -> None:
        super().__init__(**hparams)
        self.model = ResidualGRU(input_size, layers)
        self.criterion = #define criterion
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
