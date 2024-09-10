from torch import nn, Tensor
from lightorch.nn.sequential.residual import ResidualLSTM, ResidualGRU
from lightorch.training.supervised import Module
from typing import Dict

input_size: int = 6
out_size: int = 4

class Model1(ResidualLSTM):
    def __init__(self, hidden_size: int, layers: int) -> None:
        super().__init__(input_size, hidden_size, layers)
        self.fc = nn.Linear(hidden_size, out_size)
    def forward(self, x: Tensor) -> None:
        return self.fc(super().forward(self, x))

class Model2(ResidualGRU):
    def __init__(self, hidden_size: int, layers: int) -> None:
        super().__init__(input_size, hidden_size, layers)
        self.fc = nn.Linear(hidden_size, out_size)
    def forward(self, x: Tensor) -> None:
        return self.fc(super().forward(self, x))

class Model(Module):
    def __init__(self, **hparams) -> None:
        super().__init__(**hparams)
        self.model = Model1(hparams['hidden_size'], hparams['layers'])

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss_forward(self, batch:Tensor, idx: int) -> Dict[str, Tensor | float]:
        out = self.model(batch[0])
        return dict(
            label = out,
            target = batch[1],
        )
