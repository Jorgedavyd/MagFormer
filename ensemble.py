from main.model import Model
from pretraining.model import Model
from synthetic.model import Model
from reconstruction.model import Model, forward

from torch import nn, Tensor

# TODO: Setup model with loaded weights

class MagFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return
