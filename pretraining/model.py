from lightorch.training.supervised import Module
from lightorch.nn.sequential import LSTM, GRU
from torch import nn, Tensor
import torch

"""
# Model idea

## 1. Temporal alignment and concatenation
Temporal Convolutions  -> concat feature space

## Pretraining
- Autoencoder reconstruction

## Model loading


"""

class CnnInterpolator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x

class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x

class Encoder(nn.Sequential):
    def __init__(self) -> None:
        super().__init__()


class Decoder(nn.Sequential):
    def __init__(self) -> None:
        super().__init__()

class Main(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bakcbone_sdo = CNN_Model()
        self.bakcbone_lasco = CNN_Model()
        self.fc_sdo = TimeAlignementModel()
        self.fc_lasco = TimeAlignementModel()

        if self.path is not None:
            self.load_state_dict(torch.load(self.path))

    def forward(self, sdo_in: Tensor, lasco_in: Tensor) -> Tensor:
        out_sdo = self.backbone_sdo(sdo_in)
        out_lasco = self.backbone_lasco(lasco_in)
        out_sdo = self.fc_sdo(out_sdo) ## (batch_size, temporal_alignment, input_size)
        out_lasco = self.fc_lasco(out_lasco) ## (batch_size, temporal_alignment, input_size)
        return torch.cat([out_sdo, out_lasco], dim = -1)

