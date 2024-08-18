from torch import nn, Tensor
import torch
from lightorch.nn.sequential import *
from lightorch.nn.criterions import LightorchLoss
from lightorch.training.supervised import Module

"""
# Model idea

## 1. Temporal alignment and concatenation
Temporal Convolutions  -> concat feature space

## Pretraining
- Autoencoder reconstruction

## Model loading


"""
class TemportalCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:

        return out

class TimeAlignementModel(nn.Module):
    def __init__(self, input: Tensor) -> None:
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Main(nn.Module):
    def __init__(self, input: Tensor, path: str | None = None) -> None:
        super().__init__()
        self.bakcbone_sdo = CNN_Model()
        self.bakcbone_lasco = CNN_Model()
        self.fc_sdo = TimeAlignementModel()
        self.fc_lasco = TimeAlignementModel()

        if self.path is not None:
            self.load_state_dict(torch.load(path))

    def forward(self, sdo_in: Tensor, lasco_in: Tensor) -> Tensor:
        out_sdo = self.backbone_sdo(sdo_in)
        out_lasco = self.backbone_lasco(lasco_in)
        out_sdo = self.fc_sdo(out_sdo) ## (batch_size, temporal_alignment, input_size)
        out_lasco = self.fc_lasco(out_lasco) ## (batch_size, temporal_alignment, input_size)
        return torch.cat([out_sdo, out_lasco], dim = -1)

class Decoder(nn.Module):
    def __init__(self, input: int) -> None:
        super().forward(self)
    def forward(self, x: Tensor) -> Tensor:
        return out

class Autoencoder(nn.Module):
    def __init__(self, input: int, ) -> None:
        super().__init__()
        self.encoder = Main()
        self.decoder = Decoder()
    def forward(self, sdo_in: Tensor, lasco_in: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.encoder(sdo_in, lasco_in)
        out_sdo, out_lasco = self.decoder(out)
        return out_sdo, out_lasco


