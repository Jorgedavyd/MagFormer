from torch import nn, Tensor
import torch
from lightorch.nn.sequential import *
from lightorch.nn.criterions import LightorchLoss
from lightorch.training.supervised import Module

class CNN_Model(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class TimeAlignementModel(nn.Module):
    def __init__(self, input: Tensor) -> None:
        super().__init__()
        self.model =
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class Main(nn.Module):
    def __init__(self, input: Tensor) -> None:
        super().__init__()
        self.bakcbone_sdo = CNN_Model()
        self.bakcbone_lasco = CNN_Model()
        self.fc_sdo = TimeAlignementModel()
        self.fc_lasco = TimeAlignementModel()

    def forward(self, sdo_in: Tensor, lasco_in: Tensor) -> Tensor:
        out_sdo = self.backbone_sdo(sdo_in)
        out_lasco = self.backbone_lasco(lasco_in)
        out_sdo = self.fc_sdo(out_sdo).unsqueeze(1)
        out_lasco = self.fc_lasco(out_lasco).unsqueeze(1)
        return torch.cat([out_sdo, out_lasco], dim = 1)

class Decoder(nn.Module):
    def __init__(self, input: int) -> None:
        super().forward(self)

    def forward(self, x: Tensor) -> Tensor:
        return

class Autoencoder(nn.Module):
    def __init__(self, input: int, ) -> None:
        super().__init__()
        self.encoder = Main()
        self.decoder = Decoder()
    def forward(self, sdo_in: Tensor, lasco_in: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.encoder(sdo_in, lasco_in)
        out_sdo, out_lasco = self.decoder(out)
        return out_sdo, out_lasco


