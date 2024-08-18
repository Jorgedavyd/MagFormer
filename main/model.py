from lightorch.nn import criterions
from torch import nn, Tensor
from ..pretraining import Main
from lightorch.nn.transformer import CrossTransformer

class MagFormer(CrossTransformer):
    def __init__(self, LASCO_SDO_model_path: str,  ) -> None:
        super().__init__()
        self.first_path = LASCO_SDO_model_path
        self.first_model = Main(..., self.first_path)

    def forward(self, L1_data: Tensor, LASCO: Tensor, SDO: Tensor) -> Tensor:
        image_input = self.first_model(LASCO, SDO)
        return out
