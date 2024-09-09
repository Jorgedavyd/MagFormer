from main.model import Model as BackBoneModel
from pretraining.model import Model as ImageEmbeddingModel
from synthetic.model import Model as TabularEmbeddingModel
from reconstruction.model import Model as ReconstructionModel
from torch import nn, Tensor

class MagFormer(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            ReconstructionModel(),
            TabularEmbeddingModel(),
            ImageEmbeddingModel(),
            BackBoneModel(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)

