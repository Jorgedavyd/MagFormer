from main.model import Model as BackBoneModel
from pretraining.model import Model as ImageEmbeddingModel
from synthetic.model import SyntheticDataModelArchitecture as TabularEmbeddingModel
from reconstruction.model import Model as ReconstructionModel
from torch import nn

class MagFormer(nn.Sequential):
    def __init__(self, **hparams) -> None:
        super().__init__(
            ReconstructionModel(**hparams),
            TabularEmbeddingModel(**hparams),
            ImageEmbeddingModel(**hparams),
            BackBoneModel(**hparams),
        )
