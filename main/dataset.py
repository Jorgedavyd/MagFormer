# TODO: Define dataset for OMNI and the new directory from pretraining
class ReconstructionDataset(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
