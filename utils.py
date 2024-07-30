from lightorch.training.supervised import Module

class General(Module):
    def __init__(self, model: nn.Module, criterion: LightorchLoss, **hparams) -> None:
        super().__init__(**hparams)
        self.model = model
        self.criterion = criterion
    def forward(self, *args) -> Tensor:
        return self.model(*args)
