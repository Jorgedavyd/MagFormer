from lightorch.nn.criterions import LagrangianFunctional

class Loss(LagrangianFunctional):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
