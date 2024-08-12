from lightorch.nn.criterions import LighTorchLoss
from torch import nn, Tensor

from mhd_constraints import calc, PIConstraint

class criterion(LighTorchLoss):
    def __init__(self, *factors) -> None:
        assert(len(factors) == 8), "Not valid amount of parameters"
        labels = [
            'GaussLawMagnetismConstraint',
            'GaussLawElectrostaticConstraint',
            'DriftVelocity',
            'ContinuityConstraint',
            'StateConstraint',
            'LorentzConstraint',
            'AmpereFaradayConstraint',
            'MotionConstraint',
        ]
        super().__init__(
                labels = labels,
                factors = {name: factor for name, factor in zip(labels, factor),
        )
    def forward(self, x: Tensor) -> Tensor:
        gauss_mag: Tensor =
