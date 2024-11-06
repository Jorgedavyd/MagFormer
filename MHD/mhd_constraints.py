from typing import List
import torch
from datetime import timedelta
from torch import nn, Tensor
from torch.utils.data import DataLoader
from lightorch.nn.criterions import LighTorchLoss, Loss
from scipy.constants import mu_0, epsilon_0, c
import torch.nn.functional as F
from math import sqrt

"""
Physics informed loss
"""

class calc:
    @staticmethod
    def grad(F: Tensor, position: Tensor, edge_order=1):

        df_dx = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 0], dim=-1, edge_order=edge_order
        )
        df_dy = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 1], dim=-1, edge_order=edge_order
        )
        df_dz = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 2], dim=-1, edge_order=edge_order
        )

        return torch.cat([df_dx, df_dy, df_dz], dim=-1)

    @staticmethod
    def div(F: Tensor, position: Tensor, edge_order=1) -> Tensor:
        _, _, dimensions = F.shape
        return torch.sum(torch.stack([torch.gradient(F[:,:,dim], spacing = (position[:,dim],), dim = -1, edge_order = edge_order)[0] for dim in range(dimensions)], dim = -1), dim = -1)

    @staticmethod
    def rot(F: Tensor, position: Tensor, edge_order=1):
        assert len(F.shape) == 3

        dFz_dy = torch.gradient(
            F[:, :, 2], spacing=position[:, :, 1], dim=-1, edge_order=edge_order
        )

        dFy_dz = torch.gradient(
            F[:, :, 1], spacing=position[:, :, 2], dim=-1, edge_order=edge_order
        )

        dFx_dz = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 2], dim=-1, edge_order=edge_order
        )
        dFz_dx = torch.gradient(
            F[:, :, 2], spacing=position[:, :, 0], dim=-1, edge_order=edge_order
        )

        dFy_dx = torch.gradient(
            F[:, :, 1], spacing=position[:, :, 0], dim=-1, edge_order=edge_order
        )
        dFx_dy = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 1], dim=-1, edge_order=edge_order
        )

        return torch.stack([dFz_dy - dFy_dz, dFx_dz - dFz_dx, dFy_dx - dFx_dy], dim=-1)

    @staticmethod
    def dF_dt(F: Tensor, step_size: timedelta, edge_order=1):
        return torch.gradient(
            F, spacing=(step_size.total_seconds(),), dim=-2, edge_order=edge_order
        )[0]

    @staticmethod
    def conv_oper(
        F: Tensor, A: Tensor, position: Tensor, edge_order=1
    ):

        dFx_dx = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 0], dim=-1, edge_order=edge_order
        )
        dFx_dy = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 1], dim=-1, edge_order=edge_order
        )
        dFx_dz = torch.gradient(
            F[:, :, 0], spacing=position[:, :, 2], dim=-1, edge_order=edge_order
        )

        dFy_dx = torch.gradient(
            F[:, :, 1], spacing=position[:, :, 0], dim=-1, edge_order=edge_order
        )
        dFy_dy = torch.gradient(
            F[:, :, 1], spacing=position[:, :, 1], dim=-1, edge_order=edge_order
        )
        dFy_dz = torch.gradient(
            F[:, :, 1], spacing=position[:, :, 2], dim=-1, edge_order=edge_order
        )

        dFz_dx = torch.gradient(
            F[:, :, 2], spacing=position[:, :, 0], dim=-1, edge_order=edge_order
        )
        dFz_dy = torch.gradient(
            F[:, :, 2], spacing=position[:, :, 1], dim=-1, edge_order=edge_order
        )
        dFz_dz = torch.gradient(
            F[:, :, 2], spacing=position[:, :, 2], dim=-1, edge_order=edge_order
        )

        out_x = Tensor(
            [A[:, :, 0] * dFx_dx + A[:, :, 1] * dFx_dy + A[:, :, 2] * dFx_dz]
        )
        out_y = Tensor(
            [A[:, :, 0] * dFy_dx + A[:, :, 1] * dFy_dy + A[:, :, 2] * dFy_dz]
        )
        out_z = Tensor(
            [A[:, :, 0] * dFz_dx + A[:, :, 1] * dFz_dy + A[:, :, 2] * dFz_dz]
        )

        return torch.cat([out_x, out_y, out_z], dim=-1)

def gamma(v):
    return (sqrt(1 - (v**2 / c**2))) ** -1

class IdealMHD(nn.Module):
    def __init__(
            self,
            step_size: timedelta = timedelta(minutes = 5),
            edge_order: int = 1
    ) -> None:
        super().__init__()
        self.step_size = step_size
        self.edge_order = edge_order

    class GaussLawMagnetismConstraint(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                labels = self.__class__.__name__,
                factors = {
                    self.__class__.__name__: factor
                }
            )

        def forward(self, B: Tensor, r: Tensor) -> Tensor:
            """
            B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
            r: position of the spacecraft | torch.tensor | (batch_size, sequence_length, 3)
            """
            return calc.div(B, r, self.edge_order).mean()

    class GaussLawElectrostaticConstraint(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                labels = self.__class__.__name__,
                factors = {self.__class__.__name__: factor}
            )
        def forward(
            self, E: Tensor, sigma: float, r: Tensor
        ):
            """
            E: electric field | torch.tensor | (batch_size, sequence_length, 3)
            sigma: charge density | torch.tensor | (batch_size, sequence_length)
            r: position of the spacecraft | torch.tensor | (batch_size, sequence_length, 3)
            """
            return (calc.div(E, r, self.edge_order) - (sigma / epsilon_0)).mean()

    class DriftVelocity(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                self.__class__.__name__,
                {self.__class__.__name__: factor}
            )
        def forward(self, B: Tensor, E: Tensor, v_drift: Tensor) -> Tensor:
            """
            B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
            E: electric field | torch.tensor | (batch_size, sequence_length, 3)
            v_drift: drift velocity | torch.tensor | (batch_size, sequence_length, 3)
            """
            return (
                (torch.cross(E, B, dim=-1) / torch.sum(B, dim=-1, keepdim=True)) - v_drift
            ).mean()

    class ContinuityConstraint(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                self.__class__.__name__,
                {self.__class__.__name__: factor}
            )
        def forward(self, rho: Tensor, v: Tensor, r: Tensor) -> Tensor:
            """
            rho: mass density | torch.tensor | (batch_size, sequence_length)
            v: mass velocity field | torch.tensor | (batch_size, sequence_length, 3)
            """
            return ((calc.dF_dt(rho, self.step_size) + calc.div(rho * v, r)) ** 2).mean()

    class StateConstraint(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                self.__class__.__name__,
                {self.__class__.__name__: factor}
            )
        def forward(self, p: Tensor, rho: Tensor, N=3) -> Tensor:
            """
            rho: mass density | torch.tensor | (batch_size, sequence_length)
            p: pressure | torch.tensor | (batch_size, sequence_length)
            """
            rho = rho.unsqueeze(-1)
            p = p.unsqueeze(-1)
            gamma = (N + 2) / 2
            return ((calc.dF_dt(p / (rho) ** gamma, self.step_size)) ** 2).mean()

    class Ohm(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                self.__class__.__name__,
                {self.__class__.__name__: factor}
            )
        def forward(self, E: Tensor, v: Tensor, B: Tensor) -> Tensor:
            """
            B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
            E: electric field | torch.tensor | (batch_size, sequence_length, 3)
            v: mass velocity field | torch.tensor | (batch_size, sequence_length, 3)
            """
            return ((E + torch.cross(v, B, dim=-1)) ** 2).mean()

    class Induction(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                labels = self.__class__.__name__,
                factors = {self.__class__.__name__: factor}
            )
        def forward(
            self, B: Tensor, v: Tensor, r: Tensor
        ):
            """
            B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
            v: mass velocity field | torch.tensor | (batch_size, sequence_length, 3)
            """
            return (
                (
                    calc.dF_dt(B, self.step_size, self.edge_order)
                    - calc.rot(torch.cross(v, B, dim=-1), r, self.edge_order)
                )
                ** 2
            ).mean()

    class MotionConstraint(LighTorchLoss):
        def __init__(self, factor: float = 1.) -> None:
            super().__init__(
                self.__class__.__name__,
                {self.__class__.__name__: factor}
            )
        def forward(self, B, J, r) -> Tensor:
            """
            J: current density | torch.tensor | (batch_size, sequence_length)
            B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
            """
            return (
                (
                    torch.cross(J, B, dim=-1)
                    - calc.conv_oper(B, B, r, self.edge_order) / mu_0
                    + calc.grad((torch.norm(B, 2, -1, True) ** 2) / (2 * mu_0), r, self.edge_order)
                )
                ** 2
            ).mean()


def compute_weights(dataloader):
    bounded = torch.bincount(
        torch.cat([batch["dst"].view(-1) for batch in dataloader], dim=-1)
    )
    return torch.softmax(1 / bounded)


class FineTuningPriorityCriterion(LighTorchLoss):
    def __init__(
        self,
        dataloader: DataLoader,
        factor: float
    ):
        super().__init__(
            labels = self.__class__.__name__,
            factors = {
                self.__class__.__name__: factor
            }
        )
        self.base_criterion = F.cross_entropy
        self.weights = compute_weights(dataloader)

    def forward(self, **kwargs) -> Tensor:
        losses = self.base_criterion(kwargs['input'], kwargs['target'], reduction="none").mean(dim=(1, 2))
        seq_dst, _ = torch.max(kwargs['target'], dim=1)
        return torch.mean(losses * self.weights[seq_dst])


"""
Forcing Criterion
Based on datasets diff
"""


def weight_scaler(dataloader):
    batchwise_loss = []
    for batch in dataloader:
        batchwise_loss.append(
            F.mse_loss(batch["l1"], batch["l2"], reduction="none").mean(dim=(1, 2))
        )
    loss = torch.cat(batchwise_loss, dim=-1)
    loss_max = loss.max()
    loss_min = loss.min()
    return lambda J: (J - loss_max) / (loss_max - loss_min)


class ForcingCriterion(LighTorchLoss):
    def __init__(
        self,
        dataloader: DataLoader,
        factor: float
    ):
        super().__init__(
            labels = self.__class__.__name__,
            factors = {
                self.__class__.__name__: factor
            }
        )
        self.base_criterion = F.mse_loss
        self.weight_scaler = weight_scaler(dataloader)

    def forward(self, **kwargs) -> Tensor:
        noise_loss: Tensor = NoiseLoss.compute(kwargs['l1'], kwargs['l2'])
        # B,1
        out = self.base_criterion(kwargs['l1_out'], kwargs['l2_out'], reduction="none").mean(dim=(1, 2))
        # (B,1 * B,1).mean()
        return torch.mean(out * noise_loss.apply_(self.weight_scaler))


class NoiseLoss:
    @staticmethod
    def compute(l1, l2):
        # (B,S,I) x (B,S,I) -> (B,1)
        return F.mse_loss(l1, l2, reduction="none").mean(dim=(1, 2))

class MainCriterion(Loss):
    def __init__(self, valid_criterions: List[str], factors: List[float]) -> None:
        assert (len(valid_criterions) == len(factors)), "Must have the same number of criterions and factors"
        self.base_criterion = IdealMHD()
        self.valid_criterions: List[str] = valid_criterions
        super().__init__(
            *[getattr(self.base_criterion, name)(factor) for name, factor in zip(self.valid_criterions, factors)]
        )
