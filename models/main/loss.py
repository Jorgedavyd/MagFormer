# General utils
import torch
from datetime import timedelta
from torch import nn
from scipy.constants import mu_0, epsilon_0, c
import torch.nn.functional as F
from math import sqrt

"""
Physics informed loss
"""


class calc:
    def grad(f: torch.Tensor, position: torch.Tensor, edge_order=1):

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
    def div(F: torch.Tensor, position: torch.Tensor, edge_order=1):
        _, _, dimensions = F.shape
        partial = []
        for dim in range(dimensions):
            partial.append(
                torch.gradient(
                    F[:, :, dim],
                    spacing=(position[:, dim],),
                    dim=-1,
                    edge_order=edge_order,
                )[0]
            )
        return sum(partial)

    @staticmethod
    def rot(F: torch.Tensor, position: torch.Tensor, edge_order=1):
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
    def dF_dt(F: torch.Tensor, step_size: timedelta, edge_order=1):
        return torch.gradient(
            F, spacing=(step_size.total_seconds(),), dim=-2, edge_order=edge_order
        )[0]

    @staticmethod
    def conv_oper(
        F: torch.Tensor, A: torch.Tensor, position: torch.Tensor, edge_order=1
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

        out_x = torch.Tensor(
            [A[:, :, 0] * dFx_dx + A[:, :, 1] * dFx_dy + A[:, :, 2] * dFx_dz]
        )
        out_y = torch.Tensor(
            [A[:, :, 0] * dFy_dx + A[:, :, 1] * dFy_dy + A[:, :, 2] * dFy_dz]
        )
        out_z = torch.Tensor(
            [A[:, :, 0] * dFz_dx + A[:, :, 1] * dFz_dy + A[:, :, 2] * dFz_dz]
        )

        return torch.cat([out_x, out_y, out_z], dim=-1)


def gamma(v):
    return (sqrt(1 - (v**2 / c**2))) ** -1


class IdealMHD(nn.Module):

    def __init__(self, step_size, edge_order):
        super().__init__()
        self.step_size = step_size
        self.edge_order = edge_order

    def GaussLawMagnetismConstraint(self, B: torch.Tensor, r: torch.Tensor):
        """
        B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
        r: position of the spacecraft | torch.tensor | (batch_size, sequence_length, 3)
        """
        return calc.div(B, r, self.edge_order).mean()

    def GaussLawElectrostaticConstraint(
        self, E: torch.Tensor, sigma: float, r: torch.Tensor
    ):
        """
        E: electric field | torch.tensor | (batch_size, sequence_length, 3)
        sigma: charge density | torch.tensor | (batch_size, sequence_length)
        r: position of the spacecraft | torch.tensor | (batch_size, sequence_length, 3)
        """
        return (calc.div(E, r, self.edge_order) - (sigma / epsilon_0)).mean()

    def DriftVelocity(self, B: torch.Tensor, E: torch.Tensor, v_drift: torch.Tensor):
        """
        B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
        E: electric field | torch.tensor | (batch_size, sequence_length, 3)
        v_drift: drift velocity | torch.tensor | (batch_size, sequence_length, 3)
        """
        return (
            (torch.cross(E, B, dim=-1) / torch.sum(B, dim=-1, keepdim=True)) - v_drift
        ).mean()

    def ContinuityConstraint(self, rho: torch.Tensor, v: torch.Tensor):
        """
        rho: mass density | torch.tensor | (batch_size, sequence_length)
        v: mass velocity field | torch.tensor | (batch_size, sequence_length, 3)
        """
        return ((calc.dF_dt(rho) + calc.div(rho * v)) ** 2).mean()

    def StateConstraint(self, p: torch.Tensor, rho: torch.Tensor, N=3):
        """
        rho: mass density | torch.tensor | (batch_size, sequence_length)
        p: pressure | torch.tensor | (batch_size, sequence_length)
        """
        rho = rho.unsqueeze(-1)
        p = p.unsqueeze(-1)
        gamma = (N + 2) / 2
        return ((calc.dF_dt(p / (rho) ** gamma, self.step_size)) ** 2).mean()

    def Ohm(self, E: torch.Tensor, v: torch.Tensor, B: torch.Tensor):
        """
        B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
        E: electric field | torch.tensor | (batch_size, sequence_length, 3)
        v: mass velocity field | torch.tensor | (batch_size, sequence_length, 3)
        """
        return ((E + torch.cross(v, B, dim=-1)) ** 2).mean()

    def Induction(
        self, B: torch.Tensor, v: torch.Tensor, r: torch.Tensor, edge_order=1
    ):
        """
        B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
        v: mass velocity field | torch.tensor | (batch_size, sequence_length, 3)
        """
        return (
            (
                calc.dF_dt(B, self.step_size, edge_order)
                - calc.rot(torch.cross(v, B, dim=-1), r, edge_order)
            )
            ** 2
        ).mean()

    def MotionConstraint(self, B, J, r, edge_order=1):
        """
        J: current density | torch.tensor | (batch_size, sequence_length)
        B: magnetic field | torch.tensor | (batch_size, sequence_length, 3)
        """
        return (
            (
                torch.cross(J, B, dim=-1)
                - calc.conv_oper(B, B, r, edge_order) / mu_0
                + calc.grad((torch.norm(B, 2, -1, True) ** 2) / (2 * mu_0))
            )
            ** 2
        ).mean()

    def forward(
        self,
        r: torch.Tensor,
        B: torch.Tensor,
        E: torch.Tensor,
        J: torch.Tensor,
        p: torch.Tensor,
        v_drift: torch.Tensor,
        v: torch.Tensor,
        sigma: torch.Tensor,
        rho: torch.Tensor,
        lambdas: list = [0.1 for _ in range(8)],
    ) -> torch.Tensor:
        return (
            torch.tensor(lambdas)
            * torch.tensor(
                [
                    self.GaussLawMagnetismConstraint(B, r),
                    self.GaussLawElectrostaticConstraint(E, sigma, r),
                    self.DriftVelocity(B, E, v_drift),
                    self.ContinuityConstraint(rho, v),
                    self.StateConstraint(p, rho),
                    self.LorentzConstraint(E, v, B),
                    self.AmpereFaradayConstraint(B, v, r),
                    self.MotionConstraint(B, J, r),
                ]
            )
            ** 2
        ).sum()


"""
LOSS FORWARD FUNCTIONS
"""

"""
Priority Criterion

"""


def compute_weights(dataloader):
    bounded = torch.bincount(
        torch.cat([batch["dst"].view(-1) for batch in dataloader], dim=-1)
    )
    return torch.softmax(1 / bounded)


class FineTuningPriorityCriterion(nn.Module):
    def __init__(
        self,
        dataloader,
    ):
        super().__init__()
        self.base_criterion = F.cross_entropy
        # Weight calc
        self.weights = compute_weights(dataloader)

    def forward(self, pred, target) -> torch.Tensor:
        losses = self.base_criterion(pred, target, reduction="none").mean(dim=(1, 2))
        seq_dst, _ = torch.max(target, dim=1)
        return torch.mean(losses * self.weights[seq_dst])


"""
Forcing Criterion
Based on datasets diff
"""


def weight_scaler(dataloader):
    ## Computing weights
    batchwise_loss = []
    for batch in dataloader:
        # output -> (B)
        batchwise_loss.append(
            F.mse_loss(batch["l1"], batch["l2"], reduction="none").mean(dim=(1, 2))
        )
    loss = torch.cat(batchwise_loss, dim=-1)
    loss_max = loss.max()
    loss_min = loss.min()
    return lambda J: (J - loss_max) / (loss_max - loss_min)


class ForcingCriterion(nn.Module):
    def __init__(
        self,
        dataloader,
    ):
        super().__init__()
        self.base_criterion = F.mse_loss
        self.weight_scaler = weight_scaler(dataloader)

    def forward(self, l1_out, l2_out, noise_loss: torch.Tensor) -> torch.Tensor:
        # B,1
        out = self.base_criterion(l1_out, l2_out, reduction="none").mean(dim=(1, 2))
        # (B,1 * B,1).mean()
        return torch.mean(out * noise_loss.apply_(self.weight_scaler))


class NoiseLoss(nn.Module):
    @staticmethod
    def compute(l1, l2):
        # (B,S,I) x (B,S,I) -> (B,1)
        return F.mse_loss(l1, l2, reduction="none").mean(dim=(1, 2))
