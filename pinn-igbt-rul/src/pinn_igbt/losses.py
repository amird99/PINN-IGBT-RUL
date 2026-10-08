"""Objectives: plain MSE and the physics-informed loss.

Physics prior used by the PINN loss (soft constraints, no autograd derivatives):
  * MDC - monotonic degradation: normalised RUL never increases from cycle to cycle.
  * BCC - boundary condition:    normalised RUL stays inside [0, 1].
"""
from __future__ import annotations

import torch
from torch import nn

from .config import LossConfig

LossOutput = tuple[torch.Tensor, dict[str, torch.Tensor]]


def monotonic_penalty(y_hat: torch.Tensor) -> torch.Tensor:
    """mean(relu(y_hat[i] - y_hat[i-1])^2) over consecutive, time-ordered predictions."""
    if y_hat.shape[0] < 2:
        return y_hat.new_zeros(())
    return torch.relu(y_hat[1:] - y_hat[:-1]).pow(2).mean()


def boundary_penalty(y_hat: torch.Tensor) -> torch.Tensor:
    """mean(relu(-y_hat)^2) + mean(relu(y_hat - 1)^2)."""
    return torch.relu(-y_hat).pow(2).mean() + torch.relu(y_hat - 1.0).pow(2).mean()


class MSEObjective(nn.Module):
    """Mean squared error with the same ``(loss, parts)`` interface as :class:`PINNLoss`."""

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> LossOutput:
        loss = torch.mean((y - y_hat) ** 2)
        return loss, {"mse": loss.detach()}


class PINNLoss(nn.Module):
    """(1 - alpha) * MSE + alpha * gamma * MDC + beta * BCC.

    ``y_hat`` and ``y`` are 1-D and MUST be ordered by time (consecutive cycles of ONE
    device); shuffled mini-batches would make the MDC term meaningless.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 1.0, gamma: float = 0.1) -> None:
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> LossOutput:
        mse = torch.mean((y - y_hat) ** 2)
        mdc = monotonic_penalty(y_hat)
        bcc = boundary_penalty(y_hat)
        loss = (1 - self.alpha) * mse + self.alpha * self.gamma * mdc + self.beta * bcc
        parts = {"mse": mse.detach(), "mdc": mdc.detach(), "bcc": bcc.detach()}
        return loss, parts


def build_loss(cfg: LossConfig) -> nn.Module:
    if cfg.name == "mse":
        return MSEObjective()
    return PINNLoss(cfg.alpha, cfg.beta, cfg.gamma)
