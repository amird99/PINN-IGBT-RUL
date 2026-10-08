"""Vanilla-RNN regressor (identical architecture for baseline and PINN; paper Eq. 2-4)."""
from __future__ import annotations

import torch
from torch import nn


class RNNRegressor(nn.Module):
    """RNN(tanh) -> last time step -> Linear -> activation -> Linear (linear output).

    Layer creation order is unchanged from the original notebook, so the same seed gives
    the same initial weights. Input: (batch, window, input_size). Output: (batch, 1), unbounded.
    PyTorch's ``nn.RNN`` has two bias vectors, so it has 7,461 parameters versus the
    7,381 reported in the paper (a single-bias recurrent layer); the model is equivalent.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 80, head_size: int = 10,
                 activation: str = "tanh") -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, head_size)
        # [FIX C5] paper Eq. 3 uses tanh here; the notebook used ReLU (activation="relu").
        self.act = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.fc2 = nn.Linear(head_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc2(self.act(self.fc1(out[:, -1, :])))
