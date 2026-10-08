"""Loss tests (need torch; skipped otherwise)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_igbt.losses import MSEObjective, PINNLoss, boundary_penalty, monotonic_penalty  # noqa: E402
from pinn_igbt.metrics import boundary_metrics, monotonicity_metrics  # noqa: E402
from pinn_igbt.models import RNNRegressor  # noqa: E402


def test_monotone_decreasing_has_zero_mdc():
    y_hat = torch.linspace(1, 0, 20)
    assert monotonic_penalty(y_hat).item() == 0.0


def test_penalties_match_numpy_metrics():
    y_hat = torch.tensor([0.9, 0.95, 0.5, 0.6, -0.1, 1.2])
    assert monotonic_penalty(y_hat).item() == pytest.approx(monotonicity_metrics(y_hat.numpy())["mdc"], rel=1e-6)
    assert boundary_penalty(y_hat).item() == pytest.approx(boundary_metrics(y_hat.numpy())["bcc"], rel=1e-6)


def test_pinn_total_matches_formula():
    y = torch.linspace(1, 0, 10)
    y_hat = y + torch.tensor([0.0, 0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.0, -0.3, 0.1])
    a, b, g = 0.1, 1.0, 0.1
    loss, parts = PINNLoss(a, b, g)(y_hat, y)
    expected = (1 - a) * parts["mse"] + a * g * parts["mdc"] + b * parts["bcc"]
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_constraint_gradient_only_when_violated():
    ok = torch.linspace(0.9, 0.1, 10, requires_grad=True)
    monotonic_penalty(ok).backward()
    assert torch.all(ok.grad == 0)
    bad = torch.tensor([0.5, 0.7, 0.3], requires_grad=True)
    monotonic_penalty(bad).backward()
    assert bad.grad.abs().sum() > 0


def test_mse_objective_and_model_shapes():
    y, y_hat = torch.rand(8), torch.rand(8)
    assert MSEObjective()(y_hat, y)[0].item() == pytest.approx(torch.nn.functional.mse_loss(y_hat, y).item())
    assert RNNRegressor()(torch.rand(4, 10, 1)).shape == (4, 1)


def test_model_matches_paper_architecture():
    model = RNNRegressor()
    assert isinstance(model.act, torch.nn.Tanh)                       # paper Eq. 3
    assert isinstance(RNNRegressor(activation="relu").act, torch.nn.ReLU)
    # Paper reports 7,381 parameters (single-bias recurrent layer); PyTorch's nn.RNN has a second
    # bias vector of 80 entries, so 7,381 + 80 = 7,461 here. The function class is identical.
    assert sum(p.numel() for p in model.parameters()) == 7461
