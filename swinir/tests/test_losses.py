import torch

from jetsr_swin.data.normalization import ChannelStats
from jetsr_swin.losses import CombinedLoss


def test_combined_loss_runs():
    stats = ChannelStats(mean=torch.zeros(3), std=torch.ones(3))
    loss_fn = CombinedLoss(lambda_l1=50.0, lambda_phys=12.0)
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    loss, components = loss_fn(pred, target, stats)
    loss.backward()
    assert pred.grad is not None
    assert "l1" in components and "phys" in components and "total" in components
