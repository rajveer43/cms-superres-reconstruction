import pytest
import torch

from jetsr_swin.models.swinir import SwinIRConfig, SwinIRGenerator


@pytest.mark.parametrize("B", [1, 2, 4])
def test_swinir_output_shape(B):
    cfg = SwinIRConfig(num_blocks=2)  # small for speed
    model = SwinIRGenerator(cfg)
    lr = torch.randn(B, 3, 64, 64)
    out = model(lr)
    assert out.shape == (B, 3, 125, 125), f"got {out.shape}"


def test_swinir_backward_runs():
    cfg = SwinIRConfig(num_blocks=2)
    model = SwinIRGenerator(cfg)
    lr = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 125, 125)
    out = model(lr)
    loss = (out - target).abs().mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "no gradients flowed into model"


def test_swinir_film_runs():
    cfg = SwinIRConfig(num_blocks=2, use_film=True)
    model = SwinIRGenerator(cfg)
    lr = torch.randn(2, 3, 64, 64)
    meta = {
        "pt": torch.tensor([0.5, -0.2]),
        "m0": torch.tensor([0.1, 0.3]),
        "y": torch.tensor([0, 1]),
    }
    out = model(lr, meta=meta)
    assert out.shape == (2, 3, 125, 125)
