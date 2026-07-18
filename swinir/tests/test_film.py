import torch

from jetsr_swin.models.layers.film import FiLMLayer, MetadataEncoder


def test_film_identity_at_init():
    """FiLM is zero-initialized so it should act as identity."""
    film = FiLMLayer(cond_dim=64, feature_dim=32)
    x = torch.randn(4, 8, 8, 32)
    cond = torch.randn(4, 64)
    y = film(x, cond)
    assert torch.allclose(x, y, atol=1e-6)


def test_film_shapes():
    enc = MetadataEncoder(num_classes=2, out_dim=128)
    pt = torch.randn(3)
    m0 = torch.randn(3)
    y = torch.tensor([0, 1, 0])
    z = enc(pt, m0, y)
    assert z.shape == (3, 128)

    film = FiLMLayer(cond_dim=128, feature_dim=96)
    x = torch.randn(3, 10, 10, 96)
    out = film(x, z)
    assert out.shape == x.shape
