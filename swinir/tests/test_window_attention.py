import torch

from jetsr_swin.models.layers.window_attention import window_partition, window_reverse


def test_partition_reverse_roundtrip():
    B, H, W, C = 2, 16, 16, 8
    ws = 4
    x = torch.randn(B, H, W, C)
    windows = window_partition(x, ws)
    assert windows.shape == (B * (H // ws) * (W // ws), ws, ws, C)
    x_back = window_reverse(windows, ws, H, W)
    assert torch.allclose(x, x_back)


def test_partition_shapes_for_swin_grid():
    # 64x64 grid with window 4 -> 256 windows per batch
    x = torch.randn(1, 64, 64, 96)
    w = window_partition(x, 4)
    assert w.shape == (256, 4, 4, 96)
