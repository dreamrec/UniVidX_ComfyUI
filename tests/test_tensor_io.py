# tests/test_tensor_io.py
import torch
import pytest

from src.tensor_io import (
    image_batch_to_video_tensor,
    video_tensor_to_image_batch,
)


def test_image_batch_to_video_tensor_shape_and_range():
    # 4 frames, 64x64, RGB, all 1.0 (white)
    img = torch.ones(4, 64, 64, 3)
    out = image_batch_to_video_tensor(img, target_height=64, target_width=64,
                                      target_frames=4, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 3, 4, 64, 64)
    # All-1 input maps to all-1 output (since 1 * 2 - 1 = 1)
    assert torch.allclose(out, torch.ones_like(out))


def test_image_batch_to_video_tensor_pads_short_input():
    # 2 frames, target 5
    img = torch.zeros(2, 32, 32, 3)
    out = image_batch_to_video_tensor(img, target_height=32, target_width=32,
                                      target_frames=5, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 3, 5, 32, 32)


def test_image_batch_to_video_tensor_truncates_long_input():
    img = torch.zeros(10, 32, 32, 3)
    out = image_batch_to_video_tensor(img, target_height=32, target_width=32,
                                      target_frames=4, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 3, 4, 32, 32)


def test_image_batch_to_video_tensor_resizes():
    img = torch.zeros(3, 100, 200, 3)
    out = image_batch_to_video_tensor(img, target_height=64, target_width=128,
                                      target_frames=3, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 3, 3, 64, 128)


def test_image_batch_to_video_tensor_normalizes_to_minus_one_one():
    img = torch.zeros(2, 16, 16, 3)  # all 0.0
    out = image_batch_to_video_tensor(img, target_height=16, target_width=16,
                                      target_frames=2, device="cpu", dtype=torch.float32)
    # 0 * 2 - 1 = -1
    assert torch.allclose(out, -torch.ones_like(out))


def test_video_tensor_to_image_batch_inverse():
    # Source tensor in UniVidX output shape [C, T, H, W] in [-1, 1].
    # Our function applies *0.5 + 0.5 internally then permutes to [T, H, W, C].
    src = torch.zeros(3, 4, 32, 32)  # all 0 -> 0.5 in [0,1]
    out = video_tensor_to_image_batch(src)
    assert out.shape == (4, 32, 32, 3)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_video_tensor_to_image_batch_clamps_out_of_range():
    src = torch.full((3, 2, 8, 8), 5.0)  # way > 1 even after *0.5+0.5
    out = video_tensor_to_image_batch(src)
    assert out.max().item() == 1.0
    assert out.min().item() <= 1.0
