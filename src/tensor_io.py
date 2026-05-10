# src/tensor_io.py
"""
Conversion between ComfyUI IMAGE batches and UniVidX video tensors.

ComfyUI IMAGE: torch.Tensor of shape [B, H, W, C] in [0, 1] (typically float32).
UniVidX input: torch.Tensor of shape [1, 3, T, H, W] in [-1, 1] (bfloat16 on CUDA).
UniVidX output: torch.Tensor of shape [3, T, H, W] in [-1, 1] (post-pipe, post-VAE-decode).
"""
import torch
import torch.nn.functional as F


def image_batch_to_video_tensor(
    image_batch: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    target_frames: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Convert a ComfyUI IMAGE [B,H,W,C in 0..1] to a UniVidX [1,3,T,H,W in -1..1].

    Mirrors UniVidX's `_load_mp4_as_video_tensor` exactly: bilinear resize,
    pad-by-repeat-last-frame to target frame count, *2-1 normalization.
    """
    if image_batch.ndim != 4 or image_batch.shape[-1] != 3:
        raise ValueError(
            f"Expected IMAGE [B,H,W,3], got shape {tuple(image_batch.shape)}"
        )

    # [B,H,W,C] -> [B,C,H,W] for F.interpolate
    video = image_batch.to(torch.float32).permute(0, 3, 1, 2)

    # Resize if needed
    if video.shape[-2] != target_height or video.shape[-1] != target_width:
        video = F.interpolate(
            video, size=(target_height, target_width),
            mode="bilinear", align_corners=False,
        )

    # Frame count adjustment
    T = video.shape[0]
    if T >= target_frames:
        video = video[:target_frames]
    else:
        pad = video[-1:].repeat(target_frames - T, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    # [T,C,H,W] -> [C,T,H,W] -> [1,C,T,H,W]
    video = video.permute(1, 0, 2, 3).contiguous()
    video = video * 2.0 - 1.0
    video = video.unsqueeze(0).to(device=device, dtype=dtype)
    return video


def video_tensor_to_image_batch(video: torch.Tensor) -> torch.Tensor:
    """
    Convert a UniVidX output tensor [C,T,H,W] in [-1,1] (or [1,C,T,H,W]) to a
    ComfyUI IMAGE [T,H,W,C] in [0,1].

    Reproduces the post-processing in UniVidX's _tensor2video plus the *0.5+0.5
    rescaling that the inference scripts apply.
    """
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch=1, got shape {tuple(video.shape)}")
        video = video.squeeze(0)
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(
            f"Expected video tensor [3,T,H,W], got shape {tuple(video.shape)}"
        )

    # [-1,1] -> [0,1]
    video = video.detach().cpu().to(torch.float32) * 0.5 + 0.5
    video = video.clamp(0.0, 1.0)
    # [C,T,H,W] -> [T,H,W,C]
    return video.permute(1, 2, 3, 0).contiguous()
