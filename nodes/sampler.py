# nodes/sampler.py
"""
UniVidXSampler: run UniVidX's pipe() inside the chdir context.
"""
import torch

try:
    from ..src.modes import required_inputs, validate_mode
    from ..src.runtime import unividx_cwd, _restore_native_sdpa_if_polluted
    from ..src.tensor_io import image_batch_to_video_tensor
except ImportError:
    from src.modes import required_inputs, validate_mode
    from src.runtime import unividx_cwd, _restore_native_sdpa_if_polluted
    from src.tensor_io import image_batch_to_video_tensor

# UniVidX's standard Chinese negative prompt — same as the inference scripts'.
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


class UniVidXSampler:
    """Run UniVidX's ``pipe()`` end-to-end.

    Accepts up to 7 optional ``IMAGE`` inputs (one per modality across both
    families); inputs not required by the active mode are silently ignored.
    Validates that the loaded model variant matches the task mode's family
    before any sampling work, and that all of the mode's required inputs are
    wired. Returns an opaque ``UNIVIDX_RESULT`` (the dict UniVidX's
    ``pipe()`` returned, plus the mode string) for the decoder to splay.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("UNIVIDX_MODEL",),
                "task": ("UNIVIDX_TASK",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True,
                                                "default": DEFAULT_NEGATIVE_PROMPT}),
                "num_inference_steps": ("INT", {
                    "default": 20, "min": 1, "max": 200,
                    "tooltip": (
                        "Number of denoising steps. UniVidX is trained "
                        "at 20 (production preset). With "
                        "step_distill_lora=lightx2v on the loader + "
                        "cfg_scale=1.0, use 4 (the fast-preview preset, "
                        "~5x faster wall, ~22-26 dB PSNR vs production "
                        "reference). 50+ rarely helps and usually hurts "
                        "on this model."
                    ),
                }),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0,
                                         "step": 0.1}),
                "denoising_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                                  "step": 0.01}),
                "num_frames": ("INT", {"default": 21, "min": 5, "max": 81, "step": 4}),
                "height": ("INT", {"default": 480, "min": 256, "max": 1024, "step": 16}),
                "width": ("INT", {"default": 640, "min": 256, "max": 1280, "step": 16}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF}),
                "tiled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Each optional IMAGE input maps to one conditioning modality.
                # The sampler ignores any input not required by the current mode.
                "rgb": ("IMAGE",),
                "albedo": ("IMAGE",),
                "irradiance": ("IMAGE",),
                "normal": ("IMAGE",),
                "pha": ("IMAGE",),
                "fgr": ("IMAGE",),
                "bgr": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("UNIVIDX_RESULT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "sample"
    CATEGORY = "UniVidX"

    def sample(self, model, task, prompt, negative_prompt, num_inference_steps,
               cfg_scale, denoising_strength, num_frames, height, width, seed, tiled,
               rgb=None, albedo=None, irradiance=None, normal=None,
               pha=None, fgr=None, bgr=None):

        model_instance, variant = model
        mode, family = task

        # Family / variant compatibility
        if family != variant:
            raise ValueError(
                f"Task mode {mode} is family={family}, but loaded model is {variant}. "
                f"Pick a matching loader (intrinsic vs alpha)."
            )

        # Validate inputs against mode requirements
        supplied_imgs = {
            "rgb": rgb, "albedo": albedo, "irradiance": irradiance, "normal": normal,
            "pha": pha, "fgr": fgr, "bgr": bgr,
        }
        supplied = {k for k, v in supplied_imgs.items() if v is not None}
        validate_mode(mode, supplied_inputs=supplied)

        # Convert each supplied IMAGE to UniVidX's expected video tensor
        device = "cuda"
        dtype = next(model_instance.parameters()).dtype if hasattr(model_instance, "parameters") \
                else torch.bfloat16
        video_tensors = {}
        for key in required_inputs(mode):
            img = supplied_imgs[key]
            video_tensors[f"inference_{key}"] = image_batch_to_video_tensor(
                img,
                target_height=height,
                target_width=width,
                target_frames=num_frames,
                device=device,
                dtype=dtype,
            )

        # Fill in None for non-required modality kwargs (the pipe expects them)
        all_input_keys = {
            "intrinsic": ["rgb", "albedo", "irradiance", "normal"],
            "alpha": ["rgb", "pha", "fgr", "bgr"],
        }[family]
        for k in all_input_keys:
            video_tensors.setdefault(f"inference_{k}", None)

        # Tile sizes — defaults from the upstream inference scripts
        tile_size = [30, 52]
        tile_stride = [15, 26]

        # Build the kwargs dict exactly as the upstream scripts do
        inference_params = {
            "prompt": [prompt, prompt, prompt, prompt],
            "negative_prompt": negative_prompt,
            "seed": int(seed),
            "num_inference_steps": int(num_inference_steps),
            "cfg_scale": float(cfg_scale),
            "cfg_merge": False,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "denoising_strength": float(denoising_strength),
            "tiled": bool(tiled),
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "is_inference": True,
            "training_mode": mode,
            **video_tensors,
        }

        # Re-run the SDPA un-pollute defensively. ComfyUI may have queued
        # other nodes between load_model() and this sample() call — if any
        # of them re-pollutes torch.nn.functional.scaled_dot_product_attention
        # (e.g. ComfyUI-3D-Pack's Stable3DGen does this when sageattention
        # is installed), UniVidX's VAE call would route to sage and crash
        # on its 1-head head_dim=channel_count attention.
        _restore_native_sdpa_if_polluted()

        with torch.no_grad(), unividx_cwd():
            video_dict = model_instance.pipe(**inference_params)

        # The result dict's keys depend on the mode; we pass it through opaque to the
        # decoder node which uses output_keys(mode) to splay it out.
        return ((video_dict, mode),)
