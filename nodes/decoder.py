# nodes/decoder.py
"""
UniVidXDecodeIntrinsic, UniVidXDecodeAlpha: splay UNIVIDX_RESULT into IMAGE outputs.

For modes where a particular modality is a CONDITION (not a target), the result
dict won't contain that key. The decoder fills the missing slot with a black
placeholder IMAGE matching the input shape so downstream nodes don't break.
"""
import torch

try:
    from ..src.modes import output_keys, family_of
    from ..src.tensor_io import video_tensor_to_image_batch
except ImportError:
    from src.modes import output_keys, family_of
    from src.tensor_io import video_tensor_to_image_batch


def _placeholder(num_frames: int, height: int, width: int) -> torch.Tensor:
    return torch.zeros(num_frames, height, width, 3)


def _decode(result_dict, mode: str, expected_keys):
    """
    Return a dict of IMAGE batches keyed by expected_keys, using black
    placeholders for any key that's a condition (not in result_dict).
    """
    actual_keys = output_keys(mode)
    images = {}

    # Determine reference shape from any present key
    ref_T = ref_H = ref_W = None
    for k in actual_keys:
        if k in result_dict and result_dict[k] is not None:
            t = result_dict[k]
            if t.ndim == 5:
                t = t.squeeze(0)
            ref_T, ref_H, ref_W = t.shape[1], t.shape[2], t.shape[3]
            break

    for k in expected_keys:
        if k in result_dict and result_dict[k] is not None:
            images[k] = video_tensor_to_image_batch(result_dict[k])
        else:
            if ref_T is None:
                # No reference shape yet (impossible if mode is valid); use 21/480/640
                ref_T, ref_H, ref_W = 21, 480, 640
            images[k] = _placeholder(ref_T, ref_H, ref_W)
    return images


class UniVidXDecodeIntrinsic:
    """Splay an intrinsic-family ``UNIVIDX_RESULT`` into 4 IMAGE batches.

    Outputs ``rgb / albedo / irradiance / normal``. Modalities that were a
    *condition* for the active mode (rather than a target) come back as a black
    placeholder ``IMAGE`` of the right shape so downstream nodes don't break.
    Raises ``ValueError`` if the mode is alpha-family — use
    ``UniVidXDecodeAlpha`` for those.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": ("UNIVIDX_RESULT",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("rgb", "albedo", "irradiance", "normal")
    FUNCTION = "decode"
    CATEGORY = "UniVidX"

    def decode(self, result):
        result_dict, mode = result
        if family_of(mode) != "intrinsic":
            raise ValueError(
                f"Mode {mode} is not intrinsic. Use UniVidXDecodeAlpha instead."
            )
        imgs = _decode(result_dict, mode,
                       expected_keys=["rgb", "albedo", "irradiance", "normal_unit"])
        return (imgs["rgb"], imgs["albedo"], imgs["irradiance"], imgs["normal_unit"])


class UniVidXDecodeAlpha:
    """Splay an alpha-family ``UNIVIDX_RESULT`` into 4 IMAGE batches.

    Outputs ``composite_rgb / alpha / foreground / background``. Modalities that
    were a *condition* for the active mode (rather than a target) come back as
    a black placeholder ``IMAGE`` of the right shape. Raises ``ValueError`` if
    the mode is intrinsic-family — use ``UniVidXDecodeIntrinsic`` for those.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result": ("UNIVIDX_RESULT",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_rgb", "alpha", "foreground", "background")
    FUNCTION = "decode"
    CATEGORY = "UniVidX"

    def decode(self, result):
        result_dict, mode = result
        if family_of(mode) != "alpha":
            raise ValueError(
                f"Mode {mode} is not alpha. Use UniVidXDecodeIntrinsic instead."
            )
        imgs = _decode(result_dict, mode,
                       expected_keys=["rgb", "pha", "fgr", "bgr"])
        return (imgs["rgb"], imgs["pha"], imgs["fgr"], imgs["bgr"])
