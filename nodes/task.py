# nodes/task.py
"""
UniVidXTaskMode: pick a mode.

Outputs UNIVIDX_TASK = (mode_string, family).
"""
try:
    from ..src.modes import INTRINSIC_MODES, ALPHA_MODES, family_of
except ImportError:
    from src.modes import INTRINSIC_MODES, ALPHA_MODES, family_of


class UniVidXTaskMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (INTRINSIC_MODES + ALPHA_MODES, {"default": "t2RAIN"}),
            }
        }

    RETURN_TYPES = ("UNIVIDX_TASK",)
    RETURN_NAMES = ("task",)
    FUNCTION = "select"
    CATEGORY = "UniVidX"

    def select(self, mode: str):
        return ((mode, family_of(mode)),)
