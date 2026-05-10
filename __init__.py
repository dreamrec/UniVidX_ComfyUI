# __init__.py
"""
ComfyUI-UniVidX custom node pack.
Strategy A: opaque pipeline wrapper.
"""
try:
    from .nodes.loader import UniVidXLoader
    from .nodes.task import UniVidXTaskMode
    from .nodes.sampler import UniVidXSampler
    from .nodes.decoder import UniVidXDecodeIntrinsic, UniVidXDecodeAlpha
except ImportError:
    # When pytest collects this file because the test dir is a sibling package
    # under a non-Python-named directory, relative imports fail. Fall back to
    # flat imports — `conftest.py` adds the project root to sys.path.
    from nodes.loader import UniVidXLoader  # type: ignore
    from nodes.task import UniVidXTaskMode  # type: ignore
    from nodes.sampler import UniVidXSampler  # type: ignore
    from nodes.decoder import UniVidXDecodeIntrinsic, UniVidXDecodeAlpha  # type: ignore


NODE_CLASS_MAPPINGS = {
    "UniVidXLoader": UniVidXLoader,
    "UniVidXTaskMode": UniVidXTaskMode,
    "UniVidXSampler": UniVidXSampler,
    "UniVidXDecodeIntrinsic": UniVidXDecodeIntrinsic,
    "UniVidXDecodeAlpha": UniVidXDecodeAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniVidXLoader": "UniVidX • Load Model",
    "UniVidXTaskMode": "UniVidX • Task Mode",
    "UniVidXSampler": "UniVidX • Sample",
    "UniVidXDecodeIntrinsic": "UniVidX • Decode (Intrinsic)",
    "UniVidXDecodeAlpha": "UniVidX • Decode (Alpha)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
