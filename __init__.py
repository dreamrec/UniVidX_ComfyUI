# __init__.py
"""
ComfyUI-UniVidX custom node pack.
Strategy A: opaque pipeline wrapper.
"""
# Two layers of defense:
# 1. Try the proper relative imports first (the path ComfyUI uses).
# 2. Fall back to flat imports for pytest collection when the package
#    name is hyphenated (relative imports fail in that case).
# 3. If torch (or any other runtime dep) is missing — common in CI/lint
#    environments — degrade to no node mappings rather than crashing
#    the import. ComfyUI hosts always have torch, so this only matters
#    in non-runtime contexts.
try:
    from .nodes.loader import UniVidXLoader
    from .nodes.task import UniVidXTaskMode
    from .nodes.sampler import UniVidXSampler
    from .nodes.decoder import UniVidXDecodeIntrinsic, UniVidXDecodeAlpha
except ImportError:
    try:
        from nodes.loader import UniVidXLoader  # type: ignore
        from nodes.task import UniVidXTaskMode  # type: ignore
        from nodes.sampler import UniVidXSampler  # type: ignore
        from nodes.decoder import UniVidXDecodeIntrinsic, UniVidXDecodeAlpha  # type: ignore
    except ImportError:
        UniVidXLoader = None              # type: ignore[assignment]
        UniVidXTaskMode = None            # type: ignore[assignment]
        UniVidXSampler = None             # type: ignore[assignment]
        UniVidXDecodeIntrinsic = None     # type: ignore[assignment]
        UniVidXDecodeAlpha = None         # type: ignore[assignment]


NODE_CLASS_MAPPINGS = {
    name: cls for name, cls in {
        "UniVidXLoader": UniVidXLoader,
        "UniVidXTaskMode": UniVidXTaskMode,
        "UniVidXSampler": UniVidXSampler,
        "UniVidXDecodeIntrinsic": UniVidXDecodeIntrinsic,
        "UniVidXDecodeAlpha": UniVidXDecodeAlpha,
    }.items() if cls is not None
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniVidXLoader": "UniVidX • Load Model",
    "UniVidXTaskMode": "UniVidX • Task Mode",
    "UniVidXSampler": "UniVidX • Sample",
    "UniVidXDecodeIntrinsic": "UniVidX • Decode (Intrinsic)",
    "UniVidXDecodeAlpha": "UniVidX • Decode (Alpha)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
