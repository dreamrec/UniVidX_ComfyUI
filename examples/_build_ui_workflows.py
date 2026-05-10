"""
Generate ComfyUI UI-format workflow JSONs for the 6 user-facing demos with
generous, validated spacing.

Spacing rules (from project feedback):
- Between nodes in the same group, horizontal: >= 60px
- Between nodes in the same group, vertical:   >= 80px
- Between groups (any edge):                    >= 60px
- Group bounding box padding around its nodes:  >= 30px

Layout strategy (left-to-right dataflow):
    [Setup] -> [Conditioning] -> [Sampling] -> [Decode] -> [Outputs]
Each group is a vertical stack; groups are horizontally arranged; bounding
boxes are *computed* from actual node positions, never hand-picked.

After laying out, an overlap pass checks every pair of node rects AND every
pair of group rects. The script fails loudly if any overlap is found.

Outputs (UI-format JSONs that drag-and-drop into ComfyUI's canvas):
    examples/t2RAIN_basic.json       (text -> RGB+A+I+N, intrinsic)
    examples/R2AIN_basic.json        (RGB -> A+I+N, intrinsic)
    examples/t2RPFB_basic.json       (text -> R+P+F+B, alpha)
    examples/R2PFB_basic.json        (RGB -> P+F+B, alpha)
    examples/I_video_output.json     (t2RAIN -> 4x VHS_VideoCombine MP4)
    examples/J_alpha_compositing.json (R2PFB matte -> ImageCompositeMasked over cyan)

The matching API-format JSONs (suffixed `_api.json`) are auto-derived for
programmatic queueing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Spacing constants — generous defaults that satisfy the feedback rules.
# ---------------------------------------------------------------------------
H_GAP_BETWEEN_GROUPS = 80   # ≥ 60 minimum
V_GAP_WITHIN_GROUP   = 100  # ≥ 80 minimum (between stacked nodes)
GROUP_PAD_LEFT       = 40   # space inside group, left of nodes
GROUP_PAD_RIGHT      = 40
GROUP_PAD_TOP        = 60   # title bar space
GROUP_PAD_BOTTOM     = 40

CANVAS_ORIGIN_X = 40        # left margin of leftmost group on canvas
CANVAS_ORIGIN_Y = 60        # top margin of topmost group on canvas


# Color palette (hex codes ComfyUI's group "color" field accepts).
COLOR_SETUP   = "#3a7c5a"   # green   — preparation
COLOR_INPUT   = "#3a8c7c"   # teal    — conditioning inputs
COLOR_SAMPLE  = "#4a5fc1"   # blue    — work
COLOR_DECODE  = "#c14a8e"   # magenta — splay
COLOR_OUTPUT  = "#c97a3a"   # orange  — sinks
COLOR_COMP    = "#7a4ac1"   # purple  — composite chain


# Node-size estimates (W, H). ComfyUI re-sizes on load; these are reasonable
# defaults that keep the initial layout legible.
SIZES: dict[str, tuple[int, int]] = {
    "UniVidXLoader":          (320, 110),
    "UniVidXTaskMode":        (320,  80),
    "UniVidXSampler":         (420, 720),
    "UniVidXDecodeIntrinsic": (280, 150),
    "UniVidXDecodeAlpha":     (280, 150),
    "LoadImage":              (320, 320),
    "RepeatImageBatch":       (320,  80),
    "SaveImage":              (320, 280),
    "VHS_VideoCombine":       (380, 480),
    "EmptyImage":             (280, 130),
    "ImageToMask":            (280,  80),
    "ImageCompositeMasked":   (320, 200),
}


# Widget value templates — must be ordered the way ComfyUI consumes them
# (positional args matching the node's INPUT_TYPES required dict).
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)
PROMPT = "a small hedgehog wearing a chef hat in a tiny kitchen"


# ---------------------------------------------------------------------------
# Builder primitives.
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: int
    type: str
    pos: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (320, 80)
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    widgets_values: list = field(default_factory=list)
    order: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2)"""
        return (self.pos[0], self.pos[1],
                self.pos[0] + self.size[0], self.pos[1] + self.size[1])


@dataclass
class Group:
    title: str
    color: str
    nodes: list[Node]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Computed from contained-node positions + padding."""
        if not self.nodes:
            raise ValueError(f"Group {self.title!r} has no nodes")
        x1 = min(n.pos[0] for n in self.nodes) - GROUP_PAD_LEFT
        y1 = min(n.pos[1] for n in self.nodes) - GROUP_PAD_TOP
        x2 = max(n.pos[0] + n.size[0] for n in self.nodes) + GROUP_PAD_RIGHT
        y2 = max(n.pos[1] + n.size[1] for n in self.nodes) + GROUP_PAD_BOTTOM
        return (x1, y1, x2, y2)

    @property
    def bounding(self) -> list[int]:
        """ComfyUI's `bounding` field: [x, y, w, h]"""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Strict overlap (touching edges OK)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


# ---------------------------------------------------------------------------
# Layout: lay out groups left-to-right, nodes inside each group top-to-bottom.
# ---------------------------------------------------------------------------

def layout(groups: list["GroupSpec"]) -> tuple[list[Node], list[Group]]:
    """
    Lay out a list of GroupSpecs left-to-right, nodes within each group
    stacked top-to-bottom. Returns flat [Node] and [Group] lists with
    positions assigned and group bboxes computed from those positions.
    """
    placed_nodes: list[Node] = []
    placed_groups: list[Group] = []

    cur_x = CANVAS_ORIGIN_X
    next_node_id = 1
    for gspec in groups:
        # Widest node in this group sets the group's content width
        max_w = max(SIZES[ntype][0] for (ntype, _, _, _) in gspec.spec)
        node_y = CANVAS_ORIGIN_Y + GROUP_PAD_TOP
        node_x = cur_x + GROUP_PAD_LEFT
        nodes_in_group: list[Node] = []
        for (ntype, internal_id, widgets, slot_specs) in gspec.spec:
            w, h = SIZES[ntype]
            # Center each node horizontally within the group's content width
            x = node_x + (max_w - w) // 2
            n = Node(
                id=next_node_id,
                type=ntype,
                pos=(x, node_y),
                size=(w, h),
                inputs=slot_specs.get("inputs", []),
                outputs=slot_specs.get("outputs", []),
                widgets_values=widgets,
                order=next_node_id - 1,
            )
            n._internal = internal_id   # stash for link resolution # type: ignore[attr-defined]
            nodes_in_group.append(n)
            placed_nodes.append(n)
            next_node_id += 1
            node_y += h + V_GAP_WITHIN_GROUP
        placed_groups.append(Group(title=gspec.title, color=gspec.color, nodes=nodes_in_group))
        cur_x += GROUP_PAD_LEFT + max_w + GROUP_PAD_RIGHT + H_GAP_BETWEEN_GROUPS
    return placed_nodes, placed_groups


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------

def validate_no_overlap(nodes: list[Node], groups: list[Group]) -> None:
    # Pairwise node-rect overlap
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if _rects_overlap(a.bbox, b.bbox):
                raise RuntimeError(
                    f"Node {a.id}({a.type}) at {a.pos} and "
                    f"Node {b.id}({b.type}) at {b.pos} overlap."
                )
    # Pairwise group-rect overlap
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            if _rects_overlap(a.bbox, b.bbox):
                raise RuntimeError(
                    f"Group {a.title!r} bbox={a.bbox} and "
                    f"Group {b.title!r} bbox={b.bbox} overlap."
                )


# ---------------------------------------------------------------------------
# Schema for INPUT_TYPES — used to populate node `inputs`/`outputs` arrays.
# Matches the structure ComfyUI's frontend serializes for each node type.
# ---------------------------------------------------------------------------

NODE_SCHEMA: dict[str, dict] = {
    "UniVidXLoader": {
        "inputs": [],
        "outputs": [{"name": "model", "type": "UNIVIDX_MODEL"}],
    },
    "UniVidXTaskMode": {
        "inputs": [],
        "outputs": [{"name": "task", "type": "UNIVIDX_TASK"}],
    },
    "UniVidXSampler": {
        "inputs": [
            {"name": "model", "type": "UNIVIDX_MODEL"},
            {"name": "task",  "type": "UNIVIDX_TASK"},
            {"name": "rgb",         "type": "IMAGE", "shape": 7},
            {"name": "albedo",      "type": "IMAGE", "shape": 7},
            {"name": "irradiance",  "type": "IMAGE", "shape": 7},
            {"name": "normal",      "type": "IMAGE", "shape": 7},
            {"name": "pha",         "type": "IMAGE", "shape": 7},
            {"name": "fgr",         "type": "IMAGE", "shape": 7},
            {"name": "bgr",         "type": "IMAGE", "shape": 7},
        ],
        "outputs": [{"name": "result", "type": "UNIVIDX_RESULT"}],
    },
    "UniVidXDecodeIntrinsic": {
        "inputs": [{"name": "result", "type": "UNIVIDX_RESULT"}],
        "outputs": [
            {"name": "rgb",        "type": "IMAGE"},
            {"name": "albedo",     "type": "IMAGE"},
            {"name": "irradiance", "type": "IMAGE"},
            {"name": "normal",     "type": "IMAGE"},
        ],
    },
    "UniVidXDecodeAlpha": {
        "inputs": [{"name": "result", "type": "UNIVIDX_RESULT"}],
        "outputs": [
            {"name": "composite_rgb", "type": "IMAGE"},
            {"name": "alpha",         "type": "IMAGE"},
            {"name": "foreground",    "type": "IMAGE"},
            {"name": "background",    "type": "IMAGE"},
        ],
    },
    "LoadImage": {
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE"},
            {"name": "MASK",  "type": "MASK"},
        ],
    },
    "RepeatImageBatch": {
        "inputs": [{"name": "image", "type": "IMAGE"}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
    },
    "SaveImage": {
        "inputs": [{"name": "images", "type": "IMAGE"}],
        "outputs": [],
    },
    "VHS_VideoCombine": {
        "inputs": [
            {"name": "images", "type": "IMAGE"},
            {"name": "audio",     "type": "AUDIO",            "shape": 7},
            {"name": "meta_batch","type": "VHS_BatchManager", "shape": 7},
            {"name": "vae",       "type": "VAE",              "shape": 7},
        ],
        "outputs": [{"name": "Filenames", "type": "VHS_FILENAMES"}],
    },
    "EmptyImage": {
        "inputs": [],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
    },
    "ImageToMask": {
        "inputs": [{"name": "image", "type": "IMAGE"}],
        "outputs": [{"name": "MASK", "type": "MASK"}],
    },
    "ImageCompositeMasked": {
        "inputs": [
            {"name": "destination", "type": "IMAGE"},
            {"name": "source",      "type": "IMAGE"},
            {"name": "mask",        "type": "MASK", "shape": 7},
        ],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
    },
}


# ---------------------------------------------------------------------------
# GroupSpec helper — bundles title/color/node-spec for the layout function.
# ---------------------------------------------------------------------------

@dataclass
class GroupSpec:
    title: str
    color: str
    spec: list[tuple[str, str, list, dict]]  # [(node_type, internal_id, widgets, schema_overrides)]


def make_node(ntype: str, iid: str, widgets: list, _schema_overrides: dict | None = None
              ) -> tuple[str, str, list, dict]:
    schema = NODE_SCHEMA[ntype]
    return (ntype, iid, widgets, {
        "inputs": [dict(s) for s in schema["inputs"]],
        "outputs": [dict(s) for s in schema["outputs"]],
    })


# ---------------------------------------------------------------------------
# Link assembly — given a logical (src_iid, src_slot) -> (dst_iid, dst_slot)
# list, resolve to actual node IDs and produce the ComfyUI links array.
# ---------------------------------------------------------------------------

def _find(nodes: list[Node], iid: str) -> Node:
    for n in nodes:
        if getattr(n, "_internal", None) == iid:
            return n
    raise KeyError(f"No node with internal id {iid!r}")


def assemble_links(nodes: list[Node],
                   logical_links: list[tuple[str, int, str, int, str]]
                   ) -> list[list]:
    """
    logical_links: list of (src_iid, src_slot, dst_iid, dst_slot, type_name).

    Mutates nodes' inputs/outputs to record the links and returns ComfyUI's
    [link_id, src_node_id, src_slot, dst_node_id, dst_slot, type] array.
    """
    out: list[list] = []
    for link_id, (src_iid, src_slot, dst_iid, dst_slot, type_name) in enumerate(logical_links, start=1):
        src = _find(nodes, src_iid)
        dst = _find(nodes, dst_iid)
        # Mark dst input
        if dst_slot < len(dst.inputs):
            dst.inputs[dst_slot]["link"] = link_id
        # Mark src output
        if src_slot < len(src.outputs):
            src.outputs[src_slot].setdefault("links", []).append(link_id)
            src.outputs[src_slot]["slot_index"] = src_slot
        out.append([link_id, src.id, src_slot, dst.id, dst_slot, type_name])
    return out


# ---------------------------------------------------------------------------
# Sampler-widget templates.
# ---------------------------------------------------------------------------

def sampler_widgets(prompt: str = PROMPT, neg: str = NEG_PROMPT,
                    *, steps: int = 20, cfg: float = 5.0, denoise: float = 1.0,
                    num_frames: int = 21, height: int = 480, width: int = 640,
                    seed: int = 42, tiled: bool = True) -> list:
    """Match the order of UniVidXSampler's `required` widgets in INPUT_TYPES."""
    return [
        prompt, neg, steps, cfg, denoise, num_frames, height, width,
        seed, "fixed", tiled,
    ]


# ---------------------------------------------------------------------------
# Workflow definitions — one function per demo.
# ---------------------------------------------------------------------------

def workflow_t2RAIN_basic() -> dict:
    setup = GroupSpec("Model Setup", COLOR_SETUP, [
        make_node("UniVidXLoader",   "loader", ["intrinsic", "bfloat16"]),
        make_node("UniVidXTaskMode", "task",   ["t2RAIN"]),
    ])
    sampling = GroupSpec("Sampling", COLOR_SAMPLE, [
        make_node("UniVidXSampler", "sampler", sampler_widgets()),
    ])
    decode = GroupSpec("Decode (Intrinsic)", COLOR_DECODE, [
        make_node("UniVidXDecodeIntrinsic", "decoder", []),
    ])
    outputs = GroupSpec("Outputs (RGB / Albedo / Irradiance / Normal)", COLOR_OUTPUT, [
        make_node("SaveImage", "save_rgb",        ["unividx_t2RAIN_rgb"]),
        make_node("SaveImage", "save_albedo",     ["unividx_t2RAIN_albedo"]),
        make_node("SaveImage", "save_irradiance", ["unividx_t2RAIN_irradiance"]),
        make_node("SaveImage", "save_normal",     ["unividx_t2RAIN_normal"]),
    ])
    groups = [setup, sampling, decode, outputs]
    nodes, placed_groups = layout(groups)
    links = assemble_links(nodes, [
        ("loader",  0, "sampler", 0, "UNIVIDX_MODEL"),
        ("task",    0, "sampler", 1, "UNIVIDX_TASK"),
        ("sampler", 0, "decoder", 0, "UNIVIDX_RESULT"),
        ("decoder", 0, "save_rgb",        0, "IMAGE"),
        ("decoder", 1, "save_albedo",     0, "IMAGE"),
        ("decoder", 2, "save_irradiance", 0, "IMAGE"),
        ("decoder", 3, "save_normal",     0, "IMAGE"),
    ])
    validate_no_overlap(nodes, placed_groups)
    return finalize(nodes, placed_groups, links,
                    name="UniVidX • t2RAIN (text -> RGB/A/I/N)")


def workflow_R2AIN_basic() -> dict:
    setup = GroupSpec("Model Setup", COLOR_SETUP, [
        make_node("UniVidXLoader",   "loader", ["intrinsic", "bfloat16"]),
        make_node("UniVidXTaskMode", "task",   ["R2AIN"]),
    ])
    cond = GroupSpec("RGB Conditioning", COLOR_INPUT, [
        make_node("LoadImage",        "load_rgb",   ["unividx_R2AIN_input.png", "image"]),
        make_node("RepeatImageBatch", "repeat_rgb", [21]),
    ])
    sampling = GroupSpec("Sampling", COLOR_SAMPLE, [
        make_node("UniVidXSampler", "sampler", sampler_widgets()),
    ])
    decode = GroupSpec("Decode (Intrinsic)", COLOR_DECODE, [
        make_node("UniVidXDecodeIntrinsic", "decoder", []),
    ])
    outputs = GroupSpec("Outputs", COLOR_OUTPUT, [
        make_node("SaveImage", "save_rgb",        ["unividx_R2AIN_rgb_placeholder"]),
        make_node("SaveImage", "save_albedo",     ["unividx_R2AIN_albedo"]),
        make_node("SaveImage", "save_irradiance", ["unividx_R2AIN_irradiance"]),
        make_node("SaveImage", "save_normal",     ["unividx_R2AIN_normal"]),
    ])
    groups = [setup, cond, sampling, decode, outputs]
    nodes, placed_groups = layout(groups)
    # Sampler's `rgb` input is at slot 2 (after model[0], task[1])
    links = assemble_links(nodes, [
        ("loader",     0, "sampler",    0, "UNIVIDX_MODEL"),
        ("task",       0, "sampler",    1, "UNIVIDX_TASK"),
        ("load_rgb",   0, "repeat_rgb", 0, "IMAGE"),
        ("repeat_rgb", 0, "sampler",    2, "IMAGE"),
        ("sampler",    0, "decoder",    0, "UNIVIDX_RESULT"),
        ("decoder",    0, "save_rgb",        0, "IMAGE"),
        ("decoder",    1, "save_albedo",     0, "IMAGE"),
        ("decoder",    2, "save_irradiance", 0, "IMAGE"),
        ("decoder",    3, "save_normal",     0, "IMAGE"),
    ])
    validate_no_overlap(nodes, placed_groups)
    return finalize(nodes, placed_groups, links,
                    name="UniVidX • R2AIN (RGB -> A/I/N)")


def workflow_t2RPFB_basic() -> dict:
    setup = GroupSpec("Model Setup", COLOR_SETUP, [
        make_node("UniVidXLoader",   "loader", ["alpha", "bfloat16"]),
        make_node("UniVidXTaskMode", "task",   ["t2RPFB"]),
    ])
    sampling = GroupSpec("Sampling", COLOR_SAMPLE, [
        make_node("UniVidXSampler", "sampler",
                  sampler_widgets(height=432, width=768)),  # alpha default res
    ])
    decode = GroupSpec("Decode (Alpha)", COLOR_DECODE, [
        make_node("UniVidXDecodeAlpha", "decoder", []),
    ])
    outputs = GroupSpec("Outputs (Composite / Pha / Fgr / Bgr)", COLOR_OUTPUT, [
        make_node("SaveImage", "save_composite", ["unividx_t2RPFB_composite"]),
        make_node("SaveImage", "save_alpha",     ["unividx_t2RPFB_alpha"]),
        make_node("SaveImage", "save_fgr",       ["unividx_t2RPFB_foreground"]),
        make_node("SaveImage", "save_bgr",       ["unividx_t2RPFB_background"]),
    ])
    groups = [setup, sampling, decode, outputs]
    nodes, placed_groups = layout(groups)
    links = assemble_links(nodes, [
        ("loader",  0, "sampler", 0, "UNIVIDX_MODEL"),
        ("task",    0, "sampler", 1, "UNIVIDX_TASK"),
        ("sampler", 0, "decoder", 0, "UNIVIDX_RESULT"),
        ("decoder", 0, "save_composite", 0, "IMAGE"),
        ("decoder", 1, "save_alpha",     0, "IMAGE"),
        ("decoder", 2, "save_fgr",       0, "IMAGE"),
        ("decoder", 3, "save_bgr",       0, "IMAGE"),
    ])
    validate_no_overlap(nodes, placed_groups)
    return finalize(nodes, placed_groups, links,
                    name="UniVidX • t2RPFB (text -> R/P/F/B)")


def workflow_R2PFB_basic() -> dict:
    setup = GroupSpec("Model Setup", COLOR_SETUP, [
        make_node("UniVidXLoader",   "loader", ["alpha", "bfloat16"]),
        make_node("UniVidXTaskMode", "task",   ["R2PFB"]),
    ])
    cond = GroupSpec("RGB Conditioning", COLOR_INPUT, [
        make_node("LoadImage",        "load_rgb",   ["unividx_R2AIN_input.png", "image"]),
        make_node("RepeatImageBatch", "repeat_rgb", [21]),
    ])
    sampling = GroupSpec("Sampling", COLOR_SAMPLE, [
        make_node("UniVidXSampler", "sampler",
                  sampler_widgets(height=432, width=768)),
    ])
    decode = GroupSpec("Decode (Alpha)", COLOR_DECODE, [
        make_node("UniVidXDecodeAlpha", "decoder", []),
    ])
    outputs = GroupSpec("Outputs", COLOR_OUTPUT, [
        make_node("SaveImage", "save_composite", ["unividx_R2PFB_composite_placeholder"]),
        make_node("SaveImage", "save_alpha",     ["unividx_R2PFB_alpha_matte"]),
        make_node("SaveImage", "save_fgr",       ["unividx_R2PFB_foreground"]),
        make_node("SaveImage", "save_bgr",       ["unividx_R2PFB_background"]),
    ])
    groups = [setup, cond, sampling, decode, outputs]
    nodes, placed_groups = layout(groups)
    links = assemble_links(nodes, [
        ("loader",     0, "sampler",    0, "UNIVIDX_MODEL"),
        ("task",       0, "sampler",    1, "UNIVIDX_TASK"),
        ("load_rgb",   0, "repeat_rgb", 0, "IMAGE"),
        ("repeat_rgb", 0, "sampler",    2, "IMAGE"),
        ("sampler",    0, "decoder",    0, "UNIVIDX_RESULT"),
        ("decoder",    0, "save_composite", 0, "IMAGE"),
        ("decoder",    1, "save_alpha",     0, "IMAGE"),
        ("decoder",    2, "save_fgr",       0, "IMAGE"),
        ("decoder",    3, "save_bgr",       0, "IMAGE"),
    ])
    validate_no_overlap(nodes, placed_groups)
    return finalize(nodes, placed_groups, links,
                    name="UniVidX • R2PFB (RGB -> Pha/Fgr/Bgr)")


def workflow_I_video_output() -> dict:
    setup = GroupSpec("Model Setup", COLOR_SETUP, [
        make_node("UniVidXLoader",   "loader", ["intrinsic", "bfloat16"]),
        make_node("UniVidXTaskMode", "task",   ["t2RAIN"]),
    ])
    sampling = GroupSpec("Sampling", COLOR_SAMPLE, [
        make_node("UniVidXSampler", "sampler", sampler_widgets()),
    ])
    decode = GroupSpec("Decode (Intrinsic)", COLOR_DECODE, [
        make_node("UniVidXDecodeIntrinsic", "decoder", []),
    ])
    outputs = GroupSpec("Video Outputs (4 MP4s)", COLOR_OUTPUT, [
        # VHS_VideoCombine widgets order: frame_rate, loop_count, filename_prefix, format, pingpong, save_output
        make_node("VHS_VideoCombine", "vhs_rgb",        [16.0, 0, "unividx_t2RAIN_video_rgb",        "video/h264-mp4", False, True]),
        make_node("VHS_VideoCombine", "vhs_albedo",     [16.0, 0, "unividx_t2RAIN_video_albedo",     "video/h264-mp4", False, True]),
        make_node("VHS_VideoCombine", "vhs_irradiance", [16.0, 0, "unividx_t2RAIN_video_irradiance", "video/h264-mp4", False, True]),
        make_node("VHS_VideoCombine", "vhs_normal",     [16.0, 0, "unividx_t2RAIN_video_normal",     "video/h264-mp4", False, True]),
    ])
    groups = [setup, sampling, decode, outputs]
    nodes, placed_groups = layout(groups)
    links = assemble_links(nodes, [
        ("loader",  0, "sampler", 0, "UNIVIDX_MODEL"),
        ("task",    0, "sampler", 1, "UNIVIDX_TASK"),
        ("sampler", 0, "decoder", 0, "UNIVIDX_RESULT"),
        ("decoder", 0, "vhs_rgb",        0, "IMAGE"),
        ("decoder", 1, "vhs_albedo",     0, "IMAGE"),
        ("decoder", 2, "vhs_irradiance", 0, "IMAGE"),
        ("decoder", 3, "vhs_normal",     0, "IMAGE"),
    ])
    validate_no_overlap(nodes, placed_groups)
    return finalize(nodes, placed_groups, links,
                    name="UniVidX • t2RAIN -> 4 MP4 videos")


def workflow_J_alpha_compositing() -> dict:
    setup = GroupSpec("Model Setup", COLOR_SETUP, [
        make_node("UniVidXLoader",   "loader", ["alpha", "bfloat16"]),
        make_node("UniVidXTaskMode", "task",   ["R2PFB"]),
    ])
    cond = GroupSpec("RGB Conditioning", COLOR_INPUT, [
        make_node("LoadImage",        "load_rgb",   ["unividx_R2AIN_input.png", "image"]),
        make_node("RepeatImageBatch", "repeat_rgb", [21]),
    ])
    sampling = GroupSpec("Sampling", COLOR_SAMPLE, [
        make_node("UniVidXSampler", "sampler",
                  sampler_widgets(height=432, width=768)),
    ])
    decode = GroupSpec("Decode (Alpha)", COLOR_DECODE, [
        make_node("UniVidXDecodeAlpha", "decoder", []),
    ])
    comp_setup = GroupSpec("Background", COLOR_COMP, [
        # EmptyImage widgets: width, height, batch_size, color  (cyan = 65535 = 0x00FFFF)
        make_node("EmptyImage", "empty_bg", [768, 432, 21, 65535]),
    ])
    comp_chain = GroupSpec("Composite", COLOR_COMP, [
        make_node("ImageToMask",          "to_mask",   ["red"]),
        # ImageCompositeMasked widgets: x, y, resize_source
        make_node("ImageCompositeMasked", "composite", [0, 0, False]),
    ])
    output = GroupSpec("Output", COLOR_OUTPUT, [
        make_node("SaveImage", "save_comp", ["unividx_J_composite_over_cyan"]),
    ])
    groups = [setup, cond, sampling, decode, comp_setup, comp_chain, output]
    nodes, placed_groups = layout(groups)
    links = assemble_links(nodes, [
        ("loader",     0, "sampler",    0, "UNIVIDX_MODEL"),
        ("task",       0, "sampler",    1, "UNIVIDX_TASK"),
        ("load_rgb",   0, "repeat_rgb", 0, "IMAGE"),
        ("repeat_rgb", 0, "sampler",    2, "IMAGE"),
        ("sampler",    0, "decoder",    0, "UNIVIDX_RESULT"),
        # Decoder slot 1 = alpha modality (P) -> ImageToMask
        ("decoder",    1, "to_mask",    0, "IMAGE"),
        # Decoder slot 2 = foreground (F) -> ImageCompositeMasked.source[1]
        ("empty_bg",   0, "composite",  0, "IMAGE"),    # destination
        ("decoder",    2, "composite",  1, "IMAGE"),    # source
        ("to_mask",    0, "composite",  2, "MASK"),     # mask
        ("composite",  0, "save_comp",  0, "IMAGE"),
    ])
    validate_no_overlap(nodes, placed_groups)
    return finalize(nodes, placed_groups, links,
                    name="UniVidX • R2PFB matte -> Composite over cyan background")


# ---------------------------------------------------------------------------
# Finalize: assemble the ComfyUI UI-format dict.
# ---------------------------------------------------------------------------

def finalize(nodes: list[Node], groups: list[Group],
             links: list[list], *, name: str) -> dict:
    last_node_id = max((n.id for n in nodes), default=0)
    last_link_id = max((l[0] for l in links), default=0)
    return {
        "id": name.replace(" ", "_").replace("•", "x").lower()[:64],
        "revision": 0,
        "last_node_id": last_node_id,
        "last_link_id": last_link_id,
        "nodes": [
            {
                "id": n.id,
                "type": n.type,
                "pos": list(n.pos),
                "size": list(n.size),
                "flags": {},
                "order": n.order,
                "mode": 0,
                "inputs": n.inputs,
                "outputs": n.outputs,
                "properties": {"Node name for S&R": n.type},
                "widgets_values": n.widgets_values,
            }
            for n in nodes
        ],
        "links": links,
        "groups": [
            {
                "title": g.title,
                "bounding": g.bounding,
                "color": g.color,
                "font_size": 24,
                "flags": {},
            }
            for g in groups
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.7, "offset": [0, 0]},
            "info": {"name": name, "author": "comfyui-unividx"},
        },
        "version": 0.4,
    }


# ---------------------------------------------------------------------------
# Drivers.
# ---------------------------------------------------------------------------

DEMOS = {
    "t2RAIN_basic":         workflow_t2RAIN_basic,
    "R2AIN_basic":          workflow_R2AIN_basic,
    "t2RPFB_basic":         workflow_t2RPFB_basic,
    "R2PFB_basic":          workflow_R2PFB_basic,
    "I_video_output":       workflow_I_video_output,
    "J_alpha_compositing":  workflow_J_alpha_compositing,
}


def _ui_to_api(ui: dict) -> dict:
    """Lightweight UI -> API conversion for programmatic queueing."""
    by_id = {n["id"]: n for n in ui["nodes"]}
    # Map link_id -> (src_node_id, src_slot)
    link_src: dict[int, tuple[int, int]] = {l[0]: (l[1], l[2]) for l in ui["links"]}

    api: dict[str, dict] = {}
    for n in ui["nodes"]:
        nid = str(n["id"])
        ctype = n["type"]
        widgets = list(n.get("widgets_values", []) or [])
        inputs: dict[str, Any] = {}
        # Resolve linked inputs
        for inp in n.get("inputs", []):
            link_id = inp.get("link")
            if link_id is None:
                continue
            src_id, src_slot = link_src[link_id]
            inputs[inp["name"]] = [str(src_id), src_slot]
        # Inline widget values (these are positional but ComfyUI's API format is
        # by-name, so we need to consult each node type's widget order)
        widget_names = {
            "UniVidXLoader":          ["variant", "dtype"],
            "UniVidXTaskMode":        ["mode"],
            "UniVidXSampler":         ["prompt", "negative_prompt", "num_inference_steps",
                                       "cfg_scale", "denoising_strength", "num_frames",
                                       "height", "width", "seed", "_seed_control", "tiled"],
            "UniVidXDecodeIntrinsic": [],
            "UniVidXDecodeAlpha":     [],
            "LoadImage":              ["image", "_upload"],
            "RepeatImageBatch":       ["amount"],
            "SaveImage":              ["filename_prefix"],
            "VHS_VideoCombine":       ["frame_rate", "loop_count", "filename_prefix",
                                       "format", "pingpong", "save_output"],
            "EmptyImage":             ["width", "height", "batch_size", "color"],
            "ImageToMask":            ["channel"],
            "ImageCompositeMasked":   ["x", "y", "resize_source"],
        }.get(ctype, [])
        for name, val in zip(widget_names, widgets):
            if name.startswith("_"):     # internal markers we skipped earlier
                continue
            inputs[name] = val
        api[nid] = {"class_type": ctype, "inputs": inputs}
    return api


def main() -> None:
    for name, builder in DEMOS.items():
        ui = builder()
        ui_path = HERE / f"{name}.json"
        ui_path.write_text(json.dumps(ui, indent=2, ensure_ascii=False), encoding="utf-8")
        api = _ui_to_api(ui)
        api_path = HERE / f"{name}_api.json"
        api_path.write_text(json.dumps(api, indent=2, ensure_ascii=False), encoding="utf-8")
        nodes_count = len(ui["nodes"])
        groups_count = len(ui["groups"])
        # Compute total canvas size for reporting
        max_x = max(n["pos"][0] + n["size"][0] for n in ui["nodes"])
        max_y = max(n["pos"][1] + n["size"][1] for n in ui["nodes"])
        print(f"  Wrote {name}: {nodes_count} nodes, {groups_count} groups, "
              f"canvas {max_x}x{max_y}")
    print("\nAll workflows passed overlap validation.")


if __name__ == "__main__":
    main()
