"""
Build the t2RAIN smoke-test workflow in two formats:

1. examples/t2RAIN_basic_api.json — API format (flat {id: {class_type, inputs}}).
   This is what /prompt accepts. Use _smoke_runner.py to execute it.

2. examples/t2RAIN_basic.json — UI format with positions, sizes, links, and
   four colour-coded groups: Model Setup / Sampling / Decoding / Outputs.
   Drag this file into ComfyUI's canvas to see the laid-out graph.

Layout philosophy (left-to-right, dataflow direction):
    [Setup]   ->   [Sampling]   ->   [Decoding]   ->   [Outputs]
    Loader        Sampler         Decoder       4× SaveImage
    TaskMode

Spacing between groups: 40 px gap. Padding inside group: 30 px on sides,
50 px under the title. Vertical row gap between stacked nodes: 50 px.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)

PROMPT = "a small hedgehog wearing a chef hat in a tiny kitchen"

# ---------------------------------------------------------------------------
# API format
# ---------------------------------------------------------------------------
api = {
    "1": {
        "class_type": "UniVidXLoader",
        "inputs": {"variant": "intrinsic", "dtype": "bfloat16"},
    },
    "2": {
        "class_type": "UniVidXTaskMode",
        "inputs": {"mode": "t2RAIN"},
    },
    "3": {
        "class_type": "UniVidXSampler",
        "inputs": {
            "model": ["1", 0],
            "task": ["2", 0],
            "prompt": PROMPT,
            "negative_prompt": NEG,
            "num_inference_steps": 20,
            "cfg_scale": 5.0,
            "denoising_strength": 1.0,
            "num_frames": 21,
            "height": 480,
            "width": 640,
            "seed": 42,
            "tiled": True,
        },
    },
    "4": {
        "class_type": "UniVidXDecodeIntrinsic",
        "inputs": {"result": ["3", 0]},
    },
    "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "unividx_t2RAIN_rgb"}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["4", 1], "filename_prefix": "unividx_t2RAIN_albedo"}},
    "7": {"class_type": "SaveImage", "inputs": {"images": ["4", 2], "filename_prefix": "unividx_t2RAIN_irradiance"}},
    "8": {"class_type": "SaveImage", "inputs": {"images": ["4", 3], "filename_prefix": "unividx_t2RAIN_normal"}},
}

(HERE / "t2RAIN_basic_api.json").write_text(json.dumps(api, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote API format: {HERE / 't2RAIN_basic_api.json'}  ({len(api)} nodes)")

# ---------------------------------------------------------------------------
# UI format with positions, sizes, groups
# ---------------------------------------------------------------------------
# Coordinates and sizes were chosen for visual clarity, not pixel-perfection.
# The Sampler is the tallest node (12+ widgets); other nodes align to its top.

GROUP_GAP = 40       # horizontal gap between groups
NODE_SIDE_PAD = 30   # padding inside group (left/right of nodes)
NODE_TOP_PAD = 50    # padding inside group (under the title bar, before first node)
ROW_GAP = 50         # vertical gap between stacked nodes

# Group 1: Model Setup — Loader (top), TaskMode (below)
G1_X = 40
G1_W = 360
LOADER_POS = [G1_X + NODE_SIDE_PAD, NODE_TOP_PAD + 30]
LOADER_SIZE = [300, 130]
TASKMODE_POS = [G1_X + NODE_SIDE_PAD, LOADER_POS[1] + LOADER_SIZE[1] + ROW_GAP]
TASKMODE_SIZE = [300, 80]
G1_H = TASKMODE_POS[1] + TASKMODE_SIZE[1] + 30  # bottom padding

# Group 2: Sampling — Sampler (tall, ~800)
G2_X = G1_X + G1_W + GROUP_GAP
G2_W = 460
SAMPLER_POS = [G2_X + NODE_SIDE_PAD, NODE_TOP_PAD + 30]
SAMPLER_SIZE = [400, 800]
G2_H = SAMPLER_POS[1] + SAMPLER_SIZE[1] + 30

# Group 3: Decoding — DecodeIntrinsic
G3_X = G2_X + G2_W + GROUP_GAP
G3_W = 320
DECODER_POS = [G3_X + NODE_SIDE_PAD, NODE_TOP_PAD + 30]
DECODER_SIZE = [260, 130]
G3_H = DECODER_POS[1] + DECODER_SIZE[1] + 30

# Group 4: Outputs — 4 SaveImage stacked
G4_X = G3_X + G3_W + GROUP_GAP
G4_W = 380
SAVE_SIZE = [300, 110]
SAVE_POSITIONS = [
    [G4_X + NODE_SIDE_PAD, NODE_TOP_PAD + 30],
    [G4_X + NODE_SIDE_PAD, NODE_TOP_PAD + 30 + (SAVE_SIZE[1] + ROW_GAP)],
    [G4_X + NODE_SIDE_PAD, NODE_TOP_PAD + 30 + 2 * (SAVE_SIZE[1] + ROW_GAP)],
    [G4_X + NODE_SIDE_PAD, NODE_TOP_PAD + 30 + 3 * (SAVE_SIZE[1] + ROW_GAP)],
]
G4_H = SAVE_POSITIONS[3][1] + SAVE_SIZE[1] + 30

# All groups have the same height for visual symmetry — pick the max.
GROUP_H = max(G1_H, G2_H, G3_H, G4_H)

# Group colours (matching ComfyUI's standard palette)
COLORS = {
    "setup": "#3a7c5a",      # green — preparation
    "sampling": "#4a5fc1",   # blue — work
    "decoding": "#c14a8e",   # magenta — transformation
    "outputs": "#c97a3a",    # orange — sinks
}

ui = {
    "id": "unividx-t2rain-smoke",
    "revision": 0,
    "last_node_id": 8,
    "last_link_id": 6,
    "nodes": [
        {
            "id": 1,
            "type": "UniVidXLoader",
            "pos": LOADER_POS,
            "size": LOADER_SIZE,
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "model", "type": "UNIVIDX_MODEL", "links": [1], "slot_index": 0}],
            "properties": {"Node name for S&R": "UniVidXLoader"},
            "widgets_values": ["intrinsic", "bfloat16"],
        },
        {
            "id": 2,
            "type": "UniVidXTaskMode",
            "pos": TASKMODE_POS,
            "size": TASKMODE_SIZE,
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "task", "type": "UNIVIDX_TASK", "links": [2], "slot_index": 0}],
            "properties": {"Node name for S&R": "UniVidXTaskMode"},
            "widgets_values": ["t2RAIN"],
        },
        {
            "id": 3,
            "type": "UniVidXSampler",
            "pos": SAMPLER_POS,
            "size": SAMPLER_SIZE,
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "UNIVIDX_MODEL", "link": 1},
                {"name": "task",  "type": "UNIVIDX_TASK",  "link": 2},
                {"name": "rgb",        "type": "IMAGE", "link": None, "shape": 7},
                {"name": "albedo",     "type": "IMAGE", "link": None, "shape": 7},
                {"name": "irradiance", "type": "IMAGE", "link": None, "shape": 7},
                {"name": "normal",     "type": "IMAGE", "link": None, "shape": 7},
                {"name": "pha", "type": "IMAGE", "link": None, "shape": 7},
                {"name": "fgr", "type": "IMAGE", "link": None, "shape": 7},
                {"name": "bgr", "type": "IMAGE", "link": None, "shape": 7},
            ],
            "outputs": [{"name": "result", "type": "UNIVIDX_RESULT", "links": [3], "slot_index": 0}],
            "properties": {"Node name for S&R": "UniVidXSampler"},
            "widgets_values": [
                PROMPT, NEG, 20, 5.0, 1.0, 21, 480, 640, 42, "fixed", True,
            ],
        },
        {
            "id": 4,
            "type": "UniVidXDecodeIntrinsic",
            "pos": DECODER_POS,
            "size": DECODER_SIZE,
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [{"name": "result", "type": "UNIVIDX_RESULT", "link": 3}],
            "outputs": [
                {"name": "rgb",        "type": "IMAGE", "links": [4], "slot_index": 0},
                {"name": "albedo",     "type": "IMAGE", "links": [5], "slot_index": 1},
                {"name": "irradiance", "type": "IMAGE", "links": [6], "slot_index": 2},
                {"name": "normal",     "type": "IMAGE", "links": [7], "slot_index": 3},
            ],
            "properties": {"Node name for S&R": "UniVidXDecodeIntrinsic"},
            "widgets_values": [],
        },
        # SaveImage × 4
        *[
            {
                "id": 5 + i,
                "type": "SaveImage",
                "pos": SAVE_POSITIONS[i],
                "size": SAVE_SIZE,
                "flags": {},
                "order": 4 + i,
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 4 + i}],
                "outputs": [],
                "properties": {"Node name for S&R": "SaveImage"},
                "widgets_values": [
                    f"unividx_t2RAIN_{m}"
                    for m in ["rgb", "albedo", "irradiance", "normal"]
                ][i : i + 1],
            }
            for i in range(4)
        ],
    ],
    "links": [
        # [link_id, src_node, src_slot, tgt_node, tgt_slot, type]
        [1, 1, 0, 3, 0, "UNIVIDX_MODEL"],
        [2, 2, 0, 3, 1, "UNIVIDX_TASK"],
        [3, 3, 0, 4, 0, "UNIVIDX_RESULT"],
        [4, 4, 0, 5, 0, "IMAGE"],
        [5, 4, 1, 6, 0, "IMAGE"],
        [6, 4, 2, 7, 0, "IMAGE"],
        [7, 4, 3, 8, 0, "IMAGE"],
    ],
    "groups": [
        {
            "title": "Model Setup",
            "bounding": [G1_X, 20, G1_W, GROUP_H],
            "color": COLORS["setup"],
            "font_size": 24,
            "flags": {},
        },
        {
            "title": "Sampling",
            "bounding": [G2_X, 20, G2_W, GROUP_H],
            "color": COLORS["sampling"],
            "font_size": 24,
            "flags": {},
        },
        {
            "title": "Decoding",
            "bounding": [G3_X, 20, G3_W, GROUP_H],
            "color": COLORS["decoding"],
            "font_size": 24,
            "flags": {},
        },
        {
            "title": "Outputs",
            "bounding": [G4_X, 20, G4_W, GROUP_H],
            "color": COLORS["outputs"],
            "font_size": 24,
            "flags": {},
        },
    ],
    "config": {},
    "extra": {
        "ds": {"scale": 0.7, "offset": [0, 0]},
        "info": {
            "name": "UniVidX • t2RAIN smoke test",
            "author": "comfyui-unividx",
            "description": (
                "Strategy A smoke test for UniVidX intrinsic decomposition. "
                "Generates RGB / Albedo / Irradiance / Normal videos from a "
                "text prompt at 480x640x21f. Uses 20 inference steps for fast "
                "validation; bump to 50 for final quality."
            ),
        },
    },
    "version": 0.4,
}

(HERE / "t2RAIN_basic.json").write_text(json.dumps(ui, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote UI format:  {HERE / 't2RAIN_basic.json'}  ({len(ui['nodes'])} nodes, {len(ui['groups'])} groups)")

# Layout summary
print("\nLayout:")
print(f"  Group 1 'Model Setup' : x={G1_X:5d}  w={G1_W} h={GROUP_H}  (Loader + TaskMode)")
print(f"  Group 2 'Sampling'    : x={G2_X:5d}  w={G2_W} h={GROUP_H}  (Sampler)")
print(f"  Group 3 'Decoding'    : x={G3_X:5d}  w={G3_W} h={GROUP_H}  (DecodeIntrinsic)")
print(f"  Group 4 'Outputs'     : x={G4_X:5d}  w={G4_W} h={GROUP_H}  (4x SaveImage)")
print(f"  Total canvas width    : {G4_X + G4_W + 40}")
