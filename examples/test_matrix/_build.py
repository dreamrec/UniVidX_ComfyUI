"""
Build the comprehensive test-matrix workflow JSONs.

Each test exercises a different code path. All use tiny config (256x256,
5 frames, 3 steps) for fast iteration; the alpha variant gets a cold load
on first use, but subsequent alpha runs hit the cache.

Output: examples/test_matrix/<test_id>.json (one per test)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEG_SHORT = "色调艳丽，过曝，静态，细节模糊不清"
PROMPT = "a small hedgehog wearing a chef hat in a tiny kitchen"
RES = 256
FRAMES = 5
STEPS = 3
SEED = 42

INPUT_RGB = "unividx_R2AIN_input.png"
INPUT_ALBEDO = "unividx_input_albedo.png"
INPUT_IRRADIANCE = "unividx_input_irradiance.png"


def _sampler_inputs(mode, *, conditioning_links=None):
    """Common sampler kwargs. conditioning_links maps modality_name -> [node_id, slot]."""
    inputs = {
        "model": ["1", 0],
        "task": ["2", 0],
        "prompt": PROMPT,
        "negative_prompt": NEG_SHORT,
        "num_inference_steps": STEPS,
        "cfg_scale": 5.0,
        "denoising_strength": 1.0,
        "num_frames": FRAMES,
        "height": RES,
        "width": RES,
        "seed": SEED,
        "tiled": True,
    }
    if conditioning_links:
        inputs.update(conditioning_links)
    return inputs


def build_image_input_chain(load_id, repeat_id, filename):
    """Pair of nodes: LoadImage(filename) -> RepeatImageBatch(amount=FRAMES)."""
    return {
        load_id: {
            "class_type": "LoadImage",
            "inputs": {"image": filename},
        },
        repeat_id: {
            "class_type": "RepeatImageBatch",
            "inputs": {"image": [load_id, 0], "amount": FRAMES},
        },
    }


def build_intrinsic_decoder_and_saves(decode_id, save_prefix):
    """UniVidXDecodeIntrinsic + 4 SaveImage nodes."""
    nodes = {
        decode_id: {
            "class_type": "UniVidXDecodeIntrinsic",
            "inputs": {"result": ["3", 0]},
        }
    }
    save_id = 5
    for i, mod in enumerate(["rgb", "albedo", "irradiance", "normal"]):
        nodes[str(save_id + i)] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": [decode_id, i],
                "filename_prefix": f"{save_prefix}_{mod}",
            },
        }
    return nodes


def build_alpha_decoder_and_saves(decode_id, save_prefix):
    """UniVidXDecodeAlpha + 4 SaveImage nodes."""
    nodes = {
        decode_id: {
            "class_type": "UniVidXDecodeAlpha",
            "inputs": {"result": ["3", 0]},
        }
    }
    save_id = 5
    for i, mod in enumerate(["composite_rgb", "alpha", "foreground", "background"]):
        nodes[str(save_id + i)] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": [decode_id, i],
                "filename_prefix": f"{save_prefix}_{mod}",
            },
        }
    return nodes


# Test C: RA2IN — intrinsic, RGB+Albedo conditioning, generate I+N
def build_C_RA2IN():
    workflow = {
        "1": {"class_type": "UniVidXLoader",   "inputs": {"variant": "intrinsic", "dtype": "bfloat16"}},
        "2": {"class_type": "UniVidXTaskMode", "inputs": {"mode": "RA2IN"}},
    }
    workflow.update(build_image_input_chain("10", "11", INPUT_RGB))
    workflow.update(build_image_input_chain("12", "13", INPUT_ALBEDO))
    workflow["3"] = {
        "class_type": "UniVidXSampler",
        "inputs": _sampler_inputs("RA2IN", conditioning_links={
            "rgb":    ["11", 0],
            "albedo": ["13", 0],
        }),
    }
    workflow.update(build_intrinsic_decoder_and_saves("4", "matrix_C_RA2IN"))
    return workflow


# Test D: RAI2N — intrinsic, RGB+Albedo+Irradiance conditioning, generate Normal only
def build_D_RAI2N():
    workflow = {
        "1": {"class_type": "UniVidXLoader",   "inputs": {"variant": "intrinsic", "dtype": "bfloat16"}},
        "2": {"class_type": "UniVidXTaskMode", "inputs": {"mode": "RAI2N"}},
    }
    workflow.update(build_image_input_chain("10", "11", INPUT_RGB))
    workflow.update(build_image_input_chain("12", "13", INPUT_ALBEDO))
    workflow.update(build_image_input_chain("14", "15", INPUT_IRRADIANCE))
    workflow["3"] = {
        "class_type": "UniVidXSampler",
        "inputs": _sampler_inputs("RAI2N", conditioning_links={
            "rgb":        ["11", 0],
            "albedo":     ["13", 0],
            "irradiance": ["15", 0],
        }),
    }
    workflow.update(build_intrinsic_decoder_and_saves("4", "matrix_D_RAI2N"))
    return workflow


# Test E: t2RPFB — alpha variant, text-to-all-4 alpha modalities
def build_E_t2RPFB():
    workflow = {
        "1": {"class_type": "UniVidXLoader",   "inputs": {"variant": "alpha", "dtype": "bfloat16"}},
        "2": {"class_type": "UniVidXTaskMode", "inputs": {"mode": "t2RPFB"}},
        "3": {"class_type": "UniVidXSampler",  "inputs": _sampler_inputs("t2RPFB")},
    }
    workflow.update(build_alpha_decoder_and_saves("4", "matrix_E_t2RPFB"))
    return workflow


# Test F: R2PFB — alpha variant, RGB-conditioned, generate P+F+B
def build_F_R2PFB():
    workflow = {
        "1": {"class_type": "UniVidXLoader",   "inputs": {"variant": "alpha", "dtype": "bfloat16"}},
        "2": {"class_type": "UniVidXTaskMode", "inputs": {"mode": "R2PFB"}},
    }
    workflow.update(build_image_input_chain("10", "11", INPUT_RGB))
    workflow["3"] = {
        "class_type": "UniVidXSampler",
        "inputs": _sampler_inputs("R2PFB", conditioning_links={
            "rgb": ["11", 0],
        }),
    }
    workflow.update(build_alpha_decoder_and_saves("4", "matrix_F_R2PFB"))
    return workflow


# Test G: error path — variant mismatch (intrinsic loader + alpha-family mode)
# Note: ComfyUI's queue validator rejects prompts with no output node, so we
# include a SaveImage so the prompt is accepted; the Sampler's runtime check
# then raises before any image is actually written.
def build_G_error_family_mismatch():
    return {
        "1": {"class_type": "UniVidXLoader",   "inputs": {"variant": "intrinsic", "dtype": "bfloat16"}},
        "2": {"class_type": "UniVidXTaskMode", "inputs": {"mode": "t2RPFB"}},  # alpha-family
        "3": {"class_type": "UniVidXSampler",  "inputs": _sampler_inputs("t2RPFB")},
        "4": {"class_type": "UniVidXDecodeIntrinsic", "inputs": {"result": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0],
                                                       "filename_prefix": "matrix_G_should_not_exist"}},
    }


# Test H: error path — missing required input (R2AIN needs rgb, none provided)
def build_H_error_missing_input():
    return {
        "1": {"class_type": "UniVidXLoader",   "inputs": {"variant": "intrinsic", "dtype": "bfloat16"}},
        "2": {"class_type": "UniVidXTaskMode", "inputs": {"mode": "R2AIN"}},
        "3": {"class_type": "UniVidXSampler",  "inputs": _sampler_inputs("R2AIN")},  # no rgb wired
        "4": {"class_type": "UniVidXDecodeIntrinsic", "inputs": {"result": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0],
                                                       "filename_prefix": "matrix_H_should_not_exist"}},
    }


TEST_BUILDERS = {
    "C_RA2IN": build_C_RA2IN,
    "D_RAI2N": build_D_RAI2N,
    "E_t2RPFB": build_E_t2RPFB,
    "F_R2PFB": build_F_R2PFB,
    "G_error_family_mismatch": build_G_error_family_mismatch,
    "H_error_missing_input": build_H_error_missing_input,
}


def main():
    for name, builder in TEST_BUILDERS.items():
        wf = builder()
        out = HERE / f"{name}.json"
        out.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
        node_count = len(wf)
        print(f"  Wrote {out.name}: {node_count} nodes")


if __name__ == "__main__":
    main()
