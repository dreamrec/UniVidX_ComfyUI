"""Compose intrinsic_quad and alpha_quad images from the LTX_2.3_t2v_00239_
R2AIN + R2PFB run outputs. Picks frame 11 (mid-clip) by default.

Run from repo root:  python assets/_build_LTX_quads.py
"""
from PIL import Image, ImageDraw, ImageFont
import os
import sys

OUT = "C:/Users/dr5090/Documents/ComfyUI/output"
DEST = "assets/results"
FRAME = 11  # 1-indexed; LTX video is 21 frames so 11 is the middle


def font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def quad(out_name: str, panels: list[tuple[str, str]], cell_aspect: float = 4 / 3) -> str:
    """panels: list of (label, png_path) tuples, top-left to bottom-right (row-major)."""
    cell_h = 320
    cell_w = int(cell_h * cell_aspect)
    W, H = cell_w * 2, cell_h * 2
    img = Image.new("RGB", (W, H), (20, 20, 28))
    d = ImageDraw.Draw(img)
    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
    for (label, p), pos in zip(panels, positions):
        if not os.path.exists(p):
            print(f"  MISSING: {p}", file=sys.stderr)
            continue
        thumb = Image.open(p).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        img.paste(thumb, pos)
        bbox = d.textbbox((0, 0), label, font=font(16, True))
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = pos
        d.rectangle([x + 8, y + 8, x + 8 + tw + 12, y + 8 + th + 8], fill=(0, 0, 0))
        d.text((x + 14, y + 12), label, fill=(255, 255, 255), font=font(16, True))
    d.line([(cell_w, 0), (cell_w, H)], fill=(40, 40, 55), width=2)
    d.line([(0, cell_h), (W, cell_h)], fill=(40, 40, 55), width=2)
    out_path = f"{DEST}/{out_name}"
    img.save(out_path, quality=88, optimize=True)
    print(f"Saved {out_path} ({os.path.getsize(out_path) // 1024} KB)")
    return out_path


def per_modality(out_name: str, src_path: str, w: int = 400, h: int = 300) -> str:
    if not os.path.exists(src_path):
        print(f"  MISSING: {src_path}", file=sys.stderr)
        return ""
    img = Image.open(src_path).convert("RGB").resize((w, h), Image.LANCZOS)
    out_path = f"{DEST}/{out_name}"
    img.save(out_path, quality=88, optimize=True)
    print(f"Saved {out_path} ({os.path.getsize(out_path) // 1024} KB)")
    return out_path


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    f = FRAME

    # === Intrinsic R2AIN: rgb is the conditioning input (not regenerated),
    # so we paste the conditioning input itself in the RGB cell to make the
    # comparison legible. ComfyUI saves the input via VHS - it isn't a normal
    # output - so we extract the same frame from the source MP4 instead.
    rgb_input = f"{DEST}/_ltx_input_{f:05d}.png"
    if not os.path.exists(rgb_input):
        import cv2
        cap = cv2.VideoCapture("C:/Users/dr5090/Documents/ComfyUI/input/LTX_2.3_t2v_00239_.mp4")
        # The workflow uses select_every_nth=23, so output frame i corresponds
        # to source frame (i-1)*23. Frame 11 -> source frame 230.
        cap.set(cv2.CAP_PROP_POS_FRAMES, (f - 1) * 23)
        ok, bgr = cap.read()
        if ok:
            import numpy as np
            rgb = bgr[:, :, ::-1]
            # Center-crop and resize to 640x480 to match the workflow's framing
            from PIL import Image as PI
            pil = PI.fromarray(rgb)
            pw, ph = pil.size
            target_aspect = 640 / 480
            src_aspect = pw / ph
            if src_aspect > target_aspect:
                new_w = int(ph * target_aspect)
                left = (pw - new_w) // 2
                pil = pil.crop((left, 0, left + new_w, ph))
            else:
                new_h = int(pw / target_aspect)
                top = (ph - new_h) // 2
                pil = pil.crop((0, top, pw, top + new_h))
            pil = pil.resize((640, 480), PI.LANCZOS)
            pil.save(rgb_input)

    quad("LTX_intrinsic_quad.jpg", [
        ("RGB (input)", rgb_input),
        ("ALBEDO",     f"{OUT}/unividx_LTX_R2AIN_albedo_{f:05d}_.png"),
        ("IRRADIANCE", f"{OUT}/unividx_LTX_R2AIN_irradiance_{f:05d}_.png"),
        ("NORMAL",     f"{OUT}/unividx_LTX_R2AIN_normal_{f:05d}_.png"),
    ], cell_aspect=4/3)

    # Per-modality 400x300 thumbs for the workflow PNG composer
    per_modality("LTX_intrinsic_rgb.jpg",        rgb_input)
    per_modality("LTX_intrinsic_albedo.jpg",     f"{OUT}/unividx_LTX_R2AIN_albedo_{f:05d}_.png")
    per_modality("LTX_intrinsic_irradiance.jpg", f"{OUT}/unividx_LTX_R2AIN_irradiance_{f:05d}_.png")
    per_modality("LTX_intrinsic_normal.jpg",     f"{OUT}/unividx_LTX_R2AIN_normal_{f:05d}_.png")

    # === Alpha R2PFB: same trick - composite_rgb is the input, so use the
    # actual conditioning frame (resampled to 768x432 alpha aspect)
    rgb_input_alpha = f"{DEST}/_ltx_input_alpha_{f:05d}.png"
    if not os.path.exists(rgb_input_alpha):
        import cv2
        from PIL import Image as PI
        cap = cv2.VideoCapture("C:/Users/dr5090/Documents/ComfyUI/input/LTX_2.3_t2v_00239_.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, (f - 1) * 23)
        ok, bgr = cap.read()
        if ok:
            rgb = bgr[:, :, ::-1]
            pil = PI.fromarray(rgb)
            pw, ph = pil.size
            target_aspect = 768 / 432
            src_aspect = pw / ph
            if src_aspect > target_aspect:
                new_w = int(ph * target_aspect)
                left = (pw - new_w) // 2
                pil = pil.crop((left, 0, left + new_w, ph))
            else:
                new_h = int(pw / target_aspect)
                top = (ph - new_h) // 2
                pil = pil.crop((0, top, pw, top + new_h))
            pil = pil.resize((768, 432), PI.LANCZOS)
            pil.save(rgb_input_alpha)

    quad("LTX_alpha_quad.jpg", [
        ("COMPOSITE (input)", rgb_input_alpha),
        ("ALPHA MATTE",       f"{OUT}/unividx_LTX_R2PFB_matte_{f:05d}_.png"),
        ("FOREGROUND",        f"{OUT}/unividx_LTX_R2PFB_foreground_{f:05d}_.png"),
        ("BACKGROUND",        f"{OUT}/unividx_LTX_R2PFB_background_{f:05d}_.png"),
    ], cell_aspect=768/432)


if __name__ == "__main__":
    main()
