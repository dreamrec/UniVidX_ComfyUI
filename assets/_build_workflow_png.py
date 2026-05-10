"""Composite the t2RAIN workflow diagram as a PNG for the README.

Draws 5 ComfyUI-styled node cards (loader, task mode, sampler, decode intrinsic),
wires them with bezier connections in their socket-type colors, and pastes the
real result thumbnails into the decoder's 4 IMAGE outputs.

Run from repo root:  python assets/_build_workflow_png.py
"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1500, 760
BG = (14, 14, 28)
GRID_LINE = (28, 28, 44)
WIRE_BEZIER_STEPS = 32

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img, "RGBA")

def font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

for x in range(0, W, 40):
    d.line([(x, 0), (x, H)], fill=GRID_LINE, width=1)
for y in range(0, H, 40):
    d.line([(0, y), (W, y)], fill=GRID_LINE, width=1)

d.text((30, 18), "t2RAIN - text-to-(RGB+Albedo+Irradiance+Normal) workflow",
       fill=(232, 232, 240), font=font(22, True))
d.text((30, 48), "5 nodes - 21 frames - 480x640 - ~10 min on RTX 5090",
       fill=(128, 134, 162), font=font(14))

def vgrad(x0, y0, x1, y1, c0, c1):
    h = y1 - y0
    for i in range(h):
        t = i / max(1, h - 1)
        rr = int(c0[0] + (c1[0] - c0[0]) * t)
        gg = int(c0[1] + (c1[1] - c0[1]) * t)
        bb = int(c0[2] + (c1[2] - c0[2]) * t)
        d.line([(x0, y0 + i), (x1, y0 + i)], fill=(rr, gg, bb))

def card(x, y, w, h, title, c0, c1, accent_label):
    d.rounded_rectangle([x, y, x + w, y + h], radius=10,
                        fill=(38, 37, 52), outline=(68, 67, 87), width=2)
    bar_h = 32
    d.rounded_rectangle([x, y, x + w, y + bar_h + 8], radius=10, fill=c0)
    vgrad(x + 1, y + 1, x + w - 1, y + bar_h, c0, c1)
    d.rectangle([x, y + bar_h, x + w, y + bar_h + 1], fill=(0, 0, 0, 80))
    d.text((x + 14, y + 8), title, fill=(255, 255, 255), font=font(15, True))
    bbox = d.textbbox((0, 0), accent_label, font=font(11))
    d.text((x + w - 14 - (bbox[2] - bbox[0]), y + 11),
           accent_label, fill=(255, 255, 255, 200), font=font(11))

def socket(cx, cy, color, label=None, side="left", value_color=None):
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6],
              fill=(28, 27, 39), outline=color, width=2)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
    if label:
        if side == "left":
            d.text((cx + 12, cy - 8), label,
                   fill=value_color or (220, 220, 235), font=font(12, True))
        else:
            bbox = d.textbbox((0, 0), label, font=font(12, True))
            d.text((cx - 12 - (bbox[2] - bbox[0]), cy - 8),
                   label, fill=value_color or (220, 220, 235), font=font(12, True))

def widget_row(x, y, w, label, value, value_color=(253, 230, 138)):
    d.text((x + 12, y + 4), label, fill=(156, 163, 175), font=font(11))
    rect_x0 = x + w - 92
    d.rounded_rectangle([rect_x0, y + 1, x + w - 12, y + 19], radius=3,
                        fill=(59, 58, 77), outline=(82, 81, 106), width=1)
    bbox = d.textbbox((0, 0), str(value), font=font(11, True))
    d.text((rect_x0 + (80 - (bbox[2] - bbox[0])) // 2, y + 5),
           str(value), fill=value_color, font=font(11, True))

def wire(x0, y0, x1, y1, color, width=3):
    dx = max(60, abs(x1 - x0) * 0.5)
    cp1 = (x0 + dx, y0)
    cp2 = (x1 - dx, y1)
    pts = []
    for i in range(WIRE_BEZIER_STEPS + 1):
        t = i / WIRE_BEZIER_STEPS
        u = 1 - t
        bx = u * u * u * x0 + 3 * u * u * t * cp1[0] + 3 * u * t * t * cp2[0] + t * t * t * x1
        by = u * u * u * y0 + 3 * u * u * t * cp1[1] + 3 * u * t * t * cp2[1] + t * t * t * y1
        pts.append((bx, by))
    glow = color + (60,) if len(color) == 3 else color
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=glow, width=width + 4)
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=color, width=width)

# === Loader ===
LX, LY, LW, LH = 40, 110, 260, 130
card(LX, LY, LW, LH, "UniVidX Loader", (139, 92, 246), (109, 40, 217), "UNIVIDX")
widget_row(LX, LY + 50, LW, "variant", "intrinsic")
widget_row(LX, LY + 74, LW, "dtype", "bfloat16")
loader_out = (LX + LW, LY + 108)
socket(*loader_out, color=(168, 85, 247), label="model",
       side="right", value_color=(196, 181, 253))

# === TaskMode ===
TX, TY, TW, TH = 40, 280, 260, 100
card(TX, TY, TW, TH, "UniVidX Task Mode", (20, 184, 166), (15, 118, 110), "UNIVIDX")
widget_row(TX, TY + 50, TW, "mode", "t2RAIN")
task_out = (TX + TW, TY + 82)
socket(*task_out, color=(20, 184, 166), label="task",
       side="right", value_color=(94, 234, 212))

# === Sampler ===
SX, SY, SW, SH = 420, 90, 340, 540
card(SX, SY, SW, SH, "UniVidX Sampler", (249, 115, 22), (194, 65, 12), "UNIVIDX")
sampler_model = (SX, SY + 58)
socket(*sampler_model, color=(168, 85, 247), label="model",
       side="left", value_color=(196, 181, 253))
sampler_task = (SX, SY + 84)
socket(*sampler_task, color=(20, 184, 166), label="task",
       side="left", value_color=(94, 234, 212))
d.text((SX + 18, SY + 108), "- optional IMAGE inputs -",
       fill=(107, 114, 128), font=font(10))
opt_y0 = SY + 132
opt_labels = ["rgb", "albedo", "irradiance", "normal", "pha", "fgr", "bgr"]
for i, lbl in enumerate(opt_labels):
    cy = opt_y0 + i * 18
    socket(SX, cy, color=(86, 221, 86), side="left")
    d.text((SX + 12, cy - 7), lbl, fill=(134, 239, 172), font=font(11, True))

widgets = [
    ("prompt", "[multiline]", (203, 213, 225)),
    ("negative_prompt", "[chinese]", (203, 213, 225)),
    ("num_inference_steps", "50", (253, 230, 138)),
    ("cfg_scale", "5.00", (253, 230, 138)),
    ("denoising_strength", "1.00", (253, 230, 138)),
    ("num_frames", "21", (253, 230, 138)),
    ("height", "480", (253, 230, 138)),
    ("width", "640", (253, 230, 138)),
    ("seed", "1", (253, 230, 138)),
    ("tiled", "True", (134, 239, 172)),
]
for i, (lbl, val, vc) in enumerate(widgets):
    widget_row(SX + 130, SY + 130 + i * 30, 200, lbl, val, value_color=vc)
sampler_out = (SX + SW, SY + SH - 30)
socket(*sampler_out, color=(236, 72, 153), label="result",
       side="right", value_color=(251, 207, 232))

# === Decode Intrinsic ===
DX, DY, DW, DH = 820, 200, 260, 200
card(DX, DY, DW, DH, "Decode Intrinsic", (34, 197, 94), (21, 128, 61), "UNIVIDX")
decode_in = (DX, DY + 58)
socket(*decode_in, color=(236, 72, 153), label="result",
       side="left", value_color=(251, 207, 232))
out_labels = ["rgb", "albedo", "irradiance", "normal"]
out_y0 = DY + 90
out_sockets = []
for i, lbl in enumerate(out_labels):
    cy = out_y0 + i * 26
    socket(DX + DW, cy, color=(86, 221, 86), label=lbl,
           side="right", value_color=(134, 239, 172))
    out_sockets.append((DX + DW, cy))

# === Wires ===
PURPLE = (168, 85, 247)
TEAL = (20, 184, 166)
PINK = (236, 72, 153)
GREEN = (86, 221, 86)
wire(loader_out[0], loader_out[1], sampler_model[0], sampler_model[1], PURPLE)
wire(task_out[0], task_out[1], sampler_task[0], sampler_task[1], TEAL)
wire(sampler_out[0], sampler_out[1], decode_in[0], decode_in[1], PINK)

# === Result thumbnails ===
THUMB_H = 154
THUMB_W = int(THUMB_H * 4 / 3)
gap = 14
THX = W - THUMB_W - 30
THY0 = 90
thumb_paths = [
    ("RGB",        "assets/results/intrinsic_rgb.jpg"),
    ("ALBEDO",     "assets/results/intrinsic_albedo.jpg"),
    ("IRRADIANCE", "assets/results/intrinsic_irradiance.jpg"),
    ("NORMAL",     "assets/results/intrinsic_normal.jpg"),
]
for i, (label, p) in enumerate(thumb_paths):
    if not os.path.exists(p):
        continue
    thumb = Image.open(p).convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
    ty = THY0 + i * (THUMB_H + gap)
    d.rounded_rectangle([THX - 3, ty - 3, THX + THUMB_W + 3, ty + THUMB_H + 3],
                        radius=4, outline=(82, 81, 106), width=2)
    img.paste(thumb, (THX, ty))
    bbox = d.textbbox((0, 0), label, font=font(11, True))
    tw = bbox[2] - bbox[0]
    d.rectangle([THX + 8, ty + 8, THX + 8 + tw + 10, ty + 8 + 18],
                fill=(0, 0, 0, 210))
    d.text((THX + 13, ty + 10), label, fill=(255, 255, 255), font=font(11, True))
    sock = out_sockets[i]
    wire(sock[0], sock[1], THX - 3, ty + THUMB_H // 2, GREEN)

d.text((30, H - 28),
       "github.com/dreamrec/UniVidX_ComfyUI - workflow: examples/t2RAIN_basic.json",
       fill=(94, 99, 122), font=font(11))

out_path = "assets/workflow_t2RAIN.png"
img.save(out_path, optimize=True)
print(f"Saved {out_path} {os.path.getsize(out_path)//1024} KB")
