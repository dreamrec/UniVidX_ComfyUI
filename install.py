"""Install/setup helper for the UniVidX_ComfyUI custom-node pack.

Mirrors the convention used by dreamrec's other ComfyUI repos: verify the
vendored UniVidX submodule is at the pinned commit, copy bundled demo
workflows into ComfyUI's user workflow directory, and print a friendly
status message about the (large) model files the user must download
manually.

This script is idempotent and safe to re-run.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR_UNIVIDX = ROOT / "vendor" / "UniVidX"
EXAMPLES = ROOT / "examples"

# Pinned commit of the UniVidX submodule. Update this when bumping the submodule
# (and verify the test matrix still passes — UniVidX's internal layout is not
# stable upstream).
PINNED_UNIVIDX_COMMIT = "382b9002757f4d5f04a90ec23b784dce11d56221"

# UI-format demo workflow JSONs that get copied into the user's workflow
# directory so they appear in the ComfyUI sidebar without manual import.
# Only UI-format files (with `nodes` + `groups` arrays) belong here — the
# *_api.json files are programmatic-queue payloads and are skipped.
# `R2AIN_basic.json` and `R2PFB_basic.json` were removed in 0.2.0; the
# replacement video-conditioned workflows ship as API-format only
# (`R2AIN_video_api.json` / `R2PFB_video_api.json`) and are intentionally
# omitted from auto-copy — users drop them on the canvas manually.
DEMO_WORKFLOW_NAMES = (
    "t2RAIN_basic.json",          # text -> RGB+A+I+N (intrinsic)
    "t2RPFB_basic.json",          # text -> R+P+F+B (alpha)
    "I_video_output.json",        # t2RAIN -> 4x MP4 via VHS_VideoCombine
    "J_alpha_compositing.json",   # R2PFB matte -> ImageCompositeMasked
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_input_directory() -> Path | None:
    """Find ComfyUI/input/. Used by other dreamrec nodes that ship demo media."""
    try:
        import folder_paths  # type: ignore[import-not-found]
        return Path(folder_paths.get_input_directory()).resolve()
    except Exception:
        pass
    for parent in ROOT.parents:
        if parent.name == "custom_nodes":
            return (parent.parent / "input").resolve()
    return None


def _detect_workflow_directory() -> Path | None:
    """Find ComfyUI/user/default/workflows/."""
    try:
        import folder_paths  # type: ignore[import-not-found]
        return Path(folder_paths.get_user_directory()).resolve() / "default" / "workflows"
    except Exception:
        pass
    for parent in ROOT.parents:
        if parent.name == "custom_nodes":
            return (parent.parent / "user" / "default" / "workflows").resolve()
    return None


def _detect_models_root() -> Path | None:
    try:
        import folder_paths  # type: ignore[import-not-found]
        return Path(folder_paths.models_dir).resolve()
    except Exception:
        pass
    for parent in ROOT.parents:
        if parent.name == "custom_nodes":
            return (parent.parent / "models").resolve()
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Vendor submodule check
# ---------------------------------------------------------------------------

def ensure_vendor_submodule() -> None:
    """Verify vendor/UniVidX is checked out at the pinned commit."""
    required_paths = (
        "src/pipelines/univid_intrinsic.py",
        "src/pipelines/univid_alpha.py",
        "scripts/registry.py",
        "configs/wan2_1_14b_t2v_dit_config.json",
    )
    missing = [p for p in required_paths if not (VENDOR_UNIVIDX / p).exists()]
    if missing:
        msg = ", ".join(missing)
        raise RuntimeError(
            f"Vendored UniVidX is missing required files: {msg}. "
            "Did you clone with --recurse-submodules? Run "
            "`git submodule update --init --recursive` from the repo root."
        )

    # Verify the pinned commit. Best effort — if not a git checkout (e.g. tarball),
    # warn but don't fail.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=VENDOR_UNIVIDX,
            check=True, capture_output=True, text=True,
        )
        head = result.stdout.strip()
        if head != PINNED_UNIVIDX_COMMIT:
            print(
                f"WARNING: vendor/UniVidX is at {head[:8]}, expected {PINNED_UNIVIDX_COMMIT[:8]}. "
                "Behavior may diverge from the tested baseline."
            )
        else:
            print(f"Using vendored UniVidX at pinned commit {PINNED_UNIVIDX_COMMIT[:8]}.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"Vendored UniVidX present at {VENDOR_UNIVIDX} (could not verify git commit; "
            "this is fine if installed from a tarball or registry archive)."
        )


# ---------------------------------------------------------------------------
# Demo workflows
# ---------------------------------------------------------------------------

def ensure_demo_workflows() -> None:
    """Copy the bundled example workflow JSONs into the user's workflow dir."""
    workflow_dir = _detect_workflow_directory()
    if workflow_dir is None:
        print("Could not locate ComfyUI workflow directory; skipping demo workflow sync.")
        return

    workflow_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in DEMO_WORKFLOW_NAMES:
        source = EXAMPLES / name
        if not source.exists():
            print(f"Demo workflow missing in package: {source}, skipping.")
            continue
        destination = workflow_dir / name
        if destination.exists() and _file_sha256(destination) == _file_sha256(source):
            continue
        shutil.copy2(source, destination)
        copied += 1
        print(f"Copied demo workflow {name} -> {destination}")
    if copied == 0:
        print("Demo workflows already up to date.")


# ---------------------------------------------------------------------------
# Model presence hint
# ---------------------------------------------------------------------------

def hint_about_models() -> None:
    """Print a friendly status about the (large) model files the user must download."""
    models_root = _detect_models_root()
    if models_root is None:
        return
    wan = models_root / "wan21_t2v_14b"
    univ = models_root / "unividx"
    wan_present = wan.exists() and any(wan.glob("diffusion_pytorch_model-*.safetensors"))
    univ_present = (univ / "univid_intrinsic.safetensors").exists() and \
                   (univ / "univid_alpha.safetensors").exists()
    if wan_present and univ_present:
        print("All required models found.")
        return
    print()
    print("=" * 72)
    print("Model files not yet present. Download with the huggingface CLI:")
    print()
    if not wan_present:
        print(f"  hf download Wan-AI/Wan2.1-T2V-14B --local-dir {wan}")
    if not univ_present:
        print(f"  hf download houyuanchen/UniVidX  --local-dir {univ}")
    print()
    print("Wan2.1-T2V-14B is ~69 GB; UniVidX checkpoints are ~1.6 GB total.")
    print("=" * 72)


if __name__ == "__main__":
    ensure_vendor_submodule()
    ensure_demo_workflows()
    hint_about_models()
