"""
Unified test-matrix runner.

For each test in TEST_SPECS:
  - Queue the workflow against ComfyUI
  - Poll /history until success or error
  - Run mode-specific assertions on the outputs
  - Record pass/fail

Prints a final markdown report.
"""
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import OrderedDict

API = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(r"C:\Users\dr5090\Documents\ComfyUI\output")


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Test specifications
# ---------------------------------------------------------------------------
# Each entry: (test_id, workflow_path, expected_outcome, assertions)
#   expected_outcome: "success" | "error"
#   assertions: callable(history_entry) -> (ok, msg) for success;
#               callable(error_messages) -> (ok, msg) for error
# ---------------------------------------------------------------------------

def assert_intrinsic_outputs(file_prefix, expected_targets, expected_placeholders):
    """Check generated frames for an intrinsic-mode test.

    expected_targets:     modality names that should have non-zero pixel content
    expected_placeholders: modality names that should be black placeholders
    """
    def check(history_entry):
        outputs = history_entry.get("outputs", {})
        # Map node id -> filenames
        files_by_node = {}
        for nid, out in outputs.items():
            if "images" in out:
                files_by_node[nid] = [img["filename"] for img in out["images"]]

        # The decoder output slots are (rgb, albedo, irradiance, normal) for intrinsic.
        # We rely on filename_prefix encoding to identify which modality each batch is.
        # Look up actual file content.
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return True, "skipped image checks (PIL/numpy not available)"

        results = []
        for mod in expected_targets + expected_placeholders:
            # Find a frame for this modality
            patt = f"{file_prefix}_{mod}_00001_.png"
            fp = OUTPUT_DIR / patt
            if not fp.exists():
                return False, f"missing output {patt}"
            arr = np.array(Image.open(fp))
            is_black = (arr.min() == 0 and arr.max() == 0)
            if mod in expected_targets and is_black:
                return False, f"target {mod} unexpectedly all-black"
            if mod in expected_placeholders and not is_black:
                return False, f"placeholder {mod} unexpectedly non-black (max={arr.max()})"
            results.append(f"{mod}={'black' if is_black else f'mean={arr.mean():.0f},std={arr.std():.0f}'}")
        return True, " | ".join(results)
    return check


def assert_alpha_outputs(file_prefix, expected_targets, expected_placeholders):
    """Same idea, but for alpha-family decoder which uses different modality names."""
    def check(history_entry):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return True, "skipped image checks"
        results = []
        for mod in expected_targets + expected_placeholders:
            patt = f"{file_prefix}_{mod}_00001_.png"
            fp = OUTPUT_DIR / patt
            if not fp.exists():
                return False, f"missing output {patt}"
            arr = np.array(Image.open(fp))
            is_black = (arr.min() == 0 and arr.max() == 0)
            if mod in expected_targets and is_black:
                return False, f"target {mod} unexpectedly all-black"
            if mod in expected_placeholders and not is_black:
                return False, f"placeholder {mod} unexpectedly non-black"
            results.append(f"{mod}={'black' if is_black else f'mean={arr.mean():.0f},std={arr.std():.0f}'}")
        return True, " | ".join(results)
    return check


def assert_error_contains(needle):
    """For error-expected tests: confirm the exception message matches."""
    def check(messages):
        haystack = json.dumps(messages)
        if needle.lower() in haystack.lower():
            return True, f"error message contains '{needle}'"
        return False, f"error message did not contain '{needle}'. raw: {haystack[:300]}"
    return check


TEST_SPECS = [
    {
        "id": "C_RA2IN",
        "workflow": "C_RA2IN.json",
        "expected": "success",
        "timeout": 600,
        "assertions": assert_intrinsic_outputs(
            "matrix_C_RA2IN",
            expected_targets=["irradiance", "normal"],
            expected_placeholders=["rgb", "albedo"],
        ),
    },
    {
        "id": "D_RAI2N",
        "workflow": "D_RAI2N.json",
        "expected": "success",
        "timeout": 600,
        "assertions": assert_intrinsic_outputs(
            "matrix_D_RAI2N",
            expected_targets=["normal"],
            expected_placeholders=["rgb", "albedo", "irradiance"],
        ),
    },
    {
        "id": "E_t2RPFB",
        "workflow": "E_t2RPFB.json",
        "expected": "success",
        "timeout": 900,  # longer: alpha cold load
        "assertions": assert_alpha_outputs(
            "matrix_E_t2RPFB",
            expected_targets=["composite_rgb", "alpha", "foreground", "background"],
            expected_placeholders=[],
        ),
    },
    {
        "id": "F_R2PFB",
        "workflow": "F_R2PFB.json",
        "expected": "success",
        "timeout": 600,
        "assertions": assert_alpha_outputs(
            "matrix_F_R2PFB",
            expected_targets=["alpha", "foreground", "background"],
            expected_placeholders=["composite_rgb"],
        ),
    },
    {
        "id": "G_error_family_mismatch",
        "workflow": "G_error_family_mismatch.json",
        "expected": "error",
        "timeout": 60,
        "assertions": assert_error_contains("family"),  # "Task mode ... is family=alpha, but loaded model is intrinsic"
    },
    {
        "id": "H_error_missing_input",
        "workflow": "H_error_missing_input.json",
        "expected": "error",
        "timeout": 60,
        "assertions": assert_error_contains("missing"),  # "Mode R2AIN requires inputs ..., missing: ..."
    },
]


def run_test(spec):
    print(f"\n{'='*60}")
    print(f"  Test: {spec['id']}")
    print(f"  Expected: {spec['expected']}")
    print(f"{'='*60}", flush=True)

    workflow_path = HERE / spec["workflow"]
    if not workflow_path.exists():
        return {"id": spec["id"], "outcome": "BUILD_FAIL",
                "msg": f"workflow JSON not found: {workflow_path}",
                "elapsed": 0}

    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    try:
        resp = _post("/prompt", {"prompt": workflow, "client_id": f"matrix-{spec['id']}"})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"id": spec["id"], "outcome": "QUEUE_FAIL",
                "msg": f"HTTP {e.code}: {body}", "elapsed": 0}

    if "prompt_id" not in resp:
        return {"id": spec["id"], "outcome": "QUEUE_FAIL",
                "msg": f"no prompt_id in response: {resp}", "elapsed": 0}
    if resp.get("node_errors"):
        return {"id": spec["id"], "outcome": "QUEUE_FAIL",
                "msg": f"node_errors at queue time: {json.dumps(resp['node_errors'])[:300]}",
                "elapsed": 0}

    prompt_id = resp["prompt_id"]
    print(f"  Queued {prompt_id}, polling (timeout {spec['timeout']}s)...", flush=True)

    t0 = time.time()
    last_status = None
    while True:
        time.sleep(5)
        elapsed = time.time() - t0
        try:
            history = _get(f"/history/{prompt_id}")
        except Exception as e:
            print(f"  [{elapsed:5.0f}s] history poll error: {e}", flush=True)
            continue
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {}).get("status_str", "")
            if status in ("success", "error"):
                return {"id": spec["id"], "outcome": status,
                        "history_entry": entry, "elapsed": elapsed,
                        "spec": spec}
        else:
            try:
                queue = _get("/queue")
                running = [q for q in queue.get("queue_running", []) if q[1] == prompt_id]
                pending = [q for q in queue.get("queue_pending", []) if q[1] == prompt_id]
                tag = "running" if running else ("pending" if pending else "?")
                if tag != last_status:
                    print(f"  [{elapsed:5.0f}s] {tag}", flush=True)
                    last_status = tag
            except Exception:
                pass
        if elapsed > spec["timeout"]:
            return {"id": spec["id"], "outcome": "TIMEOUT",
                    "msg": f"timeout after {spec['timeout']}s", "elapsed": elapsed}


def evaluate(result):
    """Run the spec's assertions on the test result. Return (verdict, detail)."""
    spec = result.get("spec", {})
    expected = spec.get("expected")
    actual = result["outcome"]

    if actual not in ("success", "error"):
        return "FAIL", f"runner outcome: {actual} ({result.get('msg', '')})"

    if actual != expected:
        # Got the wrong outcome (e.g. expected success, got error or vice versa)
        if actual == "error":
            messages = result["history_entry"].get("status", {}).get("messages", [])
            return "FAIL", f"expected {expected} but got error. Messages: {json.dumps(messages)[:300]}"
        return "FAIL", f"expected {expected} but got {actual}"

    # Outcome matches. Run the assertion.
    fn = spec["assertions"]
    if expected == "success":
        ok, msg = fn(result["history_entry"])
    else:
        messages = result["history_entry"].get("status", {}).get("messages", [])
        ok, msg = fn(messages)
    return ("PASS" if ok else "FAIL"), msg


def main():
    # Optional CLI filter: --filter test_id_substring
    filter_arg = None
    if "--filter" in sys.argv:
        filter_arg = sys.argv[sys.argv.index("--filter") + 1]

    specs = TEST_SPECS
    if filter_arg:
        specs = [s for s in TEST_SPECS if filter_arg in s["id"]]
        if not specs:
            print(f"No tests matched filter '{filter_arg}'. Available: {[s['id'] for s in TEST_SPECS]}",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Running test matrix against {API}", flush=True)
    print(f"Tests: {[s['id'] for s in specs]}", flush=True)

    results = []
    for spec in specs:
        result = run_test(spec)
        verdict, detail = evaluate(result)
        result["verdict"] = verdict
        result["detail"] = detail
        results.append(result)
        print(f"  -> {verdict}: {detail}", flush=True)

    # Final report (markdown)
    print("\n\n" + "="*70)
    print("  TEST MATRIX RESULTS")
    print("="*70)
    print()
    print("| Test | Outcome | Time | Verdict | Detail |")
    print("|---|---|---|---|---|")
    pass_count = 0
    for r in results:
        verdict = r["verdict"]
        if verdict == "PASS":
            pass_count += 1
        elapsed = r.get("elapsed", 0)
        detail = r.get("detail", "")[:120]
        print(f"| {r['id']} | {r['outcome']} | {elapsed:.0f}s | {verdict} | {detail} |")
    print()
    print(f"Total: {pass_count}/{len(results)} passed")
    print()
    if filter_arg:
        print(f"(filtered to: {filter_arg})")
        print()

    # Save raw results for post-hoc analysis
    save_path = HERE / "_run_results.json"
    save_path.write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("history_entry", "spec")} for r in results],
        indent=2,
    ), encoding="utf-8")
    print(f"Raw results: {save_path}")

    sys.exit(0 if pass_count == len(results) else 1)


if __name__ == "__main__":
    main()
