"""
Submit examples/t2RAIN_basic.json to ComfyUI's /prompt API and watch progress.

Run with the venv Python:
    python examples\_smoke_runner.py

ComfyUI must already be running on 127.0.0.1:8001 with the comfyui-unividx
custom node loaded.
"""
import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

API = "http://127.0.0.1:8001"
HERE = Path(__file__).resolve().parent
WORKFLOW_PATH = HERE / "t2RAIN_basic.json"


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


def main():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    print(f"Loaded workflow with {len(workflow)} nodes from {WORKFLOW_PATH}")

    # Verify our nodes exist on the server
    info = _get("/object_info")
    expected = {"UniVidXLoader", "UniVidXTaskMode", "UniVidXSampler",
                "UniVidXDecodeIntrinsic", "UniVidXDecodeAlpha"}
    missing = expected - set(info.keys())
    if missing:
        print(f"FAIL: ComfyUI has not loaded these nodes: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)
    print("All 5 UniVidX nodes confirmed on server.")

    # Queue the prompt
    resp = _post("/prompt", {"prompt": workflow, "client_id": "unividx-smoke"})
    if "prompt_id" not in resp:
        print(f"FAIL queueing: {resp}", file=sys.stderr)
        sys.exit(2)
    if resp.get("node_errors"):
        print(f"FAIL — server reported node_errors at queue time:", file=sys.stderr)
        print(json.dumps(resp["node_errors"], indent=2), file=sys.stderr)
        sys.exit(2)
    prompt_id = resp["prompt_id"]
    print(f"Queued prompt {prompt_id}. Watching progress (poll every 5s)...")

    # Poll history until done
    t0 = time.time()
    last_status = None
    while True:
        time.sleep(5)
        elapsed = time.time() - t0
        try:
            history = _get(f"/history/{prompt_id}")
        except Exception as e:
            print(f"  [{elapsed:6.0f}s] history poll error: {e}")
            continue
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {}).get("status_str", "")
            messages = entry.get("status", {}).get("messages", [])
            outputs = entry.get("outputs", {})
            if status != last_status:
                print(f"  [{elapsed:6.0f}s] status={status} outputs={list(outputs.keys())}")
                last_status = status
            if status in ("success", "error"):
                if status == "error":
                    print(f"FAIL: status=error. Messages:")
                    for m in messages:
                        print(f"    {m}")
                    sys.exit(3)
                print(f"\nSUCCESS in {elapsed:.0f}s.")
                print(f"Outputs by node:")
                for node_id, out in outputs.items():
                    if "images" in out:
                        for img in out["images"]:
                            print(f"  node {node_id}: {img['filename']} ({img.get('subfolder','')})")
                return 0
        else:
            # Still queued or executing — check /queue
            try:
                queue = _get("/queue")
                running = [q for q in queue.get("queue_running", []) if q[1] == prompt_id]
                pending = [q for q in queue.get("queue_pending", []) if q[1] == prompt_id]
                tag = "running" if running else ("pending" if pending else "?")
                if tag != last_status:
                    print(f"  [{elapsed:6.0f}s] {tag}")
                    last_status = tag
            except Exception:
                pass
        # Sanity timeout: 30 minutes for a smoke test (model load + 20 steps).
        if elapsed > 1800:
            print(f"FAIL: timeout after {elapsed:.0f}s", file=sys.stderr)
            sys.exit(4)


if __name__ == "__main__":
    main()
