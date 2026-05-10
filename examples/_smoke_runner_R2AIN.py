"""
R2AIN smoke runner: feeds 21 copies of a single RGB frame as conditioning
input, generates Albedo/Irradiance/Normal videos. The decoder's `rgb`
output is expected to be a black placeholder (since RGB was the input,
not a regenerated target).

This validates:
- The IMAGE batch input plumbing in UniVidXSampler.
- tensor_io.image_batch_to_video_tensor's [B,H,W,C in 0..1] -> [1,3,T,H,W in -1..1] conversion.
- Mode-specific input validation (R2AIN requires only rgb).
- Decoder splay logic when result_dict is missing the rgb key.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent
WORKFLOW_PATH = HERE / "R2AIN_basic_api.json"


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
    print(f"Loaded R2AIN workflow ({len(workflow)} nodes)", flush=True)

    info = _get("/object_info")
    expected = {"UniVidXLoader", "UniVidXTaskMode", "UniVidXSampler",
                "UniVidXDecodeIntrinsic", "LoadImage", "RepeatImageBatch"}
    missing = expected - set(info.keys())
    if missing:
        print(f"FAIL: Missing nodes: {sorted(missing)}", file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"All required nodes confirmed.", flush=True)

    resp = _post("/prompt", {"prompt": workflow, "client_id": "unividx-R2AIN-smoke"})
    if "prompt_id" not in resp or resp.get("node_errors"):
        print(f"FAIL queueing: {json.dumps(resp, indent=2)[:1000]}", file=sys.stderr, flush=True)
        sys.exit(2)
    prompt_id = resp["prompt_id"]
    print(f"Queued {prompt_id}. Polling every 5s. Timeout 1800s.", flush=True)

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
            if status != last_status:
                print(f"  [{elapsed:5.0f}s] status={status}", flush=True)
                last_status = status
            if status in ("success", "error"):
                if status == "error":
                    print(f"FAIL: status=error.", file=sys.stderr, flush=True)
                    for m in entry.get("status", {}).get("messages", []):
                        print(f"    {m}", file=sys.stderr, flush=True)
                    sys.exit(3)
                print(f"\nSUCCESS in {elapsed:.0f}s.", flush=True)
                outputs = entry.get("outputs", {})
                files_by_node = {}
                for node_id, out in outputs.items():
                    if "images" in out:
                        files_by_node[node_id] = [img['filename'] for img in out["images"]]
                # Map node id back to modality based on the workflow we sent
                for node_id, files in sorted(files_by_node.items()):
                    print(f"  node {node_id}: {len(files)} files (e.g. {files[0]})", flush=True)
                return 0
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
        if elapsed > 1800:
            print(f"FAIL: timeout {elapsed:.0f}s", file=sys.stderr, flush=True)
            sys.exit(4)


if __name__ == "__main__":
    main()
