#!/usr/bin/env python3
"""SAM 3D Body's two releases on one frame: core's predict and render through the DINOv3 file and the ViT-H file.

The DINOv3 release loads with core's `SAM3DBody_Loader`; the ViT-H release
loads with this pack's `sam3d_body_vith.load_model`. Everything after the
loader is core's own node code, called in process: `SAM3DBody_Predict` on one
frame with one person box, then `SAM3DBody_Render` as a mesh over the frame.
The output is a contact tile (the frame with the box, then one render per
release) and a JSON of how the two predictions differ: 2D keypoints in
pixels, the camera translation, and the mesh vertices.

It answers "does the ViT-H port produce a body where the DINOv3 model does",
on the frame it is given. It is not an accuracy comparison: there is no
ground truth here, and the two releases are different models, so they are
expected to differ.

**It runs in this process, not through the server**, so it can use a node
the running server has not loaded. On the card that means a second process
holding VRAM beside the server: ask whoever holds the card first
(`AGENTS.md`, "The server process is the resource"). Masked
(`CUDA_VISIBLE_DEVICES=`), it runs on the CPU in float32, which is how it is
rehearsed.

The frame is the owner's media: the tile goes where `--out-dir` says, under
`internal/`, and is never committed. The JSON carries no path.

    <comfy venv python> bench/compare_sam3d_body_releases.py \\
        --frame <frame.png> --box 660,0,520,1012 --out-dir internal/<...>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import REPO, bootstrap, needs  # noqa: E402

DINOV3_FILE = "sam_3d_body_dinov3.safetensors"
VITH_FILE = "sam_3d_body_vith.safetensors"


def server_memory_mode() -> bool:
    """Set up ComfyUI's dynamic VRAM layer the way `main.py` does at start.

    Call it after `bootstrap()` and before `comfy.model_management` is
    imported. On the card it is not optional: core's SAM 3D Body loader picks
    half-precision weights with no manual cast, and it is this layer's ops
    that cast a weight to its input at use. Without it core's own DINOv3 path
    stops in `run_keypoint_prompt` on a float input meeting a half weight
    (seen 2026-10-05, on the first card run of this script, before the ViT-H
    model was reached). Transcribed from `main.py`, the two blocks around
    `comfy_aimdo.control.init` and `init_devices`; returns whether it took.
    """
    from comfy.cli_args import args, enables_dynamic_vram
    if not enables_dynamic_vram():
        return False
    import comfy_aimdo.control
    headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
    try:
        comfy_aimdo.control.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
    except TypeError:
        try:
            comfy_aimdo.control.init(simple_vram_headroom=headroom)
        except TypeError:
            comfy_aimdo.control.init()
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    try:
        took = comfy_aimdo.control.init_devices(
            (d.index, int(args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
    except TypeError:
        took = comfy_aimdo.control.init_devices(d.index for d in comfy.model_management.get_all_torch_devices())
    if took:
        comfy_aimdo.control.set_log_warning()
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
    return bool(took)


def _first(node_output):
    """The first value of an `io.NodeOutput`."""
    return node_output.args[0] if hasattr(node_output, "args") else node_output[0]


def _person(pose):
    frames = pose["frames"]
    if not frames or not frames[0]:
        return None
    return frames[0][0]


def _as_np(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def run_release(label, patcher, image, box, hands, nodes, mm):
    inner = patcher.model
    t0 = time.monotonic()
    pose = _first(nodes.SAM3DBody_Predict.execute(
        patcher, image, bboxes=[box], run_hand_refinement=hands))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.monotonic()
    render = _first(nodes.SAM3DBody_Render.execute(
        pose, background=image, render_style={"render_style": "mesh", "opacity": 0.8}))
    silhouette = _first(nodes.SAM3DBody_Render.execute(pose, render_style={"render_style": "silhouette"}))
    t2 = time.monotonic()
    info = {
        "model_class": type(inner).__name__,
        "backbone_class": type(inner.backbone).__name__,
        "backbone_dtype": str(inner.backbone_dtype),
        "device": str(mm.get_torch_device()),
        "predict_seconds": round(t1 - t0, 2),
        "render_seconds": round(t2 - t1, 2),
        "person_found": _person(pose) is not None,
        "silhouette_fraction_of_frame": round(float((silhouette[0, ..., 0] > 0.5).float().mean()), 4),
    }
    print(f"{label}: {info}")
    return pose, render, silhouette, info


def compare(pose_a, pose_b, image_hw):
    """How two predictions of one person differ, field by field where both are arrays."""
    a, b = _person(pose_a), _person(pose_b)
    if a is None or b is None:
        return {"comparable": False}
    out = {"comparable": True, "fields": {}}
    for key in sorted(set(a) & set(b)):
        try:
            x, y = _as_np(a[key]), _as_np(b[key])
        except Exception:
            continue
        if x.dtype.kind not in "fiu" or x.shape != y.shape or x.size == 0:
            continue
        out["fields"][key] = {"shape": list(x.shape),
                              "max_abs_difference": float(np.abs(x.astype(np.float64) - y).max())}
    for key in ("pred_keypoints_2d", "keypoints_2d"):
        if key in a and key in b:
            ka, kb = _as_np(a[key])[..., :2], _as_np(b[key])[..., :2]
            dist = np.linalg.norm(ka - kb, axis=-1)
            h, w = image_hw
            inside = (ka[..., 0] >= 0) & (ka[..., 0] < w) & (ka[..., 1] >= 0) & (ka[..., 1] < h)
            out["keypoints_2d_px"] = {
                "field": key, "count": int(dist.size),
                "mean": float(dist.mean()), "median": float(np.median(dist)), "max": float(dist.max()),
                "inside_frame": int(inside.sum()),
                "mean_inside_frame": float(dist[inside].mean()) if inside.any() else None,
            }
            break
    for key in ("pred_vertices", "vertices"):
        if key in a and key in b:
            va, vb = _as_np(a[key]), _as_np(b[key])
            # the mesh in its own frame, then with each one's mean removed
            raw = np.linalg.norm(va - vb, axis=-1)
            centred = np.linalg.norm((va - va.mean(0)) - (vb - vb.mean(0)), axis=-1)
            out["vertices_m"] = {"field": key, "count": int(raw.size), "mean": float(raw.mean()),
                                 "max": float(raw.max()), "mean_after_centring": float(centred.mean())}
            break
    for key in ("pred_cam_t", "cam_t"):
        if key in a and key in b:
            ca, cb = _as_np(a[key]).reshape(-1), _as_np(b[key]).reshape(-1)
            out["camera_translation_m"] = {"field": key, "dinov3": ca.tolist(), "vith": cb.tolist()}
            break
    return out


def tile(frame, box, renders, labels, path, width=640):
    from PIL import Image, ImageDraw
    panels = []
    base = Image.fromarray((frame[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    boxed = base.copy()
    ImageDraw.Draw(boxed).rectangle(
        [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]], outline=(255, 255, 0), width=6)
    panels.append(("frame and box", boxed))
    for label, render in zip(labels, renders):
        arr = (render[0].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        panels.append((label, Image.fromarray(arr)))
    scale = width / base.width
    size = (width, int(round(base.height * scale)))
    sheet = Image.new("RGB", (size[0] * len(panels), size[1] + 28), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(panels):
        sheet.paste(im.resize(size), (i * size[0], 28))
        draw.text((i * size[0] + 8, 8), label, fill=(255, 255, 255))
    sheet.save(path, quality=90)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--frame", required=True, type=Path, help="one image file")
    ap.add_argument("--box", required=True, help="the person's box in pixels: x,y,width,height")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tag", default="", help="suffix for the output names, e.g. cpu or card")
    ap.add_argument("--no-hands", action="store_true", help="skip core's hand refinement")
    args = ap.parse_args()
    needs(f"the frame {args.frame.name}", args.frame.is_file())

    bootstrap()                      # CPU path when the card is masked, the card otherwise
    sys.path.append(str(REPO))
    dynamic_vram = server_memory_mode()
    import comfy.model_management as mm
    import comfy_extras.nodes_sam3d_body as nodes
    import folder_paths
    import sam3d_body_vith as vith

    listed = folder_paths.get_filename_list("detection")
    for name in (DINOV3_FILE, VITH_FILE):
        needs(f"models/detection/{name}", name in listed)

    from PIL import Image
    frame = torch.from_numpy(np.asarray(Image.open(args.frame).convert("RGB"))).float().div(255)[None]
    x, y, w, h = (float(v) for v in args.box.split(","))
    box = {"x": x, "y": y, "width": w, "height": h}
    hands = not args.no_hands
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    results = {}
    with torch.inference_mode():
        for label, load in (
                ("dinov3", lambda: _first(nodes.SAM3DBody_Loader.execute(DINOV3_FILE))),
                ("vith", lambda: vith.load_model(folder_paths.get_full_path_or_raise("detection", VITH_FILE)))):
            patcher = load()
            results[label] = run_release(label, patcher, frame, box, hands, nodes, mm)
            del patcher
            mm.unload_all_models()
            mm.soft_empty_cache()

    record = {
        "script": "bench/compare_sam3d_body_releases.py",
        "frame_size": [int(frame.shape[1]), int(frame.shape[2])],
        "box_xywh": [x, y, w, h],
        "hand_refinement": hands,
        "torch": torch.__version__,
        "dynamic_vram": dynamic_vram,
        "releases": {label: r[3] for label, r in results.items()},
        "difference": compare(results["dinov3"][0], results["vith"][0], frame.shape[1:3]),
    }
    sil_a, sil_b = (results[k][2][0, ..., 0] > 0.5 for k in ("dinov3", "vith"))
    union = float((sil_a | sil_b).float().sum())
    record["difference"]["silhouette_iou"] = round(float((sil_a & sil_b).float().sum()) / union, 4) if union else None
    box_mask = torch.zeros_like(sil_a)
    box_mask[int(y):int(y + h), int(x):int(x + w)] = True
    for label, sil in (("dinov3", sil_a), ("vith", sil_b)):
        total = float(sil.float().sum())
        record["releases"][label]["silhouette_fraction_inside_box"] = (
            round(float((sil & box_mask).float().sum()) / total, 4) if total else None)

    tile_path = args.out_dir / f"sam3d_body_releases{suffix}.jpg"
    tile(frame, box, [results["dinov3"][1], results["vith"][1]],
         ["DINOv3 (core's loader)", "ViT-H (this pack's loader)"], tile_path)
    json_path = args.out_dir / f"sam3d_body_releases{suffix}.json"
    json_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {tile_path.name} and {json_path.name} in the out dir")
    diff = record["difference"]
    print(json.dumps({k: diff.get(k) for k in ("keypoints_2d_px", "vertices_m", "silhouette_iou")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
