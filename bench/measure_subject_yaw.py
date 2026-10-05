#!/usr/bin/env python3
"""Which way a person faces, frame by frame, in a render and in the source it was made from: a number for "did he turn".

The masked lane's question on a moving shot is whether the replaced subject
moves as the original did. On 2026-10-05 it was answered by reading five
frames of a contact sheet per render
(`bench/results/2026-10-05_masked_v2v_motion_arms.md`). This measures it.

**What it buys** (card `build-turn-metric`, the owner's ask): a number per
clip in place of a reading by eye, so that the per-step half of the same
card can say at which sigma the facing is decided, and a knot list for a
distill can be placed there. It applies to a shot where the subject turns
and the source's mask is kept. It is accepted when it puts the clips already
judged by eye in the same order, and the record says where it does not.

**The measurement.** Core's SAM 3D Body predictor runs on the subject's box
in each sampled frame and returns 3D keypoints in the camera's frame. The
yaw is the direction of the line from the right shoulder to the left
shoulder in the horizontal plane: 0 degrees facing the camera, 90 side-on,
180 with the back to it. The hips give a second reading of the same thing,
recorded beside it. Nothing about the model's own rotation parameters is
used: the root rotation it reports is the rig's, not the body's
(`pred_global_rots[0]` is the identity on every frame probed), and its
`global_rot` Euler angles wrap where the shoulder line does not.

**The box** is the source's kept mask (`mask_store.py`, the file under the
output folder's `masks/`), fitted to the render's canvas by the same centre
crop the song node gives the frames (`comfy.utils.common_upscale`,
"center"). A masked render regenerates the subject inside that mask, so the
source's box finds the new subject too.

**Per clip, against the source over the same frames:** the yaw at the end
of the shot (the circular mean of its last `END_SAMPLES` samples), its
difference from the source's, the mean difference over the shot, the first
sampled frame from which the two stay apart by more than the tolerance, the
largest turn (how far the yaw gets from where the shot began, which tells a
part-turn that comes back from no turn at all), and a verdict: `holds` when
the end difference is within the tolerance, `fails` otherwise. With `--eye` the clips' by-eye verdicts are read from a JSON
(`bench/turn_metric_eye_verdicts.json`) and the output says whether the end
differences put them in the same order (every "yes" below every "partial"
below every "no").

Not measured: anything but the facing. A clip can hold the turn and lose
the look, the lip sync or the hands; those stay with the eye.

In its own process, not through a server. Masked it runs on the CPU, which
is how the 2026-10-05 calibration ran. On the card ask whoever holds it
first; `bench/_lib::server_memory_mode` is called for that case.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/measure_subject_yaw.py \\
        --source <video> --start 112 --rate 24 --mask <kept mask .npz> \\
        --shot 237 279 --canvas 1344x768 --clips-dir <dir> \\
        --eye bench/turn_metric_eye_verdicts.json --out <record.json>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import REPO, bootstrap, needs, server_memory_mode  # noqa: E402

# inherited: Meta's MHR70 keypoint order (sam_3d_body/metadata/mhr70.py),
# which core's predictor returns unchanged.
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 9, 10
#: The body joints kept per frame for the motion metric
#: (`bench/measure_subject_motion.py`), by name, from the same table. Fingers
#: and toes are left out: at a clip's size they are the model's guess.
BODY_JOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14, "right_wrist": 41, "left_wrist": 62,
}

# reasoned: half way from facing the source's way to side-on. A render that
# ends within it faces where the source faces to the eye; set before the
# calibration was run, not fitted to it.
TOLERANCE_DEGREES = 45.0
# reasoned: the last few samples of the shot, so one bad frame does not
# decide the end.
END_SAMPLES = 4
# reasoned: every second frame draws a half-second turn with six samples.
EVERY = 2

RELEASES = {"vith": "sam_3d_body_vith.safetensors", "dinov3": "sam_3d_body_dinov3.safetensors"}
ORDER = {"yes": 0, "partial": 1, "no": 2}


# ----------------------------------------------------------------- the angles

def wrap(degrees: float) -> float:
    """An angle in (-180, 180]."""
    d = (float(degrees) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def apart(a: float, b: float) -> float:
    """How far apart two angles are, 0 to 180."""
    return abs(wrap(a - b))


def circular_mean(angles) -> float | None:
    values = [a for a in angles if a is not None]
    if not values:
        return None
    s = sum(math.sin(math.radians(a)) for a in values)
    c = sum(math.cos(math.radians(a)) for a in values)
    return wrap(math.degrees(math.atan2(s, c)))


def yaw_of(keypoints_3d, left: int, right: int) -> float:
    """The facing angle from a left and right joint: 0 toward the camera, 180 away.

    `keypoints_3d` is [K, 3] in the camera's frame, x to the image's right
    and z away from the camera. Facing the camera, a person's left side is
    on the image's right, so the left-minus-right vector points along +x.
    """
    k = np.asarray(keypoints_3d, dtype=np.float64)
    dx, dz = k[left, 0] - k[right, 0], k[left, 2] - k[right, 2]
    return wrap(math.degrees(math.atan2(dz, dx)))


def largest_turn(curve: list[float | None]) -> float | None:
    """How far the yaw gets from its first reading, 0 to 180: 180 for a full turn away, a few degrees for none."""
    values = [v for v in curve if v is not None]
    if not values:
        return None
    return round(max(apart(v, values[0]) for v in values), 1)


def compare(source: list[float | None], clip: list[float | None], frames: list[int],
            tolerance: float = TOLERANCE_DEGREES) -> dict:
    """How a clip's yaw curve stands against the source's over the same frames."""
    pairs = [(f, s, c) for f, s, c in zip(frames, source, clip) if s is not None and c is not None]
    if not pairs:
        return {"comparable": False, "verdict": "not measured"}
    end_source = circular_mean(source[-END_SAMPLES:])
    end_clip = circular_mean(clip[-END_SAMPLES:])
    diffs = [apart(s, c) for _f, s, c in pairs]
    diverges = None
    for i in range(len(pairs)):
        if all(d > tolerance for d in diffs[i:]):
            diverges = pairs[i][0]
            break
    end_difference = None if end_source is None or end_clip is None else apart(end_source, end_clip)
    return {
        "comparable": True,
        "samples": len(pairs),
        "largest_turn": largest_turn(clip),
        "largest_turn_source": largest_turn(source),
        "end_yaw_source": None if end_source is None else round(end_source, 1),
        "end_yaw": None if end_clip is None else round(end_clip, 1),
        "end_difference": None if end_difference is None else round(end_difference, 1),
        "mean_difference": round(sum(diffs) / len(diffs), 1),
        "diverges_at_frame": diverges,
        "tolerance": tolerance,
        "verdict": "not measured" if end_difference is None else ("holds" if end_difference <= tolerance else "fails"),
    }


def ranks_as_the_eye(measured: dict[str, float], eye: dict[str, str]) -> dict:
    """Whether end differences order the clips as the eye did: every yes below every partial below every no."""
    rows = [(label, eye[label], measured[label]) for label in eye if measured.get(label) is not None]
    wrong = []
    for a_label, a_eye, a_value in rows:
        for b_label, b_eye, b_value in rows:
            if ORDER[a_eye] < ORDER[b_eye] and not a_value < b_value:
                wrong.append({"turned_more_by_eye": a_label, "end_difference": a_value,
                              "turned_less_by_eye": b_label, "its_end_difference": b_value})
    groups = {}
    for _label, verdict, value in rows:
        groups.setdefault(verdict, []).append(value)
    return {
        "clips_compared": len(rows),
        "same_order": not wrong,
        "disagreements": wrong,
        "end_difference_by_eye_verdict": {v: {"lowest": min(vals), "highest": max(vals), "clips": len(vals)}
                                          for v, vals in sorted(groups.items(), key=lambda kv: ORDER[kv[0]])},
        "not_judged_by_eye": sorted(set(measured) - set(eye)),
    }


# ------------------------------------------------------------------ the media

def decode(path: Path, first_second: float, rate: float, count: int, size: tuple[int, int],
           crop_to: tuple[int, int] | None = None) -> torch.Tensor:
    """`count` frames from `first_second` at `rate`, [count, H, W, 3] in 0..1.

    `size` is (width, height) to scale to. With `crop_to` the frames are then
    centre-cropped to that aspect and scaled to it, as the song node fits a
    source to its canvas.
    """
    w, h = size
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{first_second:.5f}", "-i", str(path), "-vf", f"fps={rate},scale={w}:{h}",
         "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"ffmpeg could not read {path.name}: {done.stderr.decode(errors='replace')[-300:]}")
    got = len(done.stdout) // (w * h * 3)
    frames = torch.from_numpy(np.frombuffer(done.stdout[:got * w * h * 3], np.uint8).reshape(got, h, w, 3).copy())
    frames = frames.float().div(255)
    if crop_to is not None and crop_to != (w, h):
        import comfy.utils
        frames = comfy.utils.common_upscale(frames.movedim(-1, 1), crop_to[0], crop_to[1], "bilinear", "center").movedim(1, -1)
    return frames


def boxes(mask: torch.Tensor) -> list[dict | None]:
    """One {x, y, width, height} per frame of a [N, H, W] mask, None where it is empty."""
    out = []
    for m in mask:
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if xs.numel() == 0:
            out.append(None)
            continue
        out.append({"x": float(xs.min()), "y": float(ys.min()),
                    "width": float(xs.max() - xs.min() + 1), "height": float(ys.max() - ys.min() + 1)})
    return out


def joints_in_box(keypoints_2d, box: dict) -> dict[str, list[float]]:
    """Each body joint's image position from the subject's box's top left, in units of the box's HEIGHT.

    One unit for both axes, so a distance between two readings is a fraction
    of the subject's height on screen whichever way it points.
    """
    k = np.asarray(keypoints_2d, dtype=np.float64)
    return {name: [round(float((k[i, 0] - box["x"]) / box["height"]), 4),
                   round(float((k[i, 1] - box["y"]) / box["height"]), 4)] for name, i in BODY_JOINTS.items()}


def joints_in_body(keypoints_3d) -> dict[str, list[float]]:
    """Each body joint in 3D, from the middle of the hips, in units of the hips-to-shoulders distance.

    Centred and scaled on the body itself, so a subject who is larger, nearer
    or of another build reads the same when the pose is the same. The axes
    are the camera's, so the facing is part of it.
    """
    k = np.asarray(keypoints_3d, dtype=np.float64)
    hips = (k[LEFT_HIP] + k[RIGHT_HIP]) / 2
    torso = float(np.linalg.norm((k[LEFT_SHOULDER] + k[RIGHT_SHOULDER]) / 2 - hips)) or 1.0
    return {name: [round(float(v), 4) for v in (k[i] - hips) / torso] for name, i in BODY_JOINTS.items()}


def yaw_curve(predict, frames: torch.Tensor, frame_boxes: list[dict | None]) -> list[dict]:
    """One reading per frame: shoulder yaw, hip yaw and the body joints, or None where nobody was found."""
    empty = {"yaw": None, "hip_yaw": None, "in_box": None, "in_body": None}
    rows = []
    for image, box in zip(frames, frame_boxes):
        person = predict(image[None], box) if box is not None else None
        if person is None:
            rows.append(dict(empty))
            continue
        k = person["pred_keypoints_3d"]
        rows.append({"yaw": round(yaw_of(k, LEFT_SHOULDER, RIGHT_SHOULDER), 1),
                     "hip_yaw": round(yaw_of(k, LEFT_HIP, RIGHT_HIP), 1),
                     "in_box": joints_in_box(person["pred_keypoints_2d"], box),
                     "in_body": joints_in_body(k)})
    return rows


def reanalyse(args) -> int:
    """The comparisons again from an earlier output's curves, at this run's tolerance and eye verdicts."""
    record = json.loads(args.reanalyse.read_text())
    eye = {}
    if args.eye:
        eye = {label: row["turn"] for label, row in json.loads(args.eye.read_text())["clips"].items()}
    source_yaw = [r["yaw"] for r in record["source"]["curve"]]
    record["tolerance_degrees"] = args.tolerance
    record["source"]["largest_turn"] = largest_turn(source_yaw)
    for label, old in record["clips"].items():
        result = compare(source_yaw, [r["yaw"] for r in old["curve"]], record["frames"], args.tolerance)
        result["curve"] = old["curve"]
        if label in eye:
            result["eye"] = eye[label]
        record["clips"][label] = result
        print(f"{label}: end difference {result.get('end_difference')}, largest turn {result.get('largest_turn')}, "
              f"{result['verdict']}" + (f" (eye: {eye[label]})" if label in eye else ""))
    if eye:
        record["against_the_eye"] = ranks_as_the_eye(
            {label: r.get("end_difference") for label, r in record["clips"].items()}, eye)
    record["reanalysed_from"] = args.reanalyse.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--reanalyse", type=Path, metavar="JSON",
                    help="recompute the comparisons from an earlier output's curves; no model, no media")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="second of the source the window starts at")
    ap.add_argument("--rate", type=float, default=24.0, help="frames per second the window was loaded at")
    ap.add_argument("--mask", type=Path, help="the window's kept mask (.npz from the output folder's masks/)")
    ap.add_argument("--shot", type=int, nargs=2, metavar=("FIRST", "LAST"), help="frames of the window to measure, inclusive")
    ap.add_argument("--canvas", default="1344x768", help="the renders' width x height")
    ap.add_argument("--clips-dir", type=Path, help="where the renders are")
    ap.add_argument("--clip", action="append", default=[], metavar="LABEL=PATH", help="one render; repeatable")
    ap.add_argument("--eye", type=Path, help="by-eye verdicts JSON; its labels are also looked up in --clips-dir")
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_DEGREES)
    ap.add_argument("--release", choices=sorted(RELEASES), default="vith", help="which SAM 3D Body release reads the pose")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.reanalyse:
        return reanalyse(args)
    if not (args.source and args.mask and args.shot):
        ap.error("--source, --mask and --shot are required unless --reanalyse is given")
    needs(f"the source {args.source.name}", args.source.is_file())
    needs(f"the kept mask {args.mask.name}", args.mask.is_file())

    bootstrap()
    sys.path.append(str(REPO))
    dynamic_vram = server_memory_mode()
    import comfy.model_management as mm
    import comfy.utils
    import comfy_extras.nodes_sam3d_body as nodes
    import folder_paths
    import sam3d_body_vith

    needs(f"models/detection/{RELEASES[args.release]}", RELEASES[args.release] in folder_paths.get_filename_list("detection"))
    canvas = tuple(int(v) for v in args.canvas.lower().split("x"))
    first, last = args.shot
    sampled = list(range(first, last + 1, max(1, args.every)))

    eye, pattern = {}, "{label}*.mp4"
    if args.eye:
        spec = json.loads(args.eye.read_text())
        eye = {label: row["turn"] for label, row in spec["clips"].items()}
        pattern = spec.get("clip_pattern", pattern)
    clips: dict[str, Path] = {}
    if args.clips_dir:
        # the clips the eye judged, found by their label in the folder
        for label in sorted(eye):
            hits = sorted(args.clips_dir.glob(pattern.format(label=label)))
            if hits:
                clips[label] = hits[-1]
    for item in args.clip:
        label, _, path = item.partition("=")
        clips[label] = Path(path)
    missing = sorted(set(eye) - set(clips))

    with np.load(args.mask) as z:
        mask = torch.from_numpy(z["mask"])
    src_h, src_w = int(mask.shape[1]), int(mask.shape[2])
    source_boxes_all = boxes(mask)
    fitted = comfy.utils.common_upscale(mask[:, None], canvas[0], canvas[1], "bilinear", "center")[:, 0]
    render_boxes_all = boxes(fitted)

    path = folder_paths.get_full_path_or_raise("detection", RELEASES[args.release])
    patcher = sam3d_body_vith.load_model(path) if args.release == "vith" else \
        nodes.SAM3DBody_Loader.execute(RELEASES[args.release]).args[0]

    def predict(image, box):
        pose = nodes.SAM3DBody_Predict.execute(patcher, image, bboxes=[box], run_hand_refinement=False).args[0]
        people = pose["frames"][0]
        return people[0] if people else None

    began = time.monotonic()
    record = {
        "script": "bench/measure_subject_yaw.py", "release": args.release, "device": str(mm.get_torch_device()),
        "dynamic_vram": dynamic_vram, "torch": torch.__version__,
        "window": {"start_second": args.start, "rate": args.rate, "mask_frames": int(mask.shape[0]),
                   "source_size": [src_w, src_h], "canvas": list(canvas)},
        "shot": [first, last], "frames": sampled, "tolerance_degrees": args.tolerance,
        "yaw": "0 facing the camera, 90 side-on, 180 back to it; from the shoulder line, hips beside it",
        "source": {}, "clips": {}, "clips_not_found": missing,
    }
    with torch.inference_mode():
        span = decode(args.source, args.start + first / args.rate, args.rate, last - first + 1, (src_w, src_h))
        rows = yaw_curve(predict, span[::max(1, args.every)], [source_boxes_all[f] for f in sampled])
        record["source"] = {"curve": rows}
        source_yaw = [r["yaw"] for r in rows]
        print(f"source: {[r['yaw'] for r in rows]}", flush=True)
        for label, clip in clips.items():
            span = decode(clip, first / args.rate, args.rate, last - first + 1, canvas)
            rows = yaw_curve(predict, span[::max(1, args.every)], [render_boxes_all[f] for f in sampled])
            result = compare(source_yaw, [r["yaw"] for r in rows], sampled, args.tolerance)
            result["curve"] = rows
            if label in eye:
                result["eye"] = eye[label]
            record["clips"][label] = result
            print(f"{label}: end {result.get('end_yaw')} vs source {result.get('end_yaw_source')}, "
                  f"difference {result.get('end_difference')}, {result['verdict']}"
                  + (f" (eye: {eye[label]})" if label in eye else ""), flush=True)
    record["seconds"] = round(time.monotonic() - began, 1)
    if eye:
        measured = {label: r.get("end_difference") for label, r in record["clips"].items()}
        record["against_the_eye"] = ranks_as_the_eye(measured, eye)
        print(json.dumps(record["against_the_eye"], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
