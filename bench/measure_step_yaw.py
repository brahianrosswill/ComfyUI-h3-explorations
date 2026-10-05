#!/usr/bin/env python3
"""At which sampling step a render's facing is decided: the turn metric on each step's clean prediction.

`bench/measure_subject_yaw.py` says whether a finished render faces where its
source faces. This asks when, during sampling, that became true. On
2026-10-05 a video reference carried the source's turn at twelve steps and
lost it at eight, with or without a distill. If the pose is decided over a
few early steps, a knot list can put evaluations there; if the model ignores
the reference at those steps however many there are, no schedule helps.

**What it buys** (card `build-turn-metric`, second half; the owner's ask on
manual sigmas for a distill): the sigma from which the end-of-shot facing
stays put, per render, read off instead of guessed. It applies to a render
made with `MiniMaxH3StepX0Observer` on the model that feeds the sampler. It
is accepted when the last step's reading agrees with the finished video's
(the control below), and when the render that holds the turn and the one
that loses it give different curves over the steps.

**The measurement.** The observer saves every step's x0, the model's clean
prediction at that step, as a video latent with its sigma. For each step the
latent frames under the shot are decoded with the video VAE and the yaw is
read on the sampled frames exactly as on a finished clip (the same box from
the kept mask, the same shoulder line). Early predictions are a blur with no
body in them; the body model then finds nobody or reads noise, and the
record keeps both the readings and how many frames had one.

**Per step:** sigma, how many sampled frames gave a reading, the end-of-shot
yaw, its difference from the source's, the largest turn. **Per render:**
`locks_at_step`, the first step from which the end difference stays within
the tolerance through the last step (None when the last step is outside
it), and that step's sigma.

**The control.** The last step's curve is compared with the finished
video's own, measured from the mp4: they are the same frames up to the
decode of a slice and the encode of the file, so their mean difference
should be a few degrees. It is reported, and the run says so loudly when it
is not.

**The slice.** A latent frame holds four video frames, the first one frame
(`PATTERN`, as `bench/x0_step_frames.py` has it). Only the latent frames
under the shot are decoded, with one more in front; the VAE treats the
first latent it is given as a single frame, so decoded frame j of a slice
starting at latent L is video frame 4L + j for j of 1 and up, and j of 0 is
dropped.

Needs the card: the decode of one slice did not finish in ten minutes on
the CPU (2026-10-05). In its own process; ask whoever holds the card.

    <comfy venv python> bench/measure_step_yaw.py \\
        --steps "<output>/latents/<prefix>_x0_<stamp>_*_video.latent" \\
        --video <render.mp4> --mask <kept mask .npz> --shot 237 279 \\
        --source-curves <a measure_subject_yaw record with the source's curve> \\
        --out <record.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import measure_subject_yaw as Y  # noqa: E402
from _lib import REPO, bootstrap, needs, server_memory_mode  # noqa: E402

# inherited: H3's temporal packing, one video frame under the first latent
# frame and four under each later one (`bench/x0_step_frames.py::PATTERN`).
FRAMES_PER_LATENT = 4

STEP_FILE = re.compile(r"_(\d+)_video\.latent$")

# reasoned: the decode of a slice and the mp4's own encode are the only
# differences between the last step and the finished file, so more than this
# between them means the frames are not lined up.
CONTROL_DEGREES = 10.0


def latent_index(frame: int) -> int:
    """The latent frame that holds video frame `frame`."""
    return 0 if frame <= 0 else 1 + (int(frame) - 1) // FRAMES_PER_LATENT


def slice_plan(first: int, last: int) -> tuple[int, int, int]:
    """(first latent, one past the last latent, video frame of decoded index 0's successor base).

    The slice starts one latent before the shot's first, so every frame of
    the shot is decoded as one of four and none as the slice's lone first
    frame. Decoded index j >= 1 is video frame `base + j`.
    """
    start = max(latent_index(first) - 1, 0)
    stop = latent_index(last) + 1
    return start, stop, FRAMES_PER_LATENT * start


def decoded_index(frame: int, base: int) -> int:
    return int(frame) - base


def step_files(pattern: str) -> list[tuple[int, Path]]:
    """(step, path) for every observer file the pattern matches, in step order."""
    found = []
    for name in glob.glob(pattern):
        m = STEP_FILE.search(name)
        if m:
            found.append((int(m.group(1)), Path(name)))
    found.sort()
    steps = [s for s, _p in found]
    if steps != list(range(len(steps))):
        raise SystemExit(f"the step files are not one render's steps 0..n-1 in order: {steps}. "
                         "Give a pattern that names one render's stamp.")
    return found


def lock_step(end_differences: list[float | None], tolerance: float) -> int | None:
    """The first step from which every later step's end difference is within the tolerance."""
    if not end_differences or end_differences[-1] is None or end_differences[-1] > tolerance:
        return None
    lock = len(end_differences) - 1
    for i in range(len(end_differences) - 1, -1, -1):
        if end_differences[i] is None or end_differences[i] > tolerance:
            break
        lock = i
    return lock


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--steps", required=True, help="glob of one render's observer files, *_NN_video.latent")
    ap.add_argument("--video", required=True, type=Path, help="the finished render, for the control")
    ap.add_argument("--mask", required=True, type=Path, help="the window's kept mask (.npz)")
    ap.add_argument("--shot", required=True, type=int, nargs=2, metavar=("FIRST", "LAST"))
    ap.add_argument("--source-curves", required=True, type=Path,
                    help="a measure_subject_yaw record over the same shot and sampling: the source's curve is read from it")
    ap.add_argument("--rate", type=float, default=24.0)
    ap.add_argument("--tolerance", type=float, default=Y.TOLERANCE_DEGREES)
    ap.add_argument("--release", choices=sorted(Y.RELEASES), default="vith")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = step_files(args.steps)
    needs("observer files matching --steps", bool(files))
    needs(f"the render {args.video.name}", args.video.is_file())
    needs(f"the kept mask {args.mask.name}", args.mask.is_file())
    earlier = json.loads(args.source_curves.read_text())
    first, last = args.shot
    if earlier["shot"] != [first, last]:
        raise SystemExit(f"--source-curves covers shot {earlier['shot']}, not {[first, last]}")
    sampled = earlier["frames"]
    source_yaw = [r["yaw"] for r in earlier["source"]["curve"]]

    bootstrap()
    sys.path.append(str(REPO))
    sys.path.append(str(REPO / "workflows"))
    dynamic_vram = server_memory_mode()
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
    import comfy_extras.nodes_sam3d_body as nodes
    import folder_paths
    import h3_config
    import sam3d_body_vith

    needs("a CUDA device: one slice's decode did not finish in ten minutes on the CPU", torch.cuda.is_available())
    with np.load(args.mask) as z:
        mask = torch.from_numpy(z["mask"])
    probe = comfy.utils.load_torch_file(str(files[0][1]))["latent_tensor"]
    canvas = (int(probe.shape[-1]) * 16, int(probe.shape[-2]) * 16)
    fitted = comfy.utils.common_upscale(mask[:, None], canvas[0], canvas[1], "bilinear", "center")[:, 0]
    boxes = Y.boxes(fitted)
    frame_boxes = [boxes[f] for f in sampled]
    start, stop, base = slice_plan(first, last)

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])))
    path = folder_paths.get_full_path_or_raise("detection", Y.RELEASES[args.release])
    patcher = sam3d_body_vith.load_model(path) if args.release == "vith" else \
        nodes.SAM3DBody_Loader.execute(Y.RELEASES[args.release]).args[0]

    def predict(image, box):
        pose = nodes.SAM3DBody_Predict.execute(patcher, image, bboxes=[box], run_hand_refinement=False).args[0]
        people = pose["frames"][0]
        return people[0] if people else None

    began = time.monotonic()
    record = {
        "script": "bench/measure_step_yaw.py", "release": args.release, "device": str(mm.get_torch_device()),
        "dynamic_vram": dynamic_vram, "shot": [first, last], "frames": sampled,
        "latent_slice": [start, stop], "canvas": list(canvas), "tolerance_degrees": args.tolerance,
        "source_curve_from": args.source_curves.name, "steps": [],
    }
    with torch.no_grad():
        for step, file in files:
            import safetensors
            with safetensors.safe_open(str(file), "pt") as f:
                sigma = float((f.metadata() or {}).get("sigma", "nan"))
                latent = f.get_tensor("latent_tensor")
            images = vae.decode(latent[:, :, start:stop].to(torch.float32))
            images = images.reshape(-1, *images.shape[-3:]) if images.ndim == 5 else images
            picked = images[[decoded_index(f, base) for f in sampled]].float().cpu()
            with torch.inference_mode():
                rows = Y.yaw_curve(predict, picked, frame_boxes)
            curve = [r["yaw"] for r in rows]
            result = Y.compare(source_yaw, curve, sampled, args.tolerance)
            entry = {"step": step, "sigma": sigma, "readings": sum(v is not None for v in curve),
                     "end_yaw": result.get("end_yaw"), "end_difference": result.get("end_difference"),
                     "largest_turn": result.get("largest_turn"), "verdict": result["verdict"], "curve": rows}
            record["steps"].append(entry)
            print(f"step {step:2d} sigma {sigma:.4f}: {entry['readings']}/{len(sampled)} readings, "
                  f"end {entry['end_yaw']}, difference {entry['end_difference']}, {entry['verdict']}", flush=True)
            del latent, images
            mm.soft_empty_cache()

        # the control: the last step against the finished file
        span = Y.decode(args.video, first / args.rate, args.rate, last - first + 1, canvas)
        every = sampled[1] - sampled[0] if len(sampled) > 1 else 1
        with torch.inference_mode():
            video_rows = Y.yaw_curve(predict, span[::every], frame_boxes)
    video_curve = [r["yaw"] for r in video_rows]
    last_curve = [r["yaw"] for r in record["steps"][-1]["curve"]]
    pairs = [(a, b) for a, b in zip(last_curve, video_curve) if a is not None and b is not None]
    control = round(sum(Y.apart(a, b) for a, b in pairs) / len(pairs), 1) if pairs else None
    ends = [s["end_difference"] for s in record["steps"]]
    lock = lock_step(ends, args.tolerance)
    record.update({
        "video": {"curve": video_rows, **{k: v for k, v in Y.compare(source_yaw, video_curve, sampled, args.tolerance).items()}},
        "control_last_step_vs_video_mean_degrees": control,
        "control_holds": control is not None and control <= CONTROL_DEGREES,
        "locks_at_step": lock,
        "locks_at_sigma": None if lock is None else record["steps"][lock]["sigma"],
        "seconds": round(time.monotonic() - began, 1),
    })
    print(f"control: the last step and the finished video differ by {control} degrees on average"
          + ("" if record["control_holds"] else f"  -- OVER {CONTROL_DEGREES}: the frames are not lined up, do not read the steps"))
    print(f"facing locks at step {lock}" + ("" if lock is None else f", sigma {record['locks_at_sigma']:.4f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
