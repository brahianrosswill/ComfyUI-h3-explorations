#!/usr/bin/env python3
"""Does a render's subject move as the source's does: every body joint over a shot, frame by frame, render against source.

`bench/measure_subject_yaw.py` answers one question about a moving shot,
which way he faces. A turn is one motion. A walk, a profile throwing darts,
a fork to the mouth are others, and each uses different joints. This reads
every joint, and grades a clip on the joints its source moves.

**What it buys** (card `build-metric-suite`, the owner's second metric,
generalised at the owner's word on 2026-10-05): one tool for any motion.
The output is the curves, one per joint, a distance from the source in every
sampled frame; the verdict is a reading of them by a rule stated below, so
when the rule is wrong for a clip the curves are still there. It applies to
any shot `measure_subject_yaw.py` has run on: it reads that script's output
and runs no model. It is accepted, per kind of motion, when it orders clips
judged by eye the same way.

**The reading.** The pose pass stores seventeen body joints per sampled
frame, two ways:

- `in_body`: 3D, from the middle of the hips, in units of the hips-to-
  shoulders distance, in the camera's axes. A subject of another build,
  size or distance reads the same when the pose is the same. The verdict is
  on this reading.
- `in_box`: 2D, on screen, from the subject's box's corner in units of its
  height. It also moves when the new subject is narrower or shorter than
  the old, so it is recorded and gives no verdict.

**Motion, not pose.** Each side's joint is measured from its own mean
position over the shot before the two are compared. A render that stands a
little differently from the source and moves exactly as it does is on the
source's motion; where it stands is reported apart, as `offset`.

**Per joint, per frame:** `curve`, the distance between the render's
displacement and the source's. **Per joint:** `source_travel`, how far the
source's joint is on average from its own mean position, which is exactly
what any subject who never moves is off by; `followed`, one minus distance
over travel: 1 moving as the source's joint does in every frame, 0 for a
joint that never moves, below 0 for one that moves another way;
`render_travel`, the same spread for the render's own joint; and `offset`,
the distance between the two mean positions.

**Which joints count.** A joint the source moves less than `MOVED_FLOOR` is
held, not moved: its `followed` is not computed, because a ratio against
nearly nothing is noise. A dart throw moves an arm and holds the rest, and
is graded on the arm.

**Per clip:**
- `followed`: the same ratio over the moved joints together, weighted by
  how far each moves, so the joints that carry the motion carry the score.
  At no time shift, and at the shift that fits best (`best_shift_frames`,
  positive when the render is late), so "follows, late" is a reading.
- `stray`: over the joints the source HOLDS, how much more the render's
  spread than the source's do. A render that gestures
  where the source is still shows here and nowhere else.
- `parts`: the same two numbers for the head, the shoulders, the elbows,
  the hands (the wrists; fingers are not read), the hips, the knees and the
  feet (the ankles).

**The verdict, a rule and not a measurement:** `follows` when `followed` is
at or above `FOLLOWS` on the moved joints, at no time shift; `does not
follow` below it; `too still to grade` when the source moves no joint past
the floor. `stray` and the best shift are reported beside it and do not
change it. `FOLLOWS` was set from one clip's calibration and says so where
it is defined.

**What it cannot tell apart.** Joints out of frame or hidden are the body
model's guess on both sides. A render that does the source's motion with
the other arm reads as not following. Fingers, the face and the mouth are
not read. `in_box` cannot tell a different build from a different pose.

    <python> bench/measure_subject_motion.py <measure_subject_yaw record with joints> \\
        [--eye bench/turn_metric_eye_verdicts.json] --out <record.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import measure_subject_yaw as Y  # noqa: E402

PARTS = {
    "head": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    "shoulders": ("left_shoulder", "right_shoulder"),
    "elbows": ("left_elbow", "right_elbow"),
    "hands": ("left_wrist", "right_wrist"),
    "hips": ("left_hip", "right_hip"),
    "knees": ("left_knee", "right_knee"),
    "feet": ("left_ankle", "right_ankle"),
}
JOINTS = tuple(j for names in PARTS.values() for j in names)
SPACES = ("in_body", "in_box")

# measured, and set AFTER the band-clip calibration, which is the weakness of
# it: on those thirteen renders the ten that do not turn score within 0.04
# of zero and the three that turn score 0.34 and up (0.51 and up at their
# best time shift, being two to six frames late in a turn that takes about
# eighteen). A quarter sits between the two groups. It has met no other
# motion; read the curves before trusting it on one.
FOLLOWS = 0.25
# reasoned: a joint whose source travels less than this is held. The body
# model's own frame-to-frame wobble on a still person is a few hundredths of
# a torso length in the 2026-10-05 curves, and a tenth of the box's height
# on screen is a hand's width.
MOVED_FLOOR = {"in_body": 0.08, "in_box": 0.02}
# reasoned: half a second at 24 frames a second, in frames; a render that
# trails the source by more than that is not following it.
MAX_LAG_FRAMES = 12


def _pairs(n: int, shift: int) -> list[tuple[int, int]]:
    """(source index, render index) for every frame both have at this shift."""
    return [(i, i + shift) for i in range(n) if 0 <= i + shift < n]


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _dist(a, b, joint) -> float | None:
    return math.dist(a[joint], b[joint]) if a and b else None


def _centre(poses: list, joint: str) -> dict | None:
    """The mean position of a joint over the frames that have one, as a one-joint pose."""
    points = [p[joint] for p in poses if p]
    if not points:
        return None
    return {joint: [sum(c) / len(points) for c in zip(*points)]}


def joint_numbers(src: list, clip: list, joint: str, shift: int = 0) -> dict:
    """One joint over the frames both have a reading at `shift`: the curve, the two travels, the offset.

    Each side's joint is taken from its OWN mean position over those frames
    before the two are compared, so the curve is the difference between two
    motions and not between two poses: a render that stands differently and
    moves the same has a curve of zeros, and a render that never moves has
    exactly the source's travel, whatever pose it is frozen in. The
    difference between the two mean positions is `offset`, the pose.

    Two earlier forms were wrong in ways a made-up body showed on 2026-10-05:
    positions compared directly charged a constant difference in stance to
    every frame, and a baseline of "the source's first pose held" let a
    render frozen in the middle of the source's range score above zero.
    """
    pairs = _pairs(len(src), shift)
    # only frames where both have a reading, so the two means are over the same frames
    both = [(src[i], clip[j]) for i, j in pairs if src[i] and clip[j]]
    slots = [bool(src[i] and clip[j]) for i, j in pairs]
    centre_src, centre_clip = _centre([a for a, _b in both], joint), _centre([b for _a, b in both], joint)
    if centre_src is None:
        return {"curve": [None] * len(pairs), "distance": None, "source_travel": None, "render_travel": None,
                "offset": None, "frames": 0}
    cs, cc = centre_src[joint], centre_clip[joint]
    moved = iter([math.dist([p - q for p, q in zip(a[joint], cs)], [p - q for p, q in zip(b[joint], cc)])
                  for a, b in both])
    curve = [next(moved) if has else None for has in slots]
    return {
        "curve": curve,
        "distance": _mean(curve),
        "source_travel": _mean(math.dist(a[joint], cs) for a, _b in both),
        "render_travel": _mean(math.dist(b[joint], cc) for _a, b in both),
        "offset": math.dist(cs, cc),
        "frames": len(both),
    }


def followed_over(numbers: dict[str, dict], joints, floor: float) -> tuple[float | None, list[str]]:
    """(one minus summed distance over summed travel on the moved joints among `joints`, which those are)."""
    moved = [j for j in joints if numbers[j]["source_travel"] is not None and numbers[j]["source_travel"] >= floor
             and numbers[j]["distance"] is not None]
    travel = sum(numbers[j]["source_travel"] for j in moved)
    if not moved or travel <= 0:
        return None, moved
    return round(1.0 - sum(numbers[j]["distance"] for j in moved) / travel, 3), moved


def stray_over(numbers: dict[str, dict], joints, floor: float) -> float | None:
    """On the joints the source holds: how much further the render's travel than the source's, on average."""
    held = [j for j in joints if numbers[j]["source_travel"] is not None and numbers[j]["source_travel"] < floor
            and numbers[j]["render_travel"] is not None]
    if not held:
        return None
    return round(sum(numbers[j]["render_travel"] - numbers[j]["source_travel"] for j in held) / len(held), 4)


def score(source_rows: list[dict], clip_rows: list[dict], space: str, every: int, curves: bool = True) -> dict:
    """One clip against the source in one space: every joint, the parts, the clip, and the best time shift."""
    src = [r.get(space) for r in source_rows]
    clip = [r.get(space) for r in clip_rows]
    floor = MOVED_FLOOR[space]
    numbers = {j: joint_numbers(src, clip, j) for j in JOINTS}
    whole, moved = followed_over(numbers, JOINTS, floor)
    out = {
        "followed": whole,
        "moved_joints": moved,
        "held_joints": [j for j in JOINTS if j not in moved and numbers[j]["source_travel"] is not None],
        "stray": stray_over(numbers, JOINTS, floor),
        "frames": max((numbers[j]["frames"] for j in JOINTS), default=0),
        "parts": {},
        "joints": {},
    }
    for part, joints in PARTS.items():
        f, part_moved = followed_over(numbers, joints, floor)
        out["parts"][part] = {"followed": f, "moved": bool(part_moved), "stray": stray_over(numbers, joints, floor)}
    for j in JOINTS:
        n = numbers[j]
        is_moved = j in moved
        entry = {
            "moved": is_moved,
            "source_travel": None if n["source_travel"] is None else round(n["source_travel"], 4),
            "render_travel": None if n["render_travel"] is None else round(n["render_travel"], 4),
            "distance": None if n["distance"] is None else round(n["distance"], 4),
            "offset": None if n["offset"] is None else round(n["offset"], 4),
            "followed": (round(1.0 - n["distance"] / n["source_travel"], 3) if is_moved else None),
        }
        if curves:
            entry["curve"] = [None if v is None else round(v, 4) for v in n["curve"]]
        out["joints"][j] = entry

    # the time shift that fits best on the moved joints; half the frames must overlap; ties to the smaller shift
    best = None
    span = max(1, MAX_LAG_FRAMES // max(1, every))
    for shift in sorted(range(-span, span + 1), key=abs):
        shifted = {j: joint_numbers(src, clip, j, shift) for j in moved}
        if not moved or min(shifted[j]["frames"] for j in moved) < max(1, len(src) // 2):
            continue
        travel = sum(shifted[j]["source_travel"] for j in moved)
        if travel <= 0:
            continue
        f = round(1.0 - sum(shifted[j]["distance"] for j in moved) / travel, 3)
        if best is None or f > best[0] + 1e-9:
            best = (f, shift)
    if best is not None:
        out["best_shift_frames"] = best[1] * every           # positive: the render is late
        out["followed_at_best_shift"] = best[0]

    if not out["frames"]:
        out["verdict"] = "not measured"
    elif whole is None:
        out["verdict"] = "too still to grade"
    else:
        out["verdict"] = "follows" if whole >= FOLLOWS else "does not follow"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("record", type=Path, help="a measure_subject_yaw output whose rows carry the joints")
    ap.add_argument("--eye", type=Path, help="by-eye verdicts; a clip's `motion` is used, else its `turn`")
    ap.add_argument("--no-curves", action="store_true", help="leave the per-frame curves out of the output")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    record = json.loads(args.record.read_text())
    source_rows = record["source"]["curve"]
    if not any(r.get("in_body") for r in source_rows):
        raise SystemExit(f"{args.record.name} carries no joints: it was written before the pose pass kept them. "
                         "Run bench/measure_subject_yaw.py again.")
    frames = record["frames"]
    every = frames[1] - frames[0] if len(frames) > 1 else 1
    eye = {}
    if args.eye:
        eye = {label: row.get("motion", row["turn"]) for label, row in json.loads(args.eye.read_text())["clips"].items()}

    out = {
        "script": "bench/measure_subject_motion.py", "poses_from": args.record.name,
        "shot": record["shot"], "frames": frames, "follows_at": FOLLOWS, "moved_floor": MOVED_FLOOR,
        "followed": "1 moving as the source's joints do in every frame, 0 for a subject who never moves, below 0 moving another way; over the joints the source moves, each from its own mean position",
        "stray": "on the joints the source holds, how much further the render's go than the source's; units of the space",
        "clips": {},
    }
    for label, clip in record["clips"].items():
        entry = {space: score(source_rows, clip["curve"], space, every, curves=not args.no_curves) for space in SPACES}
        entry["verdict"] = entry["in_body"]["verdict"]
        if label in eye:
            entry["eye"] = eye[label]
        out["clips"][label] = entry
        body = entry["in_body"]
        print(f"{label}: followed {body['followed']} on {len(body['moved_joints'])} moved joint(s) "
              f"(best shift {body.get('best_shift_frames')} frames: {body.get('followed_at_best_shift')}), "
              f"stray {body['stray']}, {entry['verdict']}" + (f" (eye: {eye[label]})" if label in eye else ""))
    if eye:
        # smaller is better for the order check, so it is given what was NOT followed
        measured = {label: (None if c["in_body"]["followed"] is None else round(1.0 - c["in_body"]["followed"], 3))
                    for label, c in out["clips"].items()}
        against = Y.ranks_as_the_eye(measured, eye)
        against["note"] = "the figures here are one minus `followed`, so smaller is closer to the source"
        out["against_the_eye"] = against
        print(json.dumps({k: against[k] for k in ("clips_compared", "same_order", "disagreements")}, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
