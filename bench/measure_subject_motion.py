#!/usr/bin/env python3
"""Does a render's subject move as the source's does: body joints over a shot, render against source.

`bench/measure_subject_yaw.py` answers one question about a moving shot,
which way he faces. A turn is one motion. A walk, a profile throwing darts,
a hand to the mouth are others, and the facing angle says little about any
of them. This reads the whole body.

**What it buys** (card `build-metric-suite`, the owner's second metric): a
number for "moves as the source moves" that does not depend on the motion
being a turn, so the lane's benchmark can grade clips with different
motions. It applies to any shot `measure_subject_yaw.py` has run on: it
reads that script's output and runs no model. It is accepted when it puts
clips judged by eye in the same order, on clips with more than one kind of
motion; until then it is calibrated on one clip whose only motion is a turn.

**The reading.** The pose pass stores seventeen body joints per sampled
frame, two ways:

- `in_body`: 3D, from the middle of the hips, in units of the hips-to-
  shoulders distance, in the camera's axes. A subject of another build,
  size or distance reads the same when the pose is the same. This is the
  main reading.
- `in_box`: 2D, on screen, from the subject's box's corner in units of its
  height. Where he is in the picture as well as how he stands, so it also
  moves when the new subject is narrower or shorter than the old.

**The number.** For each frame the mean distance between the render's
joints and the source's. Then the same for a subject who never moves: the
source's own first pose held for the whole shot. `followed` is one minus
the ratio of the two: 1 when the render is on the source's pose in every
frame, 0 when it is no closer to the source than standing still would be,
below 0 when it moves somewhere else. It is reported for the whole body and
by part (head, torso, arms, legs), at no time shift and at the shift that
fits best, so "follows, a few frames late" is a reading and not a failure.

**The verdict** is `follows` at or above `FOLLOWS`, on the whole body, in
`in_body`, at no shift. Reasoned and stated: closer to the source's motion
than to standing still.

**What it cannot tell apart.** A shot in which the source barely moves has
a standing-still distance near zero, and the ratio means nothing; such a
shot is reported as `too still to grade` below `STILL_FLOOR`. Joints out of
frame or hidden are the body model's guess on both sides. In `in_box` a
different build looks like a different pose. Nothing here sees the hands'
fingers, the face or the mouth.

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

GROUPS = {
    "head": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "arms": ("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
    "legs": ("left_knee", "right_knee", "left_ankle", "right_ankle"),
}
SPACES = ("in_body", "in_box")

# reasoned, set before any clip was scored: at or above it a render is closer
# to the source's motion than to a subject standing still.
FOLLOWS = 0.5
# reasoned: below this the source's own motion over the shot is smaller than
# the body model's frame-to-frame wobble on a still person (a few hundredths
# of a torso length in the 2026-10-05 curves), so a ratio against it is noise.
STILL_FLOOR = {"in_body": 0.08, "in_box": 0.02}
# reasoned: half a second at 24 frames a second, in frames; a render that
# trails the source by more than that is not following it.
MAX_LAG_FRAMES = 12


def joint_distance(a: dict | None, b: dict | None, joints) -> float | None:
    """The mean distance between two readings over `joints`, or None when either frame has none."""
    if not a or not b:
        return None
    return sum(math.dist(a[j], b[j]) for j in joints) / len(joints)


def mean_distance(reference: list, other: list, joints, shift: int = 0, window: int | None = None) -> tuple[float | None, int]:
    """(mean over frames of the distance between reference[i] and other[i + shift], frames compared).

    `window` is the shift whose overlap decides WHICH frames are compared,
    when that is not `shift` itself: the standing-still distance for a
    shifted comparison has to be taken over the same frames, or a render
    that never moves earns credit just by being compared on the frames
    before the source has gone anywhere.
    """
    values = []
    for i, ref in enumerate(reference):
        j = i + shift
        k = i + (shift if window is None else window)
        if 0 <= j < len(other) and 0 <= k < len(other):
            d = joint_distance(ref, other[j], joints)
            if d is not None:
                values.append(d)
    return (sum(values) / len(values) if values else None), len(values)


def followed(distance: float | None, still: float | None) -> float | None:
    if distance is None or still is None or still <= 0:
        return None
    return round(1.0 - distance / still, 3)


def score(source_rows: list[dict], clip_rows: list[dict], space: str, every: int) -> dict:
    """One clip against the source in one space: by part and whole, at no shift and at the best one."""
    src = [r.get(space) for r in source_rows]
    clip = [r.get(space) for r in clip_rows]
    first = next((p for p in src if p), None)
    held = [first if p else None for p in src]              # the source's first pose, never moving
    out = {"parts": {}}
    whole = tuple(j for names in GROUPS.values() for j in names)
    for name, joints in (*GROUPS.items(), ("whole", whole)):
        still, _ = mean_distance(src, held, joints)
        distance, n = mean_distance(src, clip, joints)
        entry = {"distance": None if distance is None else round(distance, 4),
                 "standing_still": None if still is None else round(still, 4),
                 "followed": followed(distance, still), "frames": n}
        if name == "whole":
            out.update(entry)
        else:
            out["parts"][name] = entry
    # the time shift that fits best, on the whole body; half the frames must overlap
    best = None
    span = max(1, MAX_LAG_FRAMES // max(1, every))
    for shift in sorted(range(-span, span + 1), key=abs):   # ties go to the smaller shift
        d, n = mean_distance(src, clip, whole, shift)
        still_here, _ = mean_distance(src, held, whole, 0, window=shift)
        f = followed(d, still_here)
        if f is not None and n >= max(1, len(src) // 2) and (best is None or f > best[0] + 1e-9):
            best = (f, shift)
    if best is not None:
        out["best_shift_frames"] = best[1] * every           # positive: the render is late
        out["followed_at_best_shift"] = best[0]
    floor = STILL_FLOOR[space]
    if out["standing_still"] is None or out["followed"] is None:
        out["verdict"] = "not measured"
    elif out["standing_still"] < floor:
        out["verdict"] = "too still to grade"
    else:
        out["verdict"] = "follows" if out["followed"] >= FOLLOWS else "does not follow"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("record", type=Path, help="a measure_subject_yaw output whose rows carry the joints")
    ap.add_argument("--eye", type=Path, help="by-eye verdicts; a clip's `motion` is used, else its `turn`")
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
        "shot": record["shot"], "frames": frames, "follows_at": FOLLOWS,
        "followed": "1 on the source's pose in every frame, 0 no closer than standing still, below 0 moving elsewhere",
        "clips": {},
    }
    for label, clip in record["clips"].items():
        entry = {space: score(source_rows, clip["curve"], space, every) for space in SPACES}
        entry["verdict"] = entry["in_body"]["verdict"]
        if label in eye:
            entry["eye"] = eye[label]
        out["clips"][label] = entry
        body = entry["in_body"]
        print(f"{label}: followed {body['followed']} (best shift {body.get('best_shift_frames')} frames: "
              f"{body.get('followed_at_best_shift')}), {entry['verdict']}" + (f" (eye: {eye[label]})" if label in eye else ""))
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
