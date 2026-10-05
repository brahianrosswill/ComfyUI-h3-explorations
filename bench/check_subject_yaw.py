#!/usr/bin/env python3
"""The turn metric's arithmetic: angles across the seam, the verdict, and the comparison with the eye.

`bench/measure_subject_yaw.py` turns a body model's keypoints into "did he
turn as the source did". The model is not checked here; its part is the
calibration record. What is checked is the arithmetic around it, which is
wrong quietly in exactly the case the lane cares about: a back-to-camera
pose sits at 180 degrees, where 179 and -179 are two degrees apart and a
plain mean of them says 0, facing the camera.

  yaw_reads_the_shoulder_line   facing the camera is 0, side-on is 90 either
                                way, back to it is 180; the sign follows
                                which shoulder is nearer.
  the_seam_is_two_degrees_wide  179 and -179 are 2 apart, and their mean is
                                180, not 0. The plain mean is computed
                                beside it and must be wrong, or the case
                                tests nothing.
  a_held_turn_holds             a clip that ends where the source ends holds;
                                one that stays facing the camera fails, and
                                the frame it parts from the source is the
                                first from which it stays apart.
  a_gap_is_not_a_reading        frames with nobody found are left out, and a
                                clip with none is "not measured", not "holds".
  order_matches_or_says_where   yes below partial below no passes; a "no"
                                that measured closer than a "yes" is named.
  a_slice_lines_up_with_frames  `bench/measure_step_yaw.py` decodes only the
                                latent frames under a shot: every frame of
                                the shot lands on a decoded index of 1 or
                                more, in order, and the indices are what
                                H3's packing (one frame, then fours) gives.
  joints_are_measured_on_the_body  the motion metric's 3D joints read the
                                same for the same pose at another size and
                                place, and its on-screen joints use the
                                box's height for both axes.
  motion_follows_or_does_not    `bench/measure_subject_motion.py`: the same
                                motion scores 1, the same motion late scores
                                less and is found again at its shift, the
                                same motion from another stance scores 1 with
                                the stance as its offset, and a subject who
                                never moves scores 0 AT EVERY SHIFT and IN
                                ANY POSE, the middle of the source's range
                                included (the two controls: the first forms
                                of the metric failed each).
  a_clip_is_graded_on_the_joints_its_source_moves
                                a source that moves one arm is graded on that
                                arm: the held joints get no score, the same
                                motion on the other arm does not follow and
                                shows as stray, and a nod the source does not
                                make leaves the score alone and shows as
                                stray on the head only.
  a_still_shot_is_not_graded    a source that only wobbles is "too still to
                                grade" whatever the render does.
  the_lock_is_the_last_entry    the facing "locks" at the first step from
                                which it stays within the tolerance to the
                                end; a step that dips in and out again does
                                not count, and a render that ends outside
                                never locks.

No model, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_subject_yaw.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import case, finish  # noqa: E402

import measure_step_yaw as S  # noqa: E402
import measure_subject_motion as M  # noqa: E402
import measure_subject_yaw as Y  # noqa: E402


def _keypoints(left, right):
    k = np.zeros((70, 3))
    k[Y.LEFT_SHOULDER], k[Y.RIGHT_SHOULDER] = left, right
    return k


def yaw_reads_the_shoulder_line():
    L, R = Y.LEFT_SHOULDER, Y.RIGHT_SHOULDER
    facing = _keypoints((0.2, 0, 3.0), (-0.2, 0, 3.0))      # his left on the image's right
    away = _keypoints((-0.2, 0, 3.0), (0.2, 0, 3.0))
    left_far = _keypoints((0.0, 0, 3.2), (0.0, 0, 2.8))
    left_near = _keypoints((0.0, 0, 2.8), (0.0, 0, 3.2))
    assert abs(Y.yaw_of(facing, L, R)) < 1e-9, Y.yaw_of(facing, L, R)
    assert abs(Y.apart(Y.yaw_of(away, L, R), 180.0)) < 1e-9, Y.yaw_of(away, L, R)
    assert abs(Y.yaw_of(left_far, L, R) - 90.0) < 1e-9 and abs(Y.yaw_of(left_near, L, R) + 90.0) < 1e-9
    assert Y.apart(Y.yaw_of(facing, L, R), Y.yaw_of(away, L, R)) == 180.0


def the_seam_is_two_degrees_wide():
    assert Y.apart(179.0, -179.0) == 2.0 and Y.apart(-179.0, 179.0) == 2.0
    assert Y.wrap(540.0) == 180.0 and Y.wrap(-180.0) == 180.0 and Y.wrap(190.0) == -170.0
    mean = Y.circular_mean([179.0, -179.0, 175.0, -175.0])
    assert Y.apart(mean, 180.0) < 1e-6, f"circular mean {mean}"
    plain = sum([179.0, -179.0, 175.0, -175.0]) / 4
    assert Y.apart(plain, 180.0) > 170, "control: the plain mean is right here, so the seam is not being tested"
    assert Y.circular_mean([None, None]) is None and Y.circular_mean([10.0, None]) == 10.0


def a_held_turn_holds():
    frames = list(range(237, 257, 2))
    source = [0, 0, -20, -80, -140, 170, -175, 178, -176, 179]
    held = [5, 2, -10, -60, -120, -160, 170, -170, 175, -178]
    stayed = [3, -2, 4, 1, -3, 5, 0, 2, -1, 3]
    good = Y.compare(source, held, frames, 45.0)
    assert good["verdict"] == "holds" and good["end_difference"] < 10 and good["diverges_at_frame"] is None, good
    bad = Y.compare(source, stayed, frames, 45.0)
    assert bad["verdict"] == "fails" and bad["end_difference"] > 170, bad
    # the source passes 45 degrees between the frames at -20 and -80: the fourth sample
    assert bad["diverges_at_frame"] == frames[3], bad["diverges_at_frame"]
    late = Y.compare(source, [0, 0, -20, -80, -140, 170, 0, 0, 0, 0], frames, 45.0)
    assert late["diverges_at_frame"] == frames[6], "a clip that follows and then lets go parts where it lets go"
    # a part-turn that comes back fails like no turn does, and the largest turn is what tells them apart
    part = Y.compare(source, [4, 4, 6, 14, 30, 38, 33, 17, 0, -10], frames, 45.0)
    assert part["verdict"] == "fails" and bad["verdict"] == "fails"
    assert part["largest_turn"] == 34.0 and bad["largest_turn"] < 10 and good["largest_turn"] > 170, (
        part["largest_turn"], bad["largest_turn"], good["largest_turn"])
    assert part["largest_turn_source"] > 170


def a_gap_is_not_a_reading():
    frames = [10, 12, 14, 16, 18, 20]
    source = [0, -90, 180, 180, 180, 180]
    gappy = [0, None, 170, None, 175, -178]
    out = Y.compare(source, gappy, frames, 45.0)
    assert out["samples"] == 4 and out["verdict"] == "holds", out
    nothing = Y.compare(source, [None] * 6, frames, 45.0)
    assert nothing["verdict"] == "not measured" and nothing["comparable"] is False, nothing
    no_end = Y.compare(source, [0, -80, None, None, None, None], frames, 45.0)
    assert no_end["verdict"] == "not measured", "a clip with no reading at the shot's end got a verdict"


def order_matches_or_says_where():
    eye = {"a": "yes", "b": "yes", "c": "partial", "d": "no", "e": "no"}
    good = Y.ranks_as_the_eye({"a": 10.0, "b": 14.0, "c": 60.0, "d": 170.0, "e": 175.0, "x": 5.0}, eye)
    assert good["same_order"] and good["clips_compared"] == 5 and good["not_judged_by_eye"] == ["x"], good
    bad = Y.ranks_as_the_eye({"a": 10.0, "b": 80.0, "c": 60.0, "d": 170.0, "e": 8.0}, eye)
    assert not bad["same_order"]
    pairs = {(d["turned_more_by_eye"], d["turned_less_by_eye"]) for d in bad["disagreements"]}
    assert ("b", "c") in pairs and ("a", "e") in pairs and ("c", "e") in pairs, pairs
    assert ("a", "d") not in pairs
    skipped = Y.ranks_as_the_eye({"a": 10.0, "d": None}, eye)
    assert skipped["clips_compared"] == 1, "a clip with no measurement was ranked"


def a_slice_lines_up_with_frames():
    # H3's packing, written out independently: latent 0 holds frame 0, latent k holds frames 4k-3 .. 4k
    holds = {0: [0]}
    for k in range(1, 90):
        holds[k] = list(range(4 * k - 3, 4 * k + 1))
    for frame in (0, 1, 4, 5, 237, 240, 241, 279, 344):
        k = S.latent_index(frame)
        assert frame in holds[k], f"frame {frame} is not under latent {k}"
    for first, last in ((237, 279), (1, 8), (5, 5), (100, 344)):
        start, stop, base = S.slice_plan(first, last)
        assert start == S.latent_index(first) - 1 and stop == S.latent_index(last) + 1, (first, last, start, stop)
        # a decoder given latents start..stop-1 yields one frame, then four per latent
        decoded = [None] + [f for k in range(start + 1, stop) for f in holds[k]]
        for frame in range(first, last + 1):
            j = S.decoded_index(frame, base)
            assert j >= 1 and decoded[j] == frame, f"shot {first}-{last}: frame {frame} is decoded index {j}, which is frame {decoded[j]}"
    # the first frames of a clip sit under latent 0, which has no latent before it
    start, stop, base = S.slice_plan(0, 3)
    assert start == 0 and base == 0 and S.decoded_index(1, base) == 1


def the_lock_is_the_last_entry():
    assert S.lock_step([None, 160.0, 150.0, 20.0, 60.0, 15.0, 12.0, 10.0], 45.0) == 5, "a dip inside and out again counted"
    assert S.lock_step([160.0, 150.0, 140.0], 45.0) is None and S.lock_step([10.0, 12.0, 160.0], 45.0) is None
    assert S.lock_step([10.0, 12.0, 9.0], 45.0) == 0 and S.lock_step([], 45.0) is None
    assert S.lock_step([None, None, 30.0], 45.0) == 2 and S.lock_step([30.0, None, 30.0], 45.0) == 2, \
        "a step with no reading is not inside the tolerance"
    assert S.lock_step([30.0, 30.0, None], 45.0) is None, "a last step with no reading locked"


def _swing(t: float, arm: tuple[str, ...] = ("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
           head_nod: float = 0.0) -> dict:
    """A made-up body: the named arm joints swing with t, the head nods by `head_nod`, the rest stays put."""
    pose = {j: [0.0, 0.0, 0.0] for j in M.JOINTS}
    for j in arm:
        pose[j] = [math.sin(t), 0.5, math.cos(t)]
    for j in M.PARTS["head"]:
        pose[j] = [0.0, 1.5 + head_nod, 0.0]
    return pose


def joints_are_measured_on_the_body():
    rng = np.random.default_rng(0)
    k = rng.normal(size=(70, 3))
    k[Y.LEFT_SHOULDER], k[Y.RIGHT_SHOULDER] = (0.2, -0.5, 3.0), (-0.2, -0.5, 3.0)
    k[Y.LEFT_HIP], k[Y.RIGHT_HIP] = (0.15, 0.0, 3.0), (-0.15, 0.0, 3.0)
    base = Y.joints_in_body(k)
    moved = Y.joints_in_body(k * 1.7 + np.array([4.0, -2.0, 9.0]))      # a larger person, somewhere else
    worst = max(abs(a - b) for j in base for a, b in zip(base[j], moved[j]))
    assert worst < 1e-3, f"the same pose at another size and place reads {worst} apart"
    assert abs(math.dist(base["left_shoulder"], base["right_shoulder"]) - 0.8) < 1e-3, "the unit is not the torso's length"
    assert set(base) == set(M.JOINTS), "the pose pass and the motion metric do not name the same joints"
    # on screen: one unit for both axes, the box's height
    box = {"x": 100.0, "y": 50.0, "width": 200.0, "height": 400.0}
    k2 = np.zeros((70, 2))
    k2[Y.BODY_JOINTS["nose"]] = (300.0, 450.0)                          # the box's bottom right corner
    assert Y.joints_in_box(k2, box)["nose"] == [0.5, 1.0], Y.joints_in_box(k2, box)["nose"]


def motion_follows_or_does_not():
    src = [{"in_body": _swing(i * 0.2)} for i in range(20)]
    cases = {
        "same": [{"in_body": _swing(i * 0.2)} for i in range(20)],
        "late": [{"in_body": _swing(max(i - 3, 0) * 0.2)} for i in range(20)],
        "still": [{"in_body": _swing(0.0)} for i in range(20)],
    }
    got = {name: M.score(src, rows, "in_body", 2) for name, rows in cases.items()}
    assert got["same"]["followed"] == 1.0 and got["same"]["verdict"] == "follows" and got["same"]["best_shift_frames"] == 0
    assert 0.2 < got["late"]["followed"] < 0.9, got["late"]["followed"]
    # motion, not pose: the same motion from a different stance is fully followed, and the stance is the offset
    apart = [{"in_body": {j: [v[0] + 0.3, v[1] - 0.2, v[2]] for j, v in _swing(i * 0.2).items()}} for i in range(20)]
    stance = M.score(src, apart, "in_body", 2)
    assert stance["followed"] == 1.0, f"a constant difference in stance cost {1 - stance['followed']}"
    assert abs(stance["joints"]["left_wrist"]["offset"] - math.hypot(0.3, 0.2)) < 1e-3, stance["joints"]["left_wrist"]["offset"]
    # and a subject frozen in the MIDDLE of the source's range earns nothing either
    frozen_mid = M.score(src, [{"in_body": _swing(1.9)} for i in range(20)], "in_body", 2)
    assert frozen_mid["followed"] == 0.0 and frozen_mid["followed_at_best_shift"] == 0.0, frozen_mid["followed"]
    assert got["late"]["best_shift_frames"] == 6 and got["late"]["followed_at_best_shift"] == 1.0, \
        "a render three samples late is not read as six frames late and fully followed there"
    # the control for the shift: a subject who never moves earns nothing at any shift
    assert got["still"]["followed"] == 0.0 and got["still"]["followed_at_best_shift"] == 0.0 \
        and got["still"]["verdict"] == "does not follow", got["still"]
    # the curve is the output: one distance per sampled frame for every joint, zeros when the motions are the same
    wrist = got["late"]["joints"]["left_wrist"]
    assert len(wrist["curve"]) == 20 and min(wrist["curve"]) >= 0.0 and max(wrist["curve"]) > 0.3, wrist["curve"][:5]
    assert got["same"]["joints"]["left_wrist"]["curve"] == [0.0] * 20
    assert stance["joints"]["left_wrist"]["curve"] == [0.0] * 20, "a different stance shows in the curve; it belongs in the offset"


def a_clip_is_graded_on_the_joints_its_source_moves():
    # a dart throw: one wrist and its elbow move, everything else is held
    arm = ("right_elbow", "right_wrist")
    src = [{"in_body": _swing(i * 0.2, arm)} for i in range(20)]
    throws = M.score(src, [{"in_body": _swing(i * 0.2, arm)} for i in range(20)], "in_body", 2)
    assert throws["moved_joints"] == list(arm) and throws["followed"] == 1.0, throws["moved_joints"]
    assert throws["parts"]["hands"]["followed"] == 1.0 and throws["parts"]["head"]["followed"] is None \
        and throws["parts"]["head"]["moved"] is False, "a part the source holds was given a score"
    assert throws["joints"]["left_wrist"]["moved"] is False and throws["joints"]["left_wrist"]["followed"] is None
    # the wrong arm: the motion is there and on the other side, which reads as not following
    wrong = M.score(src, [{"in_body": _swing(i * 0.2, ("left_elbow", "left_wrist"))} for i in range(20)], "in_body", 2)
    assert wrong["verdict"] == "does not follow" and wrong["followed"] == 0.0, wrong["followed"]
    assert wrong["stray"] > 0.1, f"the other arm's motion did not show as stray: {wrong['stray']}"
    # a render that throws as the source does and also nods where the source's head is still:
    # the score is the throw's, and the nod shows as stray on the head and nowhere else
    nods = M.score(src, [{"in_body": _swing(i * 0.2, arm, head_nod=0.02 * i)} for i in range(20)], "in_body", 2)
    assert nods["followed"] == 1.0 and nods["verdict"] == "follows", nods["followed"]
    assert nods["parts"]["head"]["stray"] > 0.05 and nods["parts"]["feet"]["stray"] == 0.0, nods["parts"]["head"]
    assert throws["stray"] == 0.0


def a_still_shot_is_not_graded():
    wobble = [{"in_body": _swing(0.001 * (i % 2))} for i in range(20)]
    out = M.score(wobble, [{"in_body": _swing(0.3)} for i in range(20)], "in_body", 2)
    assert out["verdict"] == "too still to grade" and out["moved_joints"] == [], out["verdict"]
    gone = M.score(wobble, [{"in_body": None} for i in range(20)], "in_body", 2)
    assert gone["verdict"] == "not measured" and gone["followed"] is None


def main() -> int:
    for fn in (yaw_reads_the_shoulder_line, the_seam_is_two_degrees_wide, a_held_turn_holds,
               a_gap_is_not_a_reading, order_matches_or_says_where, a_slice_lines_up_with_frames,
               the_lock_is_the_last_entry, joints_are_measured_on_the_body, motion_follows_or_does_not,
               a_clip_is_graded_on_the_joints_its_source_moves, a_still_shot_is_not_graded):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
