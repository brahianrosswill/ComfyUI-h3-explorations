#!/usr/bin/env python3
"""The per-shot table says what the tracker did, and its person numbers mean one thing everywhere.

`shot_table.py` turns the Subject Track's per-shot state into a table a
person reviews in place of a render. It is wrong quietly: a row that names
the wrong person sends a reviewer's correction to somebody else, and a shot
reported on screen when its mask is empty is a shot nobody looks at again.
The tracker's own logic is driven here (`subject_track.follow`) on the
stand-in world `bench/check_subject_track.py` builds, so the table is read
off a real result and not off rows typed for the purpose.

  numbering_is_left_to_right     person K is the K-th mask from the left,
                                 ties to the higher one, from 1; the same
                                 person keeps the same number whatever order
                                 the detector returns; `detection_of` undoes
                                 `person_number`. Delete and the tile, the
                                 table and a typed correction can each mean a
                                 different person by "person 2".
  rows_cover_the_clip            one row per shot, every frame in exactly one.
  subject_number_names_the_mask  the number a row gives the subject resolves
                                 to the mask the tracker took, on a shot
                                 where the detector's order and the numbering
                                 DIFFER (asserted first, or the case could
                                 not fail).
  absent_shot_says_so            a shot holding only other people is absent,
                                 not on screen, names no subject, and names
                                 the closest person, who is on that frame.
  on_screen_is_counted           frames with the subject are counted off the
                                 mask handed in, not off the shot's length.
  every_row_has_the_slots        the caption slot, the version and the
                                 numbering sentence are there; `corrected`
                                 is empty on a tracker with no corrections
                                 and carries the tracker's own words when a
                                 shot has them, in the JSON and the text.
  text_says_what_json_says       the text has one row per shot with the same
                                 numbers, and the JSON survives a round trip.
  labels_sit_on_their_outlines   each number's point lies on the top edge of
                                 its own mask's box.
  save_node_writes_three_files   the save node writes JSON, text and one
                                 picture under core's counter, and refuses a
                                 string that is not a table and a table of
                                 another version.
  tracker_emits_the_table        the Subject Track declares the table as its
                                 last output, after mask, preview, report.
                                 Skipped until the tracker carries it.

What this cannot check: that the numbers are legible on a real tile, or that
a person finds the table quicker than a render. No model, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_shot_table.py
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import case, finish, skip  # noqa: E402

# The tracker's check puts ComfyUI on its CPU path, loads `subject_track` as
# a member of a stand-in package, and builds the world this check reads.
import check_subject_track as T  # noqa: E402

st = T.st
tbl = importlib.import_module("_h3pack.shot_table")
H, W = T.H, T.W
CUTS = [8, 16, 24]


def _run(mask_edit=None, found_edit=None):
    """The stand-in world followed, and its table. Returns (table, found, detect, boxes)."""
    boxes, detect, sign, track, _calls = T._world()
    found = st.follow(32, CUTS, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    if found_edit:
        found_edit(found)
    mask = st.assemble(32, H, W, found.pieces)
    if mask_edit:
        mask = mask_edit(mask)
    table = tbl.build(found, detect, mask, state=st._state, phrase="person", pick=st.PICK_LARGEST,
                      named_frame=True, named_value=True, cuts=CUTS)
    return table, found, detect, boxes


def _box(x0, y0, x1, y1):
    t = torch.zeros((H, W))
    t[y0:y1, x0:x1] = 1.0
    return t


def numbering_is_left_to_right():
    left, middle, right = _box(2, 20, 12, 60), _box(50, 5, 70, 60), _box(100, 10, 120, 60)
    people = {"left": left, "middle": middle, "right": right}
    for order in (("right", "left", "middle"), ("middle", "right", "left"), ("left", "middle", "right")):
        masks = torch.stack([people[name] for name in order])
        numbers = {name: tbl.person_number(masks, i) for i, name in enumerate(order)}
        assert numbers == {"left": 1, "middle": 2, "right": 3}, f"detector order {order} numbered {numbers}"
        for i in range(3):
            assert tbl.detection_of(masks, tbl.person_number(masks, i)) == i, "detection_of does not undo person_number"
    upper, lower = _box(40, 2, 60, 20), _box(40, 40, 60, 58)   # the same centre column
    assert tbl.person_order(torch.stack([lower, upper])) == [1, 0], "a tie did not go to the higher mask"
    assert tbl.person_order(torch.zeros((0, H, W))) == [], "no detections must number nobody"
    masks = torch.stack([left, right])
    assert tbl.detection_of(masks, 0) is None and tbl.detection_of(masks, 3) is None, "a person that is not there resolved"
    assert tbl.person_number(masks, None) is None and tbl.person_number(masks, 5) is None


def rows_cover_the_clip():
    table, _found, _detect, _boxes = _run()
    rows = table["shots"]
    assert [r["shot"] for r in rows] == [1, 2, 3, 4], f"shots numbered {[r['shot'] for r in rows]}"
    covered = [f for r in rows for f in range(r["first_frame"], r["last_frame"] + 1)]
    assert covered == list(range(32)), "the rows do not cover every frame exactly once"
    assert all(r["frames"] == r["last_frame"] - r["first_frame"] + 1 for r in rows)
    assert table["frames"] == 32 and table["size"] == [W, H] and table["cuts"] == CUTS


def subject_number_names_the_mask():
    table, found, detect, boxes = _run()
    third = table["shots"][2]
    masks, _ = detect(third["shown_frame"])
    # the control's precondition: on this shot the detector returns the subject second and he stands leftmost
    assert found.shots[2].index == 1 and third["subject"]["person"] == 1, (
        f"shot 3 no longer has the detector's order and the numbering disagree (index "
        f"{found.shots[2].index}, person {third['subject']['person']}), so this case cannot fail")
    for row, shot in zip(table["shots"], found.shots):
        if shot.seed is None:
            continue
        masks, _ = detect(row["shown_frame"])
        index = tbl.detection_of(masks, row["subject"]["person"])
        assert index is not None and torch.equal(masks[index], boxes[0]), (
            f"shot {row['shot']}: person {row['subject']['person']} is not the mask the tracker took")
        assert row["subject"]["seed_frame"] == shot.seed and row["closest_person"] is None
    first = table["shots"][0]["subject"]
    assert first["state"] == "picked" and "largest" in first["why"] and "frame 3" in first["why"] \
        and "the frame named" in first["why"], f"the picked shot's why: {first['why']!r}"
    assert "0.80" in third["subject"]["why"] and "frame 20" in third["subject"]["why"], third["subject"]["why"]
    return "the detector's order and the numbering differ on shot 3, and the number still names him"


def absent_shot_says_so():
    table, _found, detect, boxes = _run()
    row = table["shots"][1]
    subject = row["subject"]
    assert subject["state"] == "absent" and subject["person"] is None and subject["seed_frame"] is None, subject
    assert row["on_screen"] is False and row["frames_with_subject"] == 0
    assert "0.50" in subject["why"] and "under the line 0.80" in subject["why"], subject["why"]
    masks, _ = detect(row["shown_frame"])
    assert len(row["people"]) == int(masks.shape[0]) == 2
    closest = tbl.detection_of(masks, row["closest_person"])
    assert closest is not None and not torch.equal(masks[closest], boxes[0]), "the closest person is not on that frame"
    assert [p["person"] for p in row["people"]] == [1, 2] and row["people"][0]["box"][0] < row["people"][1]["box"][0], \
        "people are not listed left to right"


def on_screen_is_counted():
    def drop_first_four(mask):
        mask = mask.clone()
        mask[0:4] = 0
        return mask
    whole, *_ = _run()
    cut, *_ = _run(drop_first_four)
    assert whole["shots"][0]["frames_with_subject"] == 8, whole["shots"][0]
    assert cut["shots"][0]["frames_with_subject"] == 4 and cut["shots"][0]["on_screen"] is True, cut["shots"][0]
    assert whole["shots"][2]["frames_with_subject"] == 8, "a subject who enters late is tracked back to the shot's start"


def every_row_has_the_slots():
    table, *_ = _run()
    assert table["table"] == "h3 shot table" and table["version"] == tbl.TABLE_VERSION
    assert table["numbering"] == tbl.NUMBERING and "left to right" in tbl.NUMBERING
    assert all(row["caption"] == "" for row in table["shots"]), "a row has no empty caption slot"
    assert all(row["corrected"] == "" for row in table["shots"]), "a shot nobody corrected says it was"

    def correct_second(found):
        found.shots[1].corrected = "person 2"     # the field mrorange's corrections add to `Shot`
    corrected, *_ = _run(found_edit=correct_second)
    assert [row["corrected"] for row in corrected["shots"]] == ["", "person 2", "", ""], corrected["shots"][1]
    assert "(corrected: person 2)" in tbl.as_text(corrected), "the text does not show a correction"
    # a correction is the reason, whatever the automatic pass scored: taken by hand, and removed by hand
    def take_second(found):
        shot = found.shots[1]
        shot.corrected, shot.seed = "person 2", shot.shown
    taken, *_ = _run(found_edit=take_second)
    why = taken["shots"][1]["subject"]["why"]
    assert why == f"corrected by hand: person 2 on frame {taken['shots'][1]['shown_frame']}", why
    assert taken["shots"][1]["subject"]["person"] is not None and "under the line" not in why

    def remove_third(found):
        shot = found.shots[2]
        shot.corrected, shot.seed = "none", None
    removed, *_ = _run(found_edit=remove_third)
    row = removed["shots"][2]
    assert row["subject"]["why"] == "corrected by hand: none" and row["subject"]["person"] is None, row["subject"]
    assert row["closest_person"] == 1, "a shot emptied by hand no longer names who the automatic pass took"
    assert table["pick_frame"] == 3 and table["pick_frame_named"] is True and table["match_named"] is True


def text_says_what_json_says():
    table, *_ = _run()
    text = tbl.as_text(table)
    rows = [ln for ln in text.splitlines() if ln.startswith("| ") and ln[2].isdigit()]
    assert len(rows) == 4, f"{len(rows)} text rows for 4 shots"
    assert "| 3 | 16-23 | 20 | 2 | person 1, taken | 8 of 8 frames |" in text, text
    assert "| 2 | 8-15 |" in text and "absent (closest: person" in text and "| no |" in text, text
    assert json.loads(tbl.as_json(table)) == table, "the table does not survive JSON"


def labels_sit_on_their_outlines():
    table, _found, detect, _boxes = _run()
    for row in table["shots"]:
        masks, _ = detect(row["shown_frame"])
        points = tbl.label_points(masks)
        assert [n for n, _x, _y in points] == [p["person"] for p in row["people"]]
        for (number, x, y), person in zip(points, row["people"]):
            bx, by, bw, _bh = person["box"]
            assert bx <= x <= bx + bw and y == by, f"shot {row['shot']}: person {number}'s label is off its box"


def save_node_writes_three_files():
    import folder_paths
    table, _found, _detect, _boxes = _run()
    node = tbl.MiniMaxH3SaveShotTable
    schema = node.define_schema()
    assert getattr(schema, "is_output_node", False), "the save node must be an output node, or nothing runs it"
    assert [i.id for i in schema.inputs] == ["shot_table", "preview", "filename_prefix"]
    tiles = torch.rand((4, 20, 32, 3))
    before = folder_paths.get_output_directory()
    with tempfile.TemporaryDirectory() as tmp:
        folder_paths.set_output_directory(tmp)
        try:
            node.execute(tbl.as_json(table), filename_prefix="sub/clip", preview=tiles)
            node.execute(tbl.as_json(table), filename_prefix="sub/clip")
            written = sorted(p.name for p in (Path(tmp) / "sub").iterdir())
            assert written == ["clip_00001_shots.json", "clip_00001_shots.md", "clip_00001_shots.png",
                               "clip_00002_shots.json", "clip_00002_shots.md"], written
            assert json.loads((Path(tmp) / "sub" / "clip_00001_shots.json").read_text()) == table
            assert (Path(tmp) / "sub" / "clip_00001_shots.md").read_text().strip() == tbl.as_text(table)
            from PIL import Image
            assert Image.open(Path(tmp) / "sub" / "clip_00001_shots.png").size == (32, 80), "the tiles are not stacked"
            for bad, why in (('{"shots": []}', "not a shot table"),
                             (json.dumps(dict(table, version=tbl.TABLE_VERSION + 1)), "version")):
                try:
                    node.execute(bad, filename_prefix="sub/clip")
                except ValueError as exc:
                    assert why in str(exc), f"refused for the wrong reason: {exc}"
                else:
                    raise AssertionError(f"the save node accepted a string that is {why}")
        finally:
            folder_paths.set_output_directory(before)
    return "two saves, the second under the next number"


def tracker_emits_the_table():
    outputs = st.MiniMaxH3SubjectTrack.define_schema().outputs
    names = [getattr(o, "display_name", None) for o in outputs]
    if "shot_table" not in names:
        skip("the Subject Track does not carry the table yet (subject_track.py is shared; agreed order with mrorange)")
    assert names == ["mask", "preview", "report", "shot_table"], names
    assert outputs[0].io_type == "MASK" and outputs[3].io_type == "STRING"


def main() -> int:
    torch.set_grad_enabled(False)
    for fn in (numbering_is_left_to_right, rows_cover_the_clip, subject_number_names_the_mask,
               absent_shot_says_so, on_screen_is_counted, every_row_has_the_slots,
               text_says_what_json_says, labels_sit_on_their_outlines, save_node_writes_three_files,
               tracker_emits_the_table):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
