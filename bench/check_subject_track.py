#!/usr/bin/env python3
"""Following one person across cuts: the node's own logic, with stand-ins for SAM 3.

`subject_track.py` is the module. It fails quietly in the ways that matter: a
missed cut carries a mask into a shot the subject is not in, a wrong match
replaces somebody else, and a shot left empty by mistake shows the original.
The model work is behind three callables (`detect`, `sign`, `track`), so each
item drives the module's own functions on made-up frames and masks.

1. **A cut is found and a lighting change is not.** Two different pictures
   back to back score above the default threshold; the same picture with its
   colour and brightness changed scores far below it. The control is the plain
   mean difference, which the lighting change must move more than the cut
   score does, or the gradient score is not buying anything.
2. **The shots cover every frame once**, whatever the cuts given, including a
   cut at 0, one past the end and a repeated one.
2a. **The phrase asks core for every person, not one.** Core's SAM 3 prompt
   parser reads a bare phrase as one detection and `name:N` as up to N.
   `counted` writes the second form, keeps a count the user wrote, and the
   check reads the result back through core's own parser. The control: core
   still reads a bare phrase as one.
3. **The pick is the one the rule names**: the largest mask, the mask nearest
   the frame's centre, the highest score. No detections is no pick.
3a. **The signature is taken from the head and shoulders.** `top_third`
   keeps the top third of the rows a mask covers; two people with different
   heads over the same clothes are alike on the whole mask and unlike on the
   top third.
4. **The subject is followed, and nobody else is.** With three people whose
   signatures are known, the picked one is taken in each shot they are in, a
   shot holding only the others is left empty, a subject who enters after a
   shot's first frame is still found and tracked back to the shot's start, and
   the picked shot is tracked both ways from the pick frame. The comparison is
   relative to the other people on the pick frame, who then score 0; picked on
   a frame with nobody else it is the plain similarity, and the report says
   which.
5. **The mask is one per frame, at the frames' size, and empty where the
   subject is absent.** A threshold nothing can pass leaves every shot but the
   picked one empty; nothing on the pick frame leaves everything empty and the
   report says so.
6. **The node declares the phrase and every threshold as inputs**, with the
   module's constants as their defaults, and three outputs with the mask
   first. It is not an output node and it declares a `MASK_VERSION`, the two
   things `mask_store.py` needs for a kept mask to spare the tracker and to
   go stale when the node's method changes.

What this cannot check: that SAM 3's features tell real people apart, that
its tracker follows them, or that the text encoder loads. Those need the card;
`docs/research/masking/2026-10-04_mrhf.md` has what was measured.

No model, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_subject_track.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "nodes.py").exists() and (p / "workflows").is_dir())
COMFY = REPO.parent.parent
# a draft of the module checked beside its own copy of this file, before it replaces the repo's
MODULE = HERE / "subject_track.py" if (HERE / "subject_track.py").exists() else REPO / "subject_track.py"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # no CUDA context for a logic check; the sibling checks do the same


def _load():
    """`subject_track` as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.subject_track", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.subject_track"] = module
    spec.loader.exec_module(module)
    return module


st = _load()

H, W = 72, 128


def _picture(seed: int, rows: int, cols: int) -> torch.Tensor:
    """A frame of blocks, [H, W, 3] in 0..1: structure a cut score can see.

    Two pictures on the same block grid share where their edges are, which is
    what a held shot looks like to the score, so a cut needs another grid.
    """
    g = torch.Generator().manual_seed(seed)
    small = torch.rand((1, 3, rows, cols), generator=g)
    return torch.nn.functional.interpolate(small, size=(H, W), mode="nearest")[0].movedim(0, -1)


def _box(x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    m = torch.zeros((H, W), dtype=torch.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def check_cuts(problems):
    a, b = _picture(1, 6, 8), _picture(2, 9, 5)
    relit = (a * torch.tensor([0.4, 1.0, 0.7]) + 0.15).clamp(0, 1)      # same picture, another light
    frames = torch.stack([a, a, relit, relit, b, b], dim=0)
    scores = st.cut_scores(frames)
    if tuple(scores.shape) != (5,):
        problems.append(f"cut_scores returned {tuple(scores.shape)} for six frames; one score per step is five")
        return
    cuts = st.find_cuts(scores, st.CUT_THRESHOLD)
    if cuts != [4]:
        problems.append(f"cuts at the default threshold are {cuts}; the only cut is into frame 4 "
                        f"(scores {[round(float(s), 3) for s in scores]})")
    if not float(scores[1]) < 0.5 * st.CUT_THRESHOLD:
        problems.append(f"a lighting change scores {float(scores[1]):.3f}, not far below the threshold {st.CUT_THRESHOLD}")
    plain = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
    if not float(plain[1]) / float(plain[3]) > float(scores[1]) / float(scores[3]):
        problems.append("the control failed: the plain difference is no more fooled by the lighting change than "
                        "the gradient score is, so the score buys nothing on this case")
    if st.find_cuts(st.cut_scores(frames[:1]), st.CUT_THRESHOLD) != []:
        problems.append("a one-frame clip has a cut")
    if st.find_cuts(torch.tensor([0.2, 0.9, 0.89]), 0.9) != [2]:
        problems.append("a step scoring exactly the threshold is not a cut: the comparison must be inclusive")
    # the threshold from the scores. The two lists are the highest steps of the two clips measured on 2026-10-04:
    # the band segment, whose cuts score 0.99 and above, and the one-person clip, two of whose cuts score under 0.9.
    low = [0.17] * 40
    band = [1.08, 1.02, 1.02, 1.01, 1.01, 1.01, 1.00, 0.99, 0.76, 0.71, 0.65, 0.64, 0.63] + low
    solo = [0.99, 0.98, 0.95, 0.93, 0.90, 0.84, 0.46, 0.46, 0.45, 0.44] + low
    for name, scores, n_cuts in (("the band clip", band, 8), ("the one-person clip", solo, 6)):
        at = st.auto_cuts(scores)
        found = len(st.find_cuts(torch.tensor(scores), at))
        if found != n_cuts:
            problems.append(f"the automatic cut threshold on {name}'s scores is {at:.2f} and finds {found} cut(s), not {n_cuts}")
    if len(st.find_cuts(torch.tensor(solo), st.CUT_THRESHOLD)) == 6:
        problems.append("the control failed: the fixed threshold finds all six cuts of the one-person clip, "
                        "so the automatic threshold buys nothing on this case")
    for scores, why in (([0.46, 0.41, 0.30, 0.05], "no step near a cut"), ([0.95, 0.93, 0.90, 0.86, 0.83, 0.79], "no clear gap"),
                        ([], "no steps")):
        if st.auto_cuts(scores) != st.CUT_THRESHOLD:
            problems.append(f"the automatic cut threshold is {st.auto_cuts(scores):.2f} with {why}; it falls back to {st.CUT_THRESHOLD}")
    if "0.99 0.95 | 0.40" not in st.cuts_line([0.4, 0.99, 0.95], 0.9, False) or "automatic, 0.90" not in st.cuts_line([0.4], 0.9, False):
        problems.append("the report's cut line does not print the steps in order with the threshold marked")


def check_ranges(problems):
    for n, cuts in ((10, []), (10, [4]), (10, [0, 4, 4, 10, 12]), (1, [])):
        ranges = st.shot_ranges(n, cuts)
        covered = [f for s, e in ranges for f in range(s, e)]
        if covered != list(range(n)):
            problems.append(f"shot_ranges({n}, {cuts}) = {ranges} does not cover each frame once")
    if st.shot_ranges(10, [4]) != [(0, 4), (4, 10)]:
        problems.append(f"shot_ranges(10, [4]) = {st.shot_ranges(10, [4])}")


def check_counted(problems):
    """The phrase reaches core asking for more than one detection, in core's own syntax."""
    from comfy.text_encoders.sam3_clip import _parse_prompts  # core's parser: the independent answer
    for phrase, most, want in (("person", 16, [("person", 16)]), ("person:3", 16, [("person", 3)]),
                               ("lead singer, drummer:2", 8, [("lead singer", 8), ("drummer", 2)])):
        got = _parse_prompts(st.counted(phrase, most))
        if got != want:
            problems.append(f"counted({phrase!r}, {most}) is read by core as {got}, not {want}")
    if _parse_prompts("person") != [("person", 1)]:
        problems.append("the control failed: core no longer reads a bare phrase as one detection, so `counted` "
                        "may not be needed and this item is not testing what it says")
    try:
        st.counted(" , ", 4)
    except ValueError:
        pass
    else:
        problems.append("an empty phrase was accepted")


def check_choose(problems):
    big, centre, corner = _box(0, 0, 60, 40), _box(56, 30, 72, 42), _box(110, 0, 126, 10)
    masks = torch.stack([corner, big, centre], dim=0)
    scores = [0.9, 0.6, 0.7]
    for pick, want in ((st.PICK_LARGEST, 1), (st.PICK_CENTRAL, 2), (st.PICK_SCORE, 0)):
        got = st.choose(masks, scores, pick)
        if got != want:
            problems.append(f"choose with `{pick}` took detection {got}; it is {want}")
    if st.choose(torch.zeros((0, H, W)), [], st.PICK_LARGEST) is not None:
        problems.append("choose picked something out of no detections")
    try:
        st.choose(masks, scores, "tallest")
    except ValueError:
        pass
    else:
        problems.append("choose accepted an unknown rule")


def _world():
    """Four shots of 8 frames. Person 0 is the subject; 1 and 2 are others.

    shot 0 (frames 0-7): 0 and 1.   shot 1 (8-15): 1 and 2 only.
    shot 2 (16-23): 0 enters at frame 20, with 2.   shot 3 (24-31): 0 alone.
    Each person has a fixed signature and a box that says who they are.
    """
    sigs = {0: torch.tensor([1.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0, 0.0]), 2: torch.tensor([0.0, 0.0, 1.0])}
    boxes = {0: _box(40, 10, 80, 60), 1: _box(0, 20, 20, 50), 2: _box(100, 20, 120, 50)}
    def present(f: int) -> list[int]:
        if f < 8: return [1, 0]
        if f < 16: return [1, 2]
        if f < 24: return [2] + ([0] if f >= 20 else [])
        return [0]
    def who(mask: torch.Tensor) -> int:
        return next(p for p, b in boxes.items() if torch.equal(b, mask))
    calls = {"detect": [], "track": []}
    def detect(f: int):
        calls["detect"].append(f)
        people = present(f)
        if not people:
            return torch.zeros((0, H, W)), []
        return torch.stack([boxes[p] for p in people], dim=0), [0.9 - 0.1 * i for i in range(len(people))]
    def sign(_frame: int, mask: torch.Tensor):
        return sigs[who(mask)]
    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        calls["track"].append((start, end, seed, who(mask)))
        return mask[None].repeat(end - start, 1, 1)
    return boxes, detect, sign, track, calls


def check_follow(problems):
    boxes, detect, sign, track, calls = _world()
    cuts = [8, 16, 24]
    # a frame named, a value named; each shot is first judged `offset` frames in
    got = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    shots = got.shots
    if got.pick_frame != 3 or got.others != 1 or abs(got.match - 0.8) > 1e-9:
        problems.append(f"named frame 3 and value 0.8 gave pick_frame {got.pick_frame}, {got.others} other(s), cut {got.match}")
    want = [(0, 8, 3, 0), (16, 24, 20, 0), (24, 32, 25, 0)]
    if calls["track"] != want:
        problems.append(f"tracked {calls['track']}; the subject's shots, seeds and identity are {want}")
    if [s.seed for s in shots] != [3, None, 20, 25] or [s.probe for s in shots] != [1, 9, 17, 25]:
        problems.append(f"seeds {[s.seed for s in shots]} and probe frames {[s.probe for s in shots]}; shot 2 holds only "
                        "other people, shot 3's subject enters at frame 20, and each shot is first judged one frame in")
    # relative to the other person on the pick frame, that person scores 0 and a stranger 0.5: neither is the subject
    if not abs(shots[1].best - 0.5) < 1e-4:
        problems.append(f"the shot holding the two others has best similarity {shots[1].best:.3f}, not 0.5")
    mask = st.assemble(32, H, W, got.pieces)
    if tuple(mask.shape) != (32, H, W):
        problems.append(f"the mask is {tuple(mask.shape)}, not one per frame at the frames' size")
    if float(mask[8:16].sum()) != 0.0:
        problems.append("the shot the subject is not in has a mask")
    for f in (0, 7, 16, 23, 31):
        if not torch.equal(mask[f], boxes[0]):
            problems.append(f"frame {f} does not carry the subject's mask")
            break
    text = st.report(got, cuts, st.PICK_LARGEST, "person", True, True, 1.0,
                     cutting=st.cuts_line([0.99, 0.95, 0.4], 0.9, True))
    for need in ("cuts at frame(s) [8, 16, 24]", "relative to the 1 other", "absent (best similarity 0.50", "taken on frame 20",
                 "the value named, 0.80", "1.00 1.00 | 0.50", "cut threshold: the value named, 0.90"):
        if need not in text:
            problems.append(f"the report lacks {need!r}: {text!r}")
    tiles = st.preview(torch.rand((32, H, W, 3)), mask, shots, detect)
    if tiles.shape[0] != 4 or tiles.shape[2] != st.TILE_WIDTH or tiles.shape[3] != 3:
        problems.append(f"the preview is {tuple(tiles.shape)}, not one tile per shot at the tile width")
    if not (float(tiles.min()) >= 0.0 and float(tiles.max()) <= 1.0):
        problems.append("the preview leaves 0..1")


def check_automatic(problems):
    """Nothing named: the pick is the person the rule favours for most of the clip, the cut comes from the scores."""
    boxes, detect, sign, track, calls = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, None, None, detect, sign, track, stride=4, offset=1)
    # the largest person on each shot's probe frame: 0, 1 or 2 (a tie in size, the first), 2, 0. Person 0 wins the most frames
    if got.pick_frame != 1 or got.others != 1:
        problems.append(f"automatic pick took frame {got.pick_frame} with {got.others} other(s); person 0 is the largest in "
                        "shots 1 and 4, and of those frames the one showing the most people is frame 1")
    if [s.seed for s in got.shots] != [1, None, 20, 25]:
        problems.append(f"automatic: seeds {[s.seed for s in got.shots]}; the subject is in shots 1, 3 and 4")
    if not abs(got.match - st.MATCH_FLOOR) < 1e-9 and not (0.5 < got.match <= 1.0):
        problems.append(f"automatic cut is {got.match}")
    text = st.report(got, [8, 16, 24], st.PICK_LARGEST, "person", False, False, 1.0)
    if "chosen automatically" not in text or "match: automatic" not in text:
        problems.append(f"the report does not say the pick and the match were automatic: {text!r}")
    # the cut rule alone
    floor = st.MATCH_FLOOR
    for scores, want, why in (
            ([0.94, 0.92, 0.71, 0.71, 0.66, 0.51], (0.92 + 0.71) / 2, "a clear gap above the floor moves the cut to its middle"),
            ([0.94, 0.92, 0.90, 0.85], floor, "the subject in every shot: no clear gap, so everything above the floor"),
            ([0.71, 0.71, 0.66, 0.51, 0.46], floor, "the subject in no other shot: a gap among wrong people is below the floor"),
            ([0.93], floor, "one other shot"),
            ([], floor, "no other shot"),
            ([0.95, 0.83, 0.40], floor, "the widest gap decides, and here it lies below both: the two above the floor are kept"),
            ([0.95, 0.83, 0.78], (0.95 + 0.83) / 2, "the widest gap lies above the floor, so the cut moves up into it"),
            ([0.82, 0.40], floor, "a gap whose middle is under the floor does not lower the cut")):
        got_cut = st.auto_match(scores, floor)
        if abs(got_cut - want) > 1e-9:
            problems.append(f"auto_match({scores}) is {got_cut:.3f}, not {want:.3f}: {why}")
    # picked where the subject is alone: nothing to subtract, the plain floor applies
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 26, None, detect, sign, track, stride=4, offset=1)
    if got.others != 0 or abs(got.match - st.PLAIN_FLOOR) > 1e-9 or [s.seed for s in got.shots] != [1, None, 20, 26]:
        problems.append(f"picked on a frame with nobody else: {got.others} other(s), cut {got.match}, seeds "
                        f"{[s.seed for s in got.shots]}; the plain floor applies and the subject is in shots 1, 3 and 4")
    if "plain similarity" not in st.report(got, [8, 16, 24], st.PICK_LARGEST, "person", True, False, 1.0):
        problems.append("the report does not say that the plain similarity was used")


def _solo_world():
    """One person, framed differently from shot to shot, and two things the detector takes for a person.

    Four shots of 8 frames. Shot 1: a microphone alone on frames 0-3, with the person on 4-5, the person alone
    on 6-7. Shot 2: the person, full length. Shot 3: the person in close-up. Shot 4: a lamp and nobody.
    The person in close-up scores 0.75 against the full-length pick, the microphone 0.70, the lamp 0.3.
    Each signature is two views, and only the person has the second, the head.
    """
    def unit(c: float, axis: int) -> torch.Tensor:
        v = torch.zeros(4); v[0] = c; v[axis] = (1 - c * c) ** 0.5
        return v
    full, close = _box(50, 5, 70, 65), _box(20, 5, 100, 70)
    mic, lamp = _box(0, 0, 10, 30), _box(110, 0, 125, 30)
    sigs = [(full, torch.tensor([1.0, 0, 0, 0])), (close, unit(0.75, 1)), (mic, unit(0.70, 2)), (lamp, unit(0.3, 3))]
    def present(f: int) -> list[torch.Tensor]:
        if f < 4: return [mic]
        if f < 6: return [mic, close]
        if f < 8: return [close]
        if f < 16: return [full]
        if f < 24: return [close]
        return [lamp]
    tracked = []
    def detect(f: int):
        return torch.stack(present(f), dim=0), [0.9] * len(present(f))
    def sign(_frame: int, mask: torch.Tensor):
        v = next(v for m, v in sigs if torch.equal(m, mask))
        return v, (v if torch.equal(mask, close) or torch.equal(mask, full) else None)
    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        tracked.append((start, end, seed, "person" if torch.equal(mask, close) or torch.equal(mask, full) else "thing"))
        return mask[None].repeat(end - start, 1, 1)
    return detect, sign, track, tracked


def check_alone(problems):
    """A clip with one person: taken in every shot they are in, whatever the framing, and a thing is not."""
    detect, sign, track, tracked = _solo_world()
    cuts = [8, 16, 24]
    got = st.follow(32, cuts, st.PICK_LARGEST, 9, None, detect, sign, track, stride=2, offset=1)
    want = [(0, 8, 4, "person"), (8, 16, 9, "person"), (16, 24, 17, "person")]
    if tracked != want:
        problems.append(f"one person, framed differently per shot: tracked {tracked}, not {want}. Shot 1 opens on a "
                        "microphone alone and then shows it beside the person: a thing has no head, so the person is the "
                        "one with a head from frame 4; shot 4 holds a lamp, which scores under the line and has no head")
    if [s.lone for s in got.shots] != [True, False, True, False] or got.others != 0:
        problems.append(f"lone flags {[s.lone for s in got.shots]} with {got.others} other(s) on the pick frame")
    if not abs(got.match - st.PLAIN_FLOOR) < 1e-9:
        problems.append(f"the line is {got.match}; with nobody else on the pick frame it is the plain floor {st.PLAIN_FLOOR}")
    if got.views != 2 or got.views_used != 2:
        problems.append(f"the subject is compared in {got.views_used} of {got.views} places; this world gives two and the person has both")
    text = st.report(got, cuts, st.PICK_LARGEST, "person", True, False, 1.0)
    for need in ("taken on frame 4, as the only person there, similarity 0.75", "[4] frames 24-31: absent (up to 1 detection(s), none that could be compared)",
                 "framed differently, the mask is 4.0 times as wide", "one person with a head is taken", "the lower one counts"):
        if need not in text:
            problems.append(f"the one-person report lacks {need!r}: {text!r}")
    if st._state(got.shots[0]) != "taken (only person)":
        problems.append("the tile does not say a shot was taken as the only person")
    # the control: with a value named the rule is off, and the same clip loses the two shots framed differently
    detect, sign, track, tracked = _solo_world()
    got = st.follow(32, cuts, st.PICK_LARGEST, 9, st.PLAIN_FLOOR, detect, sign, track, stride=2, offset=1)
    if [s.seed for s in got.shots] != [None, 9, None, None] or any(s.lone for s in got.shots):
        problems.append(f"with a value named, seeds are {[s.seed for s in got.shots]}: the lone rule must be off, and without "
                        "it this clip's two other shots are missed, which is the failure the rule exists for")
    # not alone on the pick frame: a lone person in another shot is not presumed to be the subject
    boxes, detect, sign, track, calls = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, None, detect, sign, track, stride=4, offset=1)
    if any(s.lone for s in got.shots) or [s.seed for s in got.shots] != [3, None, 20, 25]:
        problems.append(f"picked among other people, seeds {[s.seed for s in got.shots]} and lone {[s.lone for s in got.shots]}: "
                        "the lone rule applies only when the pick frame shows nobody else")


def check_two_places(problems):
    """A match has to hold on the head as well: someone alike at the shoulders and not at the head is left alone."""
    subject, twin, other = _box(40, 10, 80, 60), _box(0, 20, 20, 50), _box(100, 20, 120, 50)
    e = lambda *v: torch.tensor(v, dtype=torch.float32) / torch.tensor(v, dtype=torch.float32).norm()
    # shoulders: the twin is the subject's double. head: the twin is somebody else
    shoulders = [(subject, e(1.0, 0, 0)), (twin, e(1.0, 0.05, 0)), (other, e(0, 1.0, 0))]
    heads = [(subject, e(1.0, 0, 0)), (twin, e(0, 0, 1.0)), (other, e(0, 1.0, 0))]
    def detect(f: int):
        people = [subject, other] if f < 8 else ([twin, other] if f < 16 else [subject, other])
        return torch.stack(people, dim=0), [0.9, 0.8]
    def find(table, mask):
        return next(v for m, v in table if torch.equal(m, mask))
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)
    both = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, lambda _f, m: (find(shoulders, m), find(heads, m)),
                     track, stride=4, offset=1)
    if [s.seed for s in both.shots] != [2, None, 17]:
        problems.append(f"compared in two places, seeds are {[s.seed for s in both.shots]}: shot 2 holds the subject's double "
                        "at the shoulders with another head, and must be left alone; shot 3 holds the subject")
    one = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, lambda _f, m: find(shoulders, m), track, stride=4, offset=1)
    if [s.seed for s in one.shots] != [2, 9, 17]:
        problems.append(f"the control failed: compared at the shoulders alone, seeds are {[s.seed for s in one.shots]}; the "
                        "double should be taken there, or the second place decides nothing in this case")
    # a person on whom no head is found is no match, however alike at the shoulders
    none = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect,
                     lambda f, m: (find(shoulders, m), None if 16 <= f else find(heads, m)), track, stride=4, offset=1)
    if [s.seed for s in none.shots] != [2, None, None]:
        problems.append(f"with no head found in shot 3, seeds are {[s.seed for s in none.shots]}: a person without a head is no match")
    # the subject has no head on the pick frame: the shoulders decide alone, and the report says so
    bare = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, lambda _f, m: (find(shoulders, m), None), track, stride=4, offset=1)
    if [s.seed for s in bare.shots] != [2, 9, 17] or bare.views_used != 1:
        problems.append(f"with no head on the subject, seeds are {[s.seed for s in bare.shots]} from {bare.views_used} place(s): "
                        "the shoulders decide alone")
    if "matched by the head and shoulders alone" not in st.report(bare, [8, 16], st.PICK_LARGEST, "person", True, False, 1.0):
        problems.append("the report does not say that no head was found on the subject")
    # head_of: the head mostly inside the person, the highest of them, and none for a thing
    person = _box(40, 10, 80, 60)
    hat, face, far = _box(50, 10, 70, 20), _box(50, 22, 70, 34), _box(0, 0, 20, 10)
    got = st.head_of(person, torch.stack([far, face, hat], dim=0))
    if got is None or not torch.equal(got, hat):
        problems.append("head_of does not return the highest head lying inside the person")
    if st.head_of(person, torch.stack([far], dim=0)) is not None or st.head_of(person, torch.zeros((0, H, W))) is not None:
        problems.append("head_of found a head for a person who has none inside their mask")


def check_empty(problems):
    boxes, detect, sign, track, _calls = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, 1.5, detect, sign, track, stride=4, offset=1)
    mask = st.assemble(32, H, W, got.pieces)
    if [s.seed for s in got.shots] != [3, None, None, None] or float(mask[8:].sum()) != 0.0:
        problems.append("with a threshold nothing can pass, a shot other than the picked one was still filled")
    if not torch.equal(mask[0], boxes[0]):
        problems.append("the picked shot lost its mask when the threshold was raised")
    none = lambda _frame: (torch.zeros((0, H, W)), [])
    for frame in (3, None):
        got = st.follow(32, [8], st.PICK_LARGEST, frame, None, none, sign, track)
        if got.pick_frame is not None or got.pieces or float(st.assemble(32, H, W, got.pieces).sum()) != 0.0:
            problems.append("with nothing detected something was still masked")
        if "every mask is empty" not in st.report(got, [8], st.PICK_LARGEST, "person", frame is not None, False, 1.0):
            problems.append("the report does not say that nothing was picked")
        tiles = st.preview(torch.rand((32, H, W, 3)), st.assemble(32, H, W, got.pieces), got.shots, none)
        if tiles.shape[0] != 2:
            problems.append("the preview of a clip with no subject is not one tile per shot")
    try:
        st.follow(32, [], st.PICK_LARGEST, 40, 0.5, detect, sign, track)
    except ValueError:
        pass
    else:
        problems.append("a pick_frame past the end of the clip was accepted")


def check_signature(problems):
    """The signature is taken from the head and shoulders, and tells two heads apart over the same clothes."""
    body = _box(40, 12, 80, 72)
    head = st.top_third(body)
    if not (float(head[12:32].sum()) == float(head.sum()) > 0 and float(head[32:].sum()) == 0):
        problems.append("top_third does not keep exactly the top third of the rows a mask covers")
    if not torch.equal(st.top_third(torch.zeros((H, W))), torch.zeros((H, W))):
        problems.append("top_third changed an empty mask")
    feats_a, feats_b = torch.zeros((3, H, W)), torch.zeros((3, H, W))
    feats_a[0, 12:32], feats_b[1, 12:32] = 1.0, 1.0      # two different heads
    feats_a[2, 32:], feats_b[2, 32:] = 1.0, 1.0          # the same clothes
    v = torch.tensor([0.6, 0.8, 0.0])
    r = st.relative(v, torch.tensor([0.6, 0.0, 0.0]))
    if r is None or not torch.allclose(r, torch.tensor([0.0, 1.0, 0.0])) or st.relative(v, None) is not v:
        problems.append("relative does not subtract the centre and return unit length, or changes a signature with no centre")
    whole = st.similarity(st.signature(feats_a, body), st.signature(feats_b, body))
    heads = st.similarity(st.signature(feats_a, head), st.signature(feats_b, head))
    if not (whole > 0.5 and heads < 0.05):
        problems.append(f"two people in the same clothes: similarity {whole:.2f} on the whole mask and {heads:.2f} on "
                        "the top third; the top third is meant to tell them apart where the whole mask cannot")
    if st.signature(feats_a, torch.zeros((H, W))) is not None or st.similarity(None, None) != -1.0:
        problems.append("an empty mask has a signature, or a missing signature has a similarity")


def check_schema(problems):
    schema = st.MiniMaxH3SubjectTrack.define_schema()
    inputs = {i.id: i for i in schema.inputs}
    for name, default in (("subject_phrase", st.SUBJECT_PHRASE), ("head_phrase", st.HEAD_PHRASE),
                          ("max_people", st.MAX_PEOPLE), ("detection_threshold", st.DETECTION_THRESHOLD),
                          ("pick", st.PICK_LARGEST)):
        if name not in inputs:
            problems.append(f"the node has no `{name}` input: what SAM is asked and how it is judged must be visible")
        elif getattr(inputs[name], "default", None) != default:
            problems.append(f"`{name}` defaults to {getattr(inputs[name], 'default', None)!r}, not the module's {default!r}")
    for name in ("frames", "segmenter", "segmenter_clip"):
        if name not in inputs:
            problems.append(f"the node has no `{name}` input")
    # the two choices: automatic first, and the named value under its own option
    for name, other, nested, default in (("pick_on", st.PICK_ON_FRAME, "pick_frame", 0),
                                         ("match", st.AT_VALUE, "match_threshold", st.MATCH_THRESHOLD),
                                         ("cuts", st.AT_VALUE, "cut_threshold", st.CUT_THRESHOLD)):
        combo = inputs.get(name)
        if combo is None:
            problems.append(f"the node has no `{name}` choice")
            continue
        options = {o.key: o for o in combo.options}
        if list(options) != [st.AUTOMATIC, other]:
            problems.append(f"`{name}` offers {list(options)}, not automatic first and then `{other}`")
            continue
        under = {i.id: i for i in options[other].inputs}
        if list(under) != [nested] or getattr(under[nested], "default", None) != default:
            problems.append(f"`{name}` -> `{other}` does not reveal `{nested}` at {default!r}")
        if options[st.AUTOMATIC].inputs:
            problems.append(f"`{name}` -> automatic asks for something")
    for flat in ("pick_frame", "match_threshold", "cut_threshold"):
        if flat in inputs:
            problems.append(f"`{flat}` is also a top-level input: two inputs for one thing")
    if len(schema.outputs) != 3 or schema.outputs[0].io_type != "MASK":
        problems.append("the node's outputs are not mask, preview, report with the mask first")
    if getattr(schema, "is_output_node", False):
        problems.append("the node is an output node: core would run the tracker on every queue, kept mask or not")
    if not isinstance(getattr(st.MiniMaxH3SubjectTrack, "MASK_VERSION", None), int):
        problems.append("the node declares no integer MASK_VERSION, so a kept mask would survive a change to how it is made")
    sel = st._selection({"match": st.AT_VALUE, "match_threshold": 0.5}, "match", "match_threshold")
    if sel != (st.AT_VALUE, 0.5) or st._selection(st.AUTOMATIC, "match", "match_threshold") != (st.AUTOMATIC, None):
        problems.append("a DynamicCombo's nested dict, or a bare selection, is not read as the choice and its value")


def main() -> int:
    problems: list[str] = []
    for check in (check_cuts, check_ranges, check_counted, check_choose, check_signature, check_follow, check_automatic,
                  check_alone, check_two_places, check_empty, check_schema):
        check(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("ok    the subject track finds a cut and not a lighting change, covers every frame once, picks by the "
              "rule, follows the subject and nobody else across shots, leaves absent shots empty, and declares "
              "what it asks SAM as inputs")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
