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
   the picked shot is tracked both ways from the pick frame.
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
REPO = HERE.parent
COMFY = REPO.parent.parent
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
    spec = importlib.util.spec_from_file_location("_h3pack.subject_track", REPO / "subject_track.py")
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


def check_ranges(problems):
    for n, cuts in ((10, []), (10, [4]), (10, [0, 4, 4, 10, 12]), (1, [])):
        ranges = st.shot_ranges(n, cuts)
        covered = [f for s, e in ranges for f in range(s, e)]
        if covered != list(range(n)):
            problems.append(f"shot_ranges({n}, {cuts}) = {ranges} does not cover each frame once")
    if st.shot_ranges(10, [4]) != [(0, 4), (4, 10)]:
        problems.append(f"shot_ranges(10, [4]) = {st.shot_ranges(10, [4])}")


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
    sigs = {0: torch.tensor([1.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0, 0.0]), 2: torch.tensor([0.6, 0.0, 0.8])}
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
    pieces, shots, picked = st.follow(32, cuts, 3, st.PICK_LARGEST, 0.8, detect, sign, track, stride=4)
    if picked is None or not torch.equal(picked, boxes[0]):
        problems.append("the largest person on the pick frame was not the one picked")
        return
    want = [(0, 8, 3, 0), (16, 24, 20, 0), (24, 32, 24, 0)]
    if calls["track"] != want:
        problems.append(f"tracked {calls['track']}; the subject's shots, seeds and identity are {want}")
    if [s.seed for s in shots] != [3, None, 20, 24]:
        problems.append(f"seeds per shot are {[s.seed for s in shots]}; shot 2 holds only other people")
    if not shots[1].best < 0.8 <= shots[2].best:
        problems.append(f"best similarity per shot is {[round(s.best, 2) for s in shots]}; the empty shot's must be "
                        "below the threshold and the found one's at or above it")
    mask = st.assemble(32, H, W, pieces)
    if tuple(mask.shape) != (32, H, W):
        problems.append(f"the mask is {tuple(mask.shape)}, not one per frame at the frames' size")
    if float(mask[8:16].sum()) != 0.0:
        problems.append("the shot the subject is not in has a mask")
    for f in (0, 7, 16, 23, 31):
        if not torch.equal(mask[f], boxes[0]):
            problems.append(f"frame {f} does not carry the subject's mask")
            break
    text = st.report(shots, cuts, 3, st.PICK_LARGEST, "person", True, 0.8, 1.0)
    if "absent" not in text or "found on frame 20" not in text or "cuts at frame(s) [8, 16, 24]" not in text:
        problems.append(f"the report does not say what happened: {text!r}")
    tiles = st.preview(torch.rand((32, H, W, 3)), mask, shots)
    if tuple(tiles.shape) != (4, H, W, 3):
        problems.append(f"the preview is {tuple(tiles.shape)}, not one frame per shot")


def check_empty(problems):
    boxes, detect, sign, track, _calls = _world()
    pieces, shots, picked = st.follow(32, [8, 16, 24], 3, st.PICK_LARGEST, 1.5, detect, sign, track, stride=4)
    mask = st.assemble(32, H, W, pieces)
    if [s.seed for s in shots] != [3, None, None, None] or float(mask[8:].sum()) != 0.0:
        problems.append("with a threshold nothing can pass, a shot other than the picked one was still filled")
    if not torch.equal(mask[0], boxes[0]):
        problems.append("the picked shot lost its mask when the threshold was raised")
    none = lambda _frame: (torch.zeros((0, H, W)), [])
    pieces, shots, picked = st.follow(32, [8], 3, st.PICK_LARGEST, 0.5, none, sign, track)
    if picked is not None or pieces or float(st.assemble(32, H, W, pieces).sum()) != 0.0:
        problems.append("with nothing detected on the pick frame something was still masked")
    if "every mask is empty" not in st.report(shots, [8], 3, st.PICK_LARGEST, "person", False, 0.5, 1.0):
        problems.append("the report does not say that nothing was picked")
    try:
        st.follow(32, [], 40, st.PICK_LARGEST, 0.5, detect, sign, track)
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
    for name, default in (("subject_phrase", st.SUBJECT_PHRASE), ("cut_threshold", st.CUT_THRESHOLD),
                          ("match_threshold", st.MATCH_THRESHOLD),
                          ("detection_threshold", st.DETECTION_THRESHOLD), ("pick", st.PICK_LARGEST)):
        if name not in inputs:
            problems.append(f"the node has no `{name}` input: what SAM is asked and how it is judged must be visible")
        elif getattr(inputs[name], "default", None) != default:
            problems.append(f"`{name}` defaults to {getattr(inputs[name], 'default', None)!r}, not the module's {default!r}")
    for name in ("pick_frame", "frames", "segmenter", "segmenter_clip"):
        if name not in inputs:
            problems.append(f"the node has no `{name}` input")
    if len(schema.outputs) != 3 or schema.outputs[0].io_type != "MASK":
        problems.append("the node's outputs are not mask, preview, report with the mask first")
    if getattr(schema, "is_output_node", False):
        problems.append("the node is an output node: core would run the tracker on every queue, kept mask or not")
    if not isinstance(getattr(st.MiniMaxH3SubjectTrack, "MASK_VERSION", None), int):
        problems.append("the node declares no integer MASK_VERSION, so a kept mask would survive a change to how it is made")


def main() -> int:
    problems: list[str] = []
    for check in (check_cuts, check_ranges, check_choose, check_signature, check_follow, check_empty, check_schema):
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
