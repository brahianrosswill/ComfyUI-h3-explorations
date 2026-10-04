"""One person, followed through a clip with cuts, as one mask per frame.

The owner's case (2026-10-04): a music video with several people, cutaways and
coloured lighting, in which one performer is to be replaced. Core's
`SAM3_VideoTrack` with a text prompt is not built for it. SAM 3's video data
was checked to have no scene cuts (its paper, the verification step of the
video annotation), so across a cut the tracker starts new objects: on the
owner's band clip the lead came out as three hand-picked object numbers. With
a text prompt it also has an object cap that detection stops at for good, a
keep-alive count that blanks a track the detector stops matching, and a
detector mask that overwrites the tracked one on a confident frame
(`comfy/ldm/sam3/tracker.py::track_video_with_detection`,
`_match_and_add_detections`). `docs/research/masking/2026-10-04_mrhf.md` has
the reading.

Upstream SAM 3 added a caller-mask request for this on 2026-09-18
(`coderef/sam3`, "Condition the video tracker on a caller mask"): tell the
tracker which object is meant, from a mask, at any frame. It is not served by
the multiplex model the 3.1 checkpoint is, and core's port does not have it.
This node does the same thing with what core has: it splits the clip into
shots, picks the subject once, finds the same person on each other shot, and
tracks every shot from that person's mask with no text prompt, which is
core's `initial_mask` path and has none of the three behaviours above.

**The steps** (`follow`):

1. Cuts (`cut_scores`, `find_cuts`, `auto_cuts`): one minus the correlation
   of consecutive frames' gradient maps at a small size. A cut changes where
   the edges are; a lighting change does not. ffmpeg's scene score failed on
   the band clip under its coloured gels; this score separated eight of its
   nine cuts from every other frame (the ninth is a jump cut inside a
   cutaway). The threshold is a value the user names, or automatic: the
   middle of the widest gap in the clip's own scores, since two clips already
   disagreed about one fixed value.
2. People: each shot is looked at on one frame a little way in
   (`PROBE_OFFSET`; the first frame after a cut is where blur and a dissolve
   land). SAM 3 is asked for `subject_phrase` there. Core's detector returns
   one detection per phrase unless the phrase carries a count
   (`comfy/text_encoders/sam3_clip.py::_parse_prompts`, `person:8`), so the
   node asks for up to `max_people` (`counted`).
3. The pick (`choose`, `main_subject`): `pick` names a rule, the largest, the
   most central or the best match for the phrase. Named a frame, the node
   applies the rule there. Left automatic, it applies the rule on every
   shot's frame and takes the person the rule favours for the most frames of
   the clip: a lead is the largest person in the long shots, and a cutaway's
   largest person is the largest only there.
4. Each other shot: its people are compared with the subject by the vision
   trunk's features pooled under the top third of each mask, the head and
   shoulders (`top_third`, `signature`). The other people on the pick frame
   are certainly not the subject, so their average signature is subtracted
   from every signature before comparing (`relative`, `similarity`). A shot
   whose best is at or above the cut is seeded there; one below it is probed
   every `PROBE_STRIDE` frames, so a subject who walks in late or turns round
   late is still found; a shot with nobody above it is left empty.
5. The cut between "the subject" and "somebody else" (`auto_match`): a value the
   user names, or automatic. Automatic takes everything at or above a floor,
   and moves the cut up to the middle of the widest gap in the shots' best
   scores when that gap lies above the floor: the widest gap is taken as the
   line between the subject and everybody else. The report prints the scores
   with the cut marked, so a wrong cut can be seen.
6. Two places, not one. SAM 3 is also asked for `head_phrase` on every frame
   that is looked at, and each person gets the head that lies inside their
   mask (`head_of`). A person is compared with the subject under the top
   third of the mask and under the head, and the lower of the two counts. A
   person on whom no head is found is no match. Measured on three clips
   (`docs/research/masking/2026-10-04_mrhf.md`): on a clip of young women in
   a car the top third alone took two other women as the lead, and the head
   alone took a wrong person on the band clip; the lower of the two took the
   right shots on both.
7. A clip with one person in it. When the pick frame shows nobody else and
   the match is automatic, there is nobody to mistake the subject for, and
   the similarity is the wrong judge: it fell to 0.73 for the same singer
   between a full-length shot and a close-up (mrblue's card run,
   2026-10-04). So a shot under the line is probed to its end, and its best
   frame showing exactly one person with a head is taken, whatever it scores.
   The head is what keeps a thing out: the detector marks a hanging
   microphone as a person in that clip's empty opening, and on every frame it
   was the one detection with no head. The report and the tile say the shot
   was taken this way. A cutaway to a different lone person would be taken
   too; naming a value for `match` turns the rule off.
8. Tracking: from the seed frame forward to the shot's end and backward to its
   start, each a tracker call with the seed mask as `initial_mask`.

**How far the matching can be trusted.** SAM 3's trunk is trained to say what
a thing is, not who, so people in the same clothes score close together. On
the owner's band clip (six people in one sweatshirt,
`docs/research/masking/2026-10-04_mrhf.md`): pooled under the whole mask a
neighbour scores as close to the pick as the lead in another shot does; under
the top third with the pick frame's other people subtracted, on the card, the
lead scored 0.92 and 0.94 on his other two shots and the best wrong person
0.71. Seen from behind he scores inside the others' range, so a shot in which
he never faces the camera is not found, and the report and the preview show
it as absent with its best value. One clip. Every value that decides the mask
is an input or is stated in the report.

**Keeping the mask** is not this node's job. `MiniMaxH3MaskedSource` keeps a
finished mask across runs (`mask_store.py`) and on a hit never asks for its
`mask` input, so core does not run this node at all. Two things here serve
that. `MASK_VERSION` on the node goes into the kept mask's key: bump it when
a change would give a different mask from the same inputs and settings (the
cut score, the pick, the signature, the automatic rules, how a shot is
seeded and tracked), and
not for a tooltip or the report's wording. And the preview and the report are
shown as the node's own UI, so nothing has to be wired to see them: a preview
or save node on either output would make core run the tracker on every queue,
whatever the Masked Source decided. The outputs exist; a shipped graph leaves
them unwired.

Every phrase SAM 3 is given is an input: `subject_phrase` and `head_phrase`.

Nothing here patches core. It calls core's own nodes (`SAM3_Detect`,
`SAM3_VideoTrack`, `SAM3_TrackToMask`) and the SAM 3 model's vision trunk.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

#: The size frames are reduced to for the cut score, width by height. Small on
#: purpose: a cut moves every edge, and grain or a gesture should not count.
#: Reasoned, not measured.
CUT_SIZE = (192, 108)
#: The default of `cut_threshold`, used when `cuts` is `at a value`, and what
#: automatic falls back to when the clip's scores show no clear gap. Reasoned
#: from two clips, neither of which it suits as a fixed value: see `CUT_FLOOR`.
CUT_THRESHOLD = 0.9
#: Automatic cuts: only a gap whose upper side is at or above this is taken as
#: the line between cuts and everything else. Measured on two clips
#: (`docs/research/masking/2026-10-04_mrhf.md`): the band segment's cuts score
#: 0.99 and above and its other frames 0.76 and below; the one-person clip's
#: cuts 0.84 and above and its other frames 0.46 and below.
CUT_FLOOR = 0.8
#: Automatic cuts: the gap has to be at least this wide. Reasoned.
CUT_MIN_GAP = 0.1
#: How far into a shot the frame it is judged on lies. Reasoned: past the
#: frame a cut detector off by one, a dissolve or motion blur lands on.
PROBE_OFFSET = 4
#: Frames between probes of a shot whose first look found no match.
#: Reasoned: half a second at the pack's frame rate.
PROBE_STRIDE = 12
#: Automatic matching, similarity relative to the pick frame's other people:
#: nothing below this is the subject. Measured on one clip on the card, the
#: band segment with five others on the pick frame: the lead 0.92 and 0.94,
#: the best wrong person 0.71 (`docs/research/masking/2026-10-04_mrhf.md`).
MATCH_FLOOR = 0.8
#: The same floor on the plain similarity, used when the pick frame shows
#: nobody else. Measured on the same clip by a CPU probe: the lead facing the
#: camera 0.945 and above, everyone else 0.89 and below.
PLAIN_FLOOR = 0.91
#: Automatic matching: a gap in the shots' best scores at least this wide,
#: above the floor, moves the cut to its middle. Reasoned: wider than the
#: spread among true matches seen so far and narrower than the gap to the
#: wrong people.
MIN_GAP = 0.1
#: The default of `match_threshold`, used when `match` is `at a value`. The
#: middle of the gap measured on the band clip on the card.
MATCH_THRESHOLD = 0.82
#: Automatic pick: two shots' favoured people are the same person at or above
#: this plain similarity. Measured by the CPU probe above. It only decides a
#: vote, so being a little off costs little.
PLAIN_SAME = 0.93
#: The default of `detection_threshold`. Inherited: core's node default
#: (`comfy_extras/nodes_sam3.py::SAM3_Detect.define_schema`).
DETECTION_THRESHOLD = 0.5
#: The default of `subject_phrase`.
SUBJECT_PHRASE = "person"
#: The default of `head_phrase`: what SAM 3 is asked for to find each
#: person's head, the second place a match has to hold.
HEAD_PHRASE = "head"
#: The default of `max_people`: the most detections of the phrase taken from a
#: frame. Reasoned: more than a stage usually shows, and each one costs a mask
#: refinement only on the few frames that are probed.
MAX_PEOPLE = 16
#: Width of a preview tile in pixels. Reasoned: wide enough to read the label.
TILE_WIDTH = 768
#: The side SAM 3's vision trunk takes. Inherited: core's detect node scales
#: every frame to it (`comfy_extras/nodes_sam3.py::SAM3_Detect.execute`).
TRUNK_SIDE = 1008

PICK_LARGEST = "largest"
PICK_CENTRAL = "most central"
PICK_SCORE = "best match for the phrase"
PICKS = (PICK_LARGEST, PICK_CENTRAL, PICK_SCORE)

AUTOMATIC = "automatic"
PICK_ON_FRAME = "a frame I name"
AT_VALUE = "at a value"


def counted(phrase: str, most: int) -> str:
    """`phrase` in core's SAM 3 prompt syntax, each comma-separated part asking for up to `most` detections.

    A part that already carries its own `:N` keeps it. Core reads `name:N` as
    at most N detections of `name`, and a bare `name` as one.
    """
    parts = [p.strip() for p in str(phrase).split(",") if p.strip()]
    if not parts:
        raise ValueError("subject_phrase is empty: say what SAM 3 should look for, for instance `person`")
    return ", ".join(p if re.match(r"^.+?\s*:\s*[\d.]+\s*$", p) else f"{p}:{max(int(most), 1)}" for p in parts)


def cut_scores(frames: torch.Tensor) -> torch.Tensor:
    """[F, H, W, C] frames to [F - 1] scores; entry i is the step into frame i + 1.

    One minus the correlation of the two frames' gradient-magnitude maps at
    `CUT_SIZE`. About 0 for a held shot, well under 1 for a lighting change or
    ordinary movement, about 1 across a cut.
    """
    w, h = CUT_SIZE
    out, prev = [], None
    for i in range(0, frames.shape[0], 64):
        grey = frames[i:i + 64, ..., :3].to(torch.float32).mean(dim=-1, keepdim=True).movedim(-1, 1)
        small = F.interpolate(grey, size=(h, w), mode="area")[:, 0]
        edges = (small[:, 1:, 1:] - small[:, 1:, :-1]).abs() + (small[:, 1:, 1:] - small[:, :-1, 1:]).abs()
        e = edges.flatten(1)
        e = e - e.mean(dim=1, keepdim=True)
        e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-6)
        if prev is not None:
            e = torch.cat([prev, e], dim=0)
        out.append(1.0 - (e[1:] * e[:-1]).sum(dim=1))
        prev = e[-1:]
    return torch.cat(out, dim=0).cpu() if out else torch.zeros(0)


def find_cuts(scores: torch.Tensor, threshold: float) -> list[int]:
    """The frames that start a new shot: every frame whose step in scores at or above `threshold`."""
    return [int(i) + 1 for i in (scores >= float(threshold)).nonzero().flatten()]


def auto_cuts(scores) -> float:
    """The cut threshold from the clip's own step scores.

    The middle of the widest gap between neighbouring scores whose upper side
    is at or above `CUT_FLOOR`, when that gap is at least `CUT_MIN_GAP` wide:
    cuts are few and score high, everything else is many and scores lower.
    With no such gap, `CUT_THRESHOLD`.
    """
    ordered = sorted((float(v) for v in scores), reverse=True)
    widest, at = 0.0, float(CUT_THRESHOLD)
    for high, low in zip(ordered, ordered[1:]):
        if high < CUT_FLOOR:
            break
        if high - low >= CUT_MIN_GAP and high - low > widest:
            widest, at = high - low, (high + low) / 2
    return at


def cuts_line(scores, at: float, named: bool, most: int = 12) -> str:
    """The report's line about the cut threshold: the highest step scores, in order, with ` | ` at the threshold."""
    top = sorted((float(v) for v in scores), reverse=True)[:most]
    above = " ".join(f"{v:.2f}" for v in top if v >= at)
    below = " ".join(f"{v:.2f}" for v in top if v < at)
    return (f"cut threshold: {'the value named' if named else 'automatic'}, {at:.2f}; "
            f"highest steps: {above or '(none)'} | {below or '(none)'}")


def shot_ranges(n_frames: int, cuts: list[int]) -> list[tuple[int, int]]:
    """[start, end) of each shot, in order, covering every frame."""
    starts = [0] + [c for c in sorted(set(cuts)) if 0 < c < n_frames]
    return [(s, e) for s, e in zip(starts, starts[1:] + [int(n_frames)]) if e > s]


def choose(masks: torch.Tensor, scores: list[float], pick: str) -> int | None:
    """Which of the detections on the pick frame is the subject. None when there are none.

    `masks` is [N, H, W] and `scores` the detector's score for each, as core
    returns them (highest score first).
    """
    if masks.shape[0] == 0:
        return None
    if pick == PICK_SCORE:
        return int(max(range(len(scores)), key=lambda i: scores[i])) if scores else 0
    on = masks > 0.5
    if pick == PICK_LARGEST:
        return int(on.flatten(1).sum(dim=1).argmax())
    if pick == PICK_CENTRAL:
        h, w = masks.shape[-2:]
        ys = torch.arange(h, dtype=torch.float32).view(1, h, 1) / max(h - 1, 1) - 0.5
        xs = torch.arange(w, dtype=torch.float32).view(1, 1, w) / max(w - 1, 1) - 0.5
        area = on.flatten(1).sum(dim=1).clamp(min=1).to(torch.float32)
        cy = (on * ys).flatten(1).sum(dim=1) / area
        cx = (on * xs).flatten(1).sum(dim=1) / area
        return int((cx ** 2 + cy ** 2).argmin())
    raise ValueError(f"unknown pick {pick!r}; one of {list(PICKS)}")


def top_third(mask: torch.Tensor) -> torch.Tensor:
    """A [H, W] mask cut to the top third of the rows it covers: the head and shoulders of a standing person.

    People in a clip often share their clothes and never their heads, so the
    signature is taken here. A mask that covers nothing comes back unchanged.
    """
    rows = (mask > 0.5).any(dim=1).nonzero().flatten()
    if rows.numel() == 0:
        return mask
    top, bottom = int(rows.min()), int(rows.max())
    out = mask.clone()
    out[top + max(1, (bottom - top + 1) // 3):] = 0
    return out


def signature(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """The trunk's features averaged under a mask, unit length. [C, h, w] and [H, W] to [C].

    None when the mask covers no feature cell.
    """
    m = F.interpolate(mask[None, None].to(torch.float32), size=features.shape[-2:], mode="area")[0, 0]
    total = m.sum()
    if float(total) < 1e-3:
        return None
    v = (features.to(torch.float32) * m).sum(dim=(1, 2)) / total
    return v / v.norm().clamp(min=1e-6)


def relative(sig: torch.Tensor | None, centre: torch.Tensor | None) -> torch.Tensor | None:
    """A signature with `centre` subtracted, unit length again. Unchanged when there is no centre."""
    if sig is None or centre is None:
        return sig
    v = sig - centre
    return v / v.norm().clamp(min=1e-6)


def similarity(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    """Cosine similarity of two signatures; -1 when either is missing."""
    if a is None or b is None:
        return -1.0
    return float((a * b).sum())


def head_of(person: torch.Tensor, heads: torch.Tensor) -> torch.Tensor | None:
    """The head that belongs to a person: of the heads lying mostly inside the person's mask, the highest.

    `person` is [H, W] and `heads` [N, H, W]. None when no head does.
    """
    inside, best, best_top = person > 0.5, None, None
    for head in heads:
        on = head > 0.5
        area = int(on.sum())
        if area == 0 or int((on & inside).sum()) * 2 < area:
            continue
        top = int(on.any(dim=1).nonzero().min())
        if best_top is None or top < best_top:
            best, best_top = head, top
    return best


def _views(sig) -> tuple:
    """A person's signatures as a tuple: one per place they are compared. A lone signature is one view."""
    return tuple(sig) if isinstance(sig, (tuple, list)) else (sig,)


def _has_head(sig) -> bool:
    """Whether a person's last view exists: with SAM's callables that is the head."""
    return _views(sig)[-1] is not None


def _width(mask: torch.Tensor) -> int:
    """How many columns a [H, W] mask covers: a rough measure of how closely a person is framed."""
    return int((mask > 0.5).any(dim=0).sum())


def auto_match(scores: list[float], floor: float) -> float:
    """Where the subject ends and somebody else begins, from the shots' best scores.

    The floor, or the middle of the widest gap between neighbouring scores
    when that gap is at least `MIN_GAP` wide, its upper side is at or above
    the floor and its middle is too. With no such gap everything at or above
    the floor is the subject, which is right when they are in every shot.
    """
    ordered = sorted((float(s) for s in scores if s >= 0), reverse=True)
    widest, cut = 0.0, float(floor)
    for high, low in zip(ordered, ordered[1:]):
        if high >= floor and high - low >= MIN_GAP and high - low > widest:
            widest, cut = high - low, (high + low) / 2
    return max(cut, float(floor))


@dataclass
class Shot:
    """What `follow` did with one shot, for the report and the preview."""
    start: int
    end: int
    probe: int                   # the frame the shot is first judged on
    seed: int | None = None      # the frame the subject was taken on; None when absent
    best: float = -1.0           # the best similarity seen
    shown: int = 0               # the frame the preview shows: the seed, or where the best was seen
    index: int | None = None     # which detection on `shown` is the subject, or the best one when absent
    candidates: int = 0          # detections on `shown`
    picked: bool = False         # the shot the pick was made in
    lone: bool = False           # taken under the line, as the only person on its frame
    width: int = 0               # columns the mask on `shown` covers
    seen: int = 0                # the most detections on any frame looked at


@dataclass
class Followed:
    """`follow`'s result."""
    shots: list[Shot]
    pieces: dict[int, torch.Tensor] = field(default_factory=dict)   # {first frame: [n, H, W]}
    pick_frame: int | None = None    # None when nothing was detected to pick
    others: int = 0                  # people on the pick frame the comparison is relative to
    match: float = 0.0               # the similarity a shot had to reach
    pick_width: int = 0              # columns the subject's mask covers on the pick frame
    looks: list[tuple[int, int, int, float]] = field(default_factory=list)   # (shot, frame, detections, best similarity)
    views: int = 1                   # places a person is compared: head and shoulders, and the head
    views_used: int = 1              # of those, how many the subject has on the pick frame


def main_subject(shots: list[Shot], pick: str, detect, sign) -> tuple[int, int] | None:
    """The frame and detection to pick when no frame is named.

    The rule is applied on every shot's probe frame. The people it favours are
    grouped by plain similarity, and the group covering the most frames wins;
    within it, the frame showing the most people, then the earliest.
    """
    winners = []
    for shot in shots:
        masks, scores = detect(shot.probe)
        i = choose(masks, scores, pick)
        if i is not None:
            winners.append((shot, i, _views(sign(shot.probe, masks[i]))[0], int(masks.shape[0])))
    if not winners:
        return None

    def same(a, b) -> bool:
        return a is b or similarity(a[2], b[2]) >= PLAIN_SAME

    def frames_won(a) -> int:
        return sum(w[0].end - w[0].start for w in winners if same(a, w))

    lead = max(winners, key=lambda a: (frames_won(a), -a[0].start))
    shot, i, _, _ = max((w for w in winners if same(lead, w)), key=lambda w: (w[3], -w[0].start))
    return shot.probe, i


def follow(n_frames: int, cuts: list[int], pick: str, pick_frame: int | None, match_threshold: float | None,
           detect: Callable[[int], tuple[torch.Tensor, list[float]]],
           sign: Callable[[int, torch.Tensor], torch.Tensor | None],
           track: Callable[[int, int, int, torch.Tensor], torch.Tensor],
           stride: int = PROBE_STRIDE, offset: int = PROBE_OFFSET) -> Followed:
    """The subject's mask per frame, shot by shot. The model work is in three callables.

    `detect(frame)` returns that frame's detections, [N, H, W] and their
    scores. `sign(frame, mask)` returns a mask's signature on that frame, or
    a tuple of them, one per place a person is compared; a match is as good
    as its worst view, and a person lacking a view the subject has is no match.
    `track(start, end, seed, mask)` returns the masks of frames `start` to
    `end` exclusive, tracked from `mask` on `seed`, in frame order.

    `pick_frame` None picks automatically (`main_subject`); `match_threshold`
    None cuts automatically (`auto_match`).
    """
    shots = [Shot(s, e, probe=min(s + max(int(offset), 0), e - 1)) for s, e in shot_ranges(n_frames, cuts)]
    for shot in shots:
        shot.shown = shot.probe
    if pick_frame is None:
        entry = main_subject(shots, pick, detect, sign)
    else:
        if not 0 <= int(pick_frame) < int(n_frames):
            raise ValueError(f"pick_frame {pick_frame} is outside the clip's {n_frames} frames")
        masks, scores = detect(int(pick_frame))
        which = choose(masks, scores, pick)
        entry = None if which is None else (int(pick_frame), which)
    result = Followed(shots)
    if entry is None:
        return result
    frame0, which = entry
    masks0, _ = detect(frame0)
    picked = masks0[which]
    mine = _views(sign(frame0, picked))
    theirs = [_views(sign(frame0, m)) for i, m in enumerate(masks0) if i != which]
    use = [k for k, v in enumerate(mine) if v is not None]         # the views the subject has
    others = [t for t in theirs if t[0] is not None]
    centres = []
    for k in range(len(mine)):
        have = [t[k] for t in theirs if t[k] is not None]
        centres.append(torch.stack(have, dim=0).mean(dim=0) if have else None)
    reference = tuple(relative(v, c) for v, c in zip(mine, centres))
    result.pick_frame, result.others, result.pick_width = frame0, len(others), _width(picked)
    result.views, result.views_used = len(mine), len(use)

    def alike(sig) -> float | None:
        """A person's similarity to the subject: the lowest over the views the subject has. None when one is missing."""
        views = _views(sig)
        if not use or any(views[k] is None for k in use):
            return None
        return min(similarity(reference[k], relative(views[k], centres[k])) for k in use)

    def look(shot: Shot, f: int) -> tuple[float, int | None, int]:
        """Frame `f`'s best similarity, which detection has it and how many there have a head; the shot remembers its best."""
        found, _ = detect(f)
        sigs = [sign(f, m) for m in found]
        sims = [alike(v) for v in sigs]
        heads = sum(1 for v in sigs if _has_head(v))
        shot.seen = max(shot.seen, len(sims))
        able = [k for k, v in enumerate(sims) if v is not None]
        if not able:
            result.looks.append((shots.index(shot) + 1, f, len(sims), -1.0))
            return -1.0, None, heads
        i = max(able, key=lambda k: sims[k])
        result.looks.append((shots.index(shot) + 1, f, len(sims), sims[i]))
        if sims[i] > shot.best:
            shot.best, shot.shown, shot.index, shot.candidates, shot.width = sims[i], f, i, len(sims), _width(found[i])
        return sims[i], i, heads

    rest = [s for s in shots if not s.start <= frame0 < s.end]
    first = {id(s): look(s, s.probe) for s in rest}
    floor = MATCH_FLOOR if others else PLAIN_FLOOR
    result.match = float(match_threshold) if match_threshold is not None else auto_match([s.best for s in rest], floor)
    # nobody else on the pick frame and nothing named: there is nobody to mistake the subject for
    alone = match_threshold is None and not others

    for shot in shots:
        if shot.start <= frame0 < shot.end:
            shot.picked, shot.seed, shot.shown, shot.index = True, frame0, frame0, which
            shot.best, shot.candidates, shot.width = 1.0, int(masks0.shape[0]), result.pick_width
            result.pieces[shot.start] = track(shot.start, shot.end, frame0, picked)
            continue
        taken = lone = None        # (similarity, frame, detection)
        later = [f for f in range(shot.start, shot.end, max(int(stride), 1)) if f != shot.probe]
        for f in [shot.probe] + later:
            score, i, heads = first[id(shot)] if f == shot.probe else look(shot, f)
            if i is None:
                continue
            if score >= result.match:
                taken = (score, f, i)
                break
            if alone and heads == 1 and (lone is None or score > lone[0]):
                lone = (score, f, i)
        if taken is None and lone is not None:
            taken, shot.lone = lone, True
        if taken is None:
            continue
        score, seed, i = taken
        seeds, _ = detect(seed)
        shot.seed, shot.shown, shot.index, shot.best = seed, seed, i, score
        shot.candidates, shot.width = int(seeds.shape[0]), _width(seeds[i])
        result.pieces[shot.start] = track(shot.start, shot.end, seed, seeds[i])
    return result


def assemble(n_frames: int, height: int, width: int, pieces: dict[int, torch.Tensor]) -> torch.Tensor:
    """[n_frames, height, width] of 0 or 1: the tracked pieces in place, zeros where the subject is absent."""
    out = torch.zeros((int(n_frames), int(height), int(width)), dtype=torch.float32)
    for start, piece in pieces.items():
        out[start:start + piece.shape[0]] = (piece > 0.5).to(torch.float32)
    return out


def _state(shot: Shot) -> str:
    if shot.picked:
        return "picked"
    if shot.seed is None:
        return "absent"
    return "taken (only person)" if shot.lone else "taken"


def _framing(shot: Shot, found: Followed) -> str:
    """Words for a shot framed much closer or wider than the pick frame, where the top third is another part of a person."""
    if not shot.width or not found.pick_width:
        return ""
    ratio = shot.width / found.pick_width
    if 2 / 3 <= ratio <= 1.5:
        return ""
    return f"; framed differently, the mask is {ratio:.1f} times as wide as on the pick frame"


def report(found: Followed, cuts: list[int], pick: str, phrase: str, named_frame: bool, named_value: bool,
           seconds: float, cutting: str | None = None) -> str:
    """What was found, in words: the cuts, the pick, the line between the subject and others, each shot, the cost.

    `cutting` is `cuts_line`'s sentence about the cut threshold, when the caller has one.
    """
    shots = found.shots
    lines = [f"{len(shots)} shot(s); cuts at frame(s) {cuts if cuts else 'none'}"]
    if cutting:
        lines.append(cutting)
    if found.pick_frame is None:
        lines.append(f"nothing matched `{phrase}` to pick from: no subject, every mask is empty")
    else:
        how = "the frame named" if named_frame else "chosen automatically: the person that rule favours for most of the clip"
        lines.append(f"subject: the {pick} `{phrase}` on frame {found.pick_frame} ({how})")
        if found.others:
            lines.append(f"similarity is relative to the {found.others} other(s) on that frame")
        else:
            lines.append("nobody else on that frame: plain similarity, where other people score about 0.8 to 0.9"
                         + ("" if named_value else "; a shot's best frame showing one person with a head is taken whatever it scores"))
        if found.views > 1:
            lines.append("a match has to hold in two places, the head and shoulders and the head; the lower one counts"
                         if found.views_used == found.views else
                         "no head was found on the subject on that frame: matched by the head and shoulders alone")
        scores = sorted((s.best for s in shots if not s.picked and s.best >= 0), reverse=True)
        above = " ".join(f"{v:.2f}" for v in scores if v >= found.match)
        below = " ".join(f"{v:.2f}" for v in scores if v < found.match)
        lines.append(f"match: {'the value named' if named_value else 'automatic'}, {found.match:.2f}; "
                     f"shots' scores: {above or '(none)'} | {below or '(none)'}")
    for n, s in enumerate(shots, 1):
        span = f"[{n}] frames {s.start}-{s.end - 1}"
        if s.picked:
            lines.append(f"{span}: the picked shot, {s.candidates} detection(s) on frame {s.seed}")
        elif s.seed is not None:
            how = "as the only person there, " if s.lone else ""
            lines.append(f"{span}: taken on frame {s.seed}, {how}similarity {s.best:.2f}, "
                         f"{s.candidates} detection(s) there{_framing(s, found) if s.lone else ''}")
        else:
            why = (f"best similarity {s.best:.2f} on frame {s.shown}{_framing(s, found)}" if s.best >= 0 else
                   "no detection" if not s.seen else f"up to {s.seen} detection(s), none that could be compared")
            lines.append(f"{span}: absent ({why})")
    lines.append(f"{seconds:.0f} s")
    return "\n".join(lines)


def _outline(mask: torch.Tensor, width: int) -> torch.Tensor:
    """The inner edge of a [H, W] mask, `width` pixels thick."""
    m = (mask > 0.5).to(torch.float32)[None, None]
    inner = -F.max_pool2d(-m, 2 * width + 1, stride=1, padding=width)
    return (m - inner)[0, 0] > 0.5


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an older Pillow has one size
        return ImageFont.load_default()


def preview(frames: torch.Tensor, mask: torch.Tensor, shots: list[Shot], detect) -> torch.Tensor:
    """One labelled frame per shot, [shots, h, TILE_WIDTH, 3].

    The frame is the one the shot was taken on, or where its best candidate
    was seen. The subject's mask is tinted. Every detection there is outlined:
    green for the one taken, orange for the best candidate of an absent shot,
    white for the rest. The label gives the shot, its frames, its score and
    whether it was picked, taken or absent.
    """
    height, width = int(frames.shape[1]), int(frames.shape[2])
    th = max(2, round(height * TILE_WIDTH / width))
    tiles = []
    for n, s in enumerate(shots, 1):
        img = frames[s.shown, ..., :3].to(torch.float32).cpu().clone()
        tint = mask[s.shown].unsqueeze(-1) * 0.35
        img = img * (1.0 - tint) + torch.tensor([1.0, 0.1, 0.1]) * tint
        found, _ = detect(s.shown)
        for i, m in enumerate(found):
            if i == s.index:
                colour, thick = ([0.1, 1.0, 0.2] if s.seed is not None else [1.0, 0.6, 0.0]), 4
            else:
                colour, thick = [1.0, 1.0, 1.0], 1
            img[_outline(m, thick)] = torch.tensor(colour)
        small = F.interpolate(img.movedim(-1, 0)[None], size=(th, TILE_WIDTH), mode="area")[0].movedim(0, -1)
        pil = Image.fromarray((small.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
        score = "" if s.picked or s.best < 0 else f"  {s.best:.2f}"
        text = f"{n}  frames {s.start}-{s.end - 1}{score}  {_state(s)}"
        draw, font = ImageDraw.Draw(pil), _font(22)
        box = draw.textbbox((8, 6), text, font=font)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        tiles.append(torch.from_numpy(np.asarray(pil).copy()).to(torch.float32) / 255.0)
    return torch.stack(tiles, dim=0) if tiles else torch.zeros((1, th, TILE_WIDTH, 3))


def _sam_callables(segmenter, segmenter_clip, frames: torch.Tensor, phrase: str, detection_threshold: float,
                   max_people: int = MAX_PEOPLE, head_phrase: str = HEAD_PHRASE):
    """`detect`, `sign` and `track` on core's SAM 3: its detect and track nodes, and the model's vision trunk."""
    import comfy.model_management
    import comfy.utils
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_TrackToMask, SAM3_VideoTrack  # core's nodes

    cond = segmenter_clip.encode_from_tokens_scheduled(segmenter_clip.tokenize(counted(phrase, max_people)))
    head_cond = segmenter_clip.encode_from_tokens_scheduled(segmenter_clip.tokenize(counted(head_phrase, max_people)))
    head_masks: dict[int, torch.Tensor] = {}
    detections: dict[int, tuple[torch.Tensor, list[float]]] = {}
    features: dict[int, torch.Tensor] = {}

    def detect(f: int):
        if f not in detections:
            out = SAM3_Detect.execute(segmenter, frames[f:f + 1], conditioning=cond,
                                      threshold=float(detection_threshold), individual_masks=True)
            masks, boxes = getattr(out, "args", out)[:2]
            scores = [float(b.get("score", 0.0)) for b in (boxes[0] if boxes else [])]
            detections[f] = (masks.to(torch.float32).cpu(), scores[:int(masks.shape[0])])
        return detections[f]

    def sign(f: int, mask: torch.Tensor):
        if f not in features:
            comfy.model_management.load_model_gpu(segmenter)
            device = comfy.model_management.get_torch_device()
            dtype = segmenter.model.get_dtype()
            x = comfy.utils.common_upscale(frames[f:f + 1, ..., :3].movedim(-1, 1), TRUNK_SIDE, TRUNK_SIDE,
                                           "bilinear", crop="disabled").to(device=device, dtype=dtype)
            trunk = segmenter.model.diffusion_model.detector.backbone["vision_backbone"].trunk(x)
            trunk = trunk[-1] if isinstance(trunk, (list, tuple)) else trunk
            features[f] = trunk[0].to(torch.float32).cpu()
        if f not in head_masks:
            out = SAM3_Detect.execute(segmenter, frames[f:f + 1], conditioning=head_cond,
                                      threshold=float(detection_threshold), individual_masks=True)
            head_masks[f] = getattr(out, "args", out)[0].to(torch.float32).cpu()
        head = head_of(mask, head_masks[f])
        return signature(features[f], top_third(mask)), (None if head is None else signature(features[f], head))

    def run(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = SAM3_VideoTrack.execute(images, segmenter, initial_mask=mask[None].to(torch.float32),
                                      conditioning=None, detection_threshold=float(detection_threshold),
                                      max_objects=1, detect_interval=1)
        data = getattr(out, "args", out)[0]
        masks = SAM3_TrackToMask.execute(data, "")
        return getattr(masks, "args", masks)[0].to(torch.float32).cpu()

    def track(start: int, end: int, seed: int, mask: torch.Tensor) -> torch.Tensor:
        forward = run(frames[seed:end], mask)
        if seed == start:
            return forward
        backward = run(torch.flip(frames[start:seed + 1], dims=[0]), mask)
        return torch.cat([torch.flip(backward, dims=[0])[:-1], forward], dim=0)

    return detect, sign, track


def _selection(value, key: str, nested: str):
    """A DynamicCombo's selection and its nested value (None when the option has none).

    It arrives as one nested dict, the selection under the input's own id. A
    bare string is accepted for a direct call from a check.
    """
    if isinstance(value, str):
        return value, None
    return value[key], value.get(nested)


class MiniMaxH3SubjectTrack(io.ComfyNode):
    #: Part of a kept mask's key (`mask_store.mask_key`). Bump when the same
    #: inputs and settings would give a different mask. 2: the comparison became
    #: relative to the pick frame's other people. 3: every detection of the
    #: phrase is taken, not core's one per phrase. 4: the automatic pick and
    #: the automatic match, and each shot judged a few frames in. 5: the cut
    #: threshold from the clip's scores and inclusive, and a lone person taken
    #: under the line when the pick frame shows nobody else. 6: a match has to
    #: hold on the head as well, and the lone person has to have a head.
    MASK_VERSION = 6

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SubjectTrack",
            display_name="MiniMax H3 Subject Track (one person across cuts)",
            category="model/latent/minimax",
            description=(
                "Follows one person through a clip that has cuts and other people in it, and gives one mask "
                "per frame, empty where they are not on screen. Wire the mask into a Masked Source node. "
                "The preview shows who it took in each shot and the report says why."),
            inputs=[
                io.Image.Input("frames", tooltip="The source video's frames, the same ones the Masked Source node gets."),
                io.Model.Input("segmenter", tooltip="The SAM 3 checkpoint's model."),
                io.Clip.Input("segmenter_clip", tooltip="The SAM 3 checkpoint's text encoder."),
                io.String.Input("subject_phrase", default=SUBJECT_PHRASE,
                                tooltip="What SAM 3 is asked to find in each shot. `person` finds everyone."),
                io.Combo.Input("pick", options=list(PICKS), default=PICK_LARGEST,
                               tooltip=("Which of the people found is the subject: the one covering the most of "
                                        "the frame, the one nearest its centre, or SAM 3's highest score.")),
                io.DynamicCombo.Input(
                    "pick_on",
                    options=[
                        io.DynamicCombo.Option(AUTOMATIC, []),
                        io.DynamicCombo.Option(PICK_ON_FRAME, [
                            io.Int.Input("pick_frame", default=0, min=0, max=1_000_000,
                                         tooltip="The frame `pick` is applied on."),
                        ]),
                    ],
                    tooltip=("Where the subject is picked. `automatic` takes the person `pick` favours for most "
                             "of the clip. `a frame I name` applies `pick` on one frame.")),
                io.DynamicCombo.Input(
                    "match",
                    options=[
                        io.DynamicCombo.Option(AUTOMATIC, []),
                        io.DynamicCombo.Option(AT_VALUE, [
                            io.Float.Input("match_threshold", default=MATCH_THRESHOLD, min=-1.0, max=1.0, step=0.01,
                                           tooltip="A person in another shot is the subject at or above this similarity."),
                        ]),
                    ],
                    tooltip=("How alike a person in another shot has to be to count as the subject. `automatic` "
                             "sets it from the shots' scores. `at a value` uses yours.")),
                io.DynamicCombo.Input(
                    "cuts",
                    options=[
                        io.DynamicCombo.Option(AUTOMATIC, []),
                        io.DynamicCombo.Option(AT_VALUE, [
                            io.Float.Input("cut_threshold", default=CUT_THRESHOLD, min=0.0, max=2.0, step=0.01,
                                           tooltip="Two consecutive frames at least this different are a cut."),
                        ]),
                    ],
                    tooltip=("How different two consecutive frames have to be to count as a cut. `automatic` "
                             "sets it from the clip's own scores. `at a value` uses yours.")),
                io.Float.Input("detection_threshold", default=DETECTION_THRESHOLD, min=0.0, max=1.0, step=0.01,
                               advanced=True, tooltip="SAM 3's score threshold for a detection of the phrase."),
                io.Int.Input("max_people", default=MAX_PEOPLE, min=1, max=64, advanced=True,
                             tooltip="The most matches of the phrase taken from one frame."),
                # appended 2026-10-04
                io.String.Input("head_phrase", default=HEAD_PHRASE, advanced=True,
                                tooltip=("What SAM 3 is asked to find on each person so they can be told apart. "
                                         "A person in another shot has to match the subject there as well.")),
            ],
            outputs=[
                io.Mask.Output(display_name="mask", tooltip="One mask per frame at the frames' size; empty where the subject is absent."),
                io.Image.Output(display_name="preview", tooltip="One labelled frame per shot."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, frames, segmenter, segmenter_clip, pick_on, match, cuts, subject_phrase=SUBJECT_PHRASE,
                pick=PICK_LARGEST, detection_threshold=DETECTION_THRESHOLD, max_people=MAX_PEOPLE,
                head_phrase=HEAD_PHRASE) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
        where, pick_frame = _selection(pick_on, "pick_on", "pick_frame")
        how, match_threshold = _selection(match, "match", "match_threshold")
        cutting, cut_threshold = _selection(cuts, "cuts", "cut_threshold")
        if where not in (AUTOMATIC, PICK_ON_FRAME) or how not in (AUTOMATIC, AT_VALUE) or cutting not in (AUTOMATIC, AT_VALUE):
            raise ValueError(f"unknown pick_on {where!r}, match {how!r} or cuts {cutting!r}")
        named_frame, named_value, named_cut = where == PICK_ON_FRAME, how == AT_VALUE, cutting == AT_VALUE
        if named_frame and pick_frame is None:
            raise ValueError("pick_on `a frame I name` needs its `pick_frame`")
        if named_value and match_threshold is None:
            raise ValueError("match `at a value` needs its `match_threshold`")
        if named_cut and cut_threshold is None:
            raise ValueError("cuts `at a value` needs its `cut_threshold`")
        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        began = time.perf_counter()
        steps = cut_scores(frames)
        cut_at = float(cut_threshold) if named_cut else auto_cuts(steps)
        found_cuts = find_cuts(steps, cut_at)
        detect, sign, track = _sam_callables(segmenter, segmenter_clip, frames, subject_phrase, detection_threshold,
                                             int(max_people), head_phrase)
        with torch.no_grad():
            found = follow(n, found_cuts, pick, int(pick_frame) if named_frame else None,
                           float(match_threshold) if named_value else None, detect, sign, track)
        mask = assemble(n, h, w, found.pieces)
        text = report(found, found_cuts, pick, subject_phrase, named_frame, named_value, time.perf_counter() - began,
                      cutting=cuts_line(steps, cut_at, named_cut))
        logger.info("[h3] MiniMaxH3SubjectTrack: %s", text.replace("\n", "; "))
        tiles = preview(frames, mask, found.shots, detect)
        shown = {**ui.PreviewImage(tiles, cls=cls).as_dict(), **ui.PreviewText(text).as_dict()}
        return io.NodeOutput(mask, tiles, text, ui=shown)
