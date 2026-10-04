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

1. Cuts (`cut_scores`, `find_cuts`): one minus the correlation of consecutive
   frames' gradient maps at a small size. A cut changes where the edges are; a
   lighting change does not. ffmpeg's scene score failed on the band clip
   under its coloured gels; this score separated eight of its nine cuts from
   every other frame (the ninth is a jump cut inside a cutaway).
2. The pick: on `pick_frame`, SAM 3 is asked for `subject_phrase` and one of
   the detections is chosen by `pick` (`choose`). That mask is the subject.
3. Each other shot: the detections on a frame of that shot are compared with
   the subject by the vision trunk's features pooled under the top third of
   each mask, the head and shoulders (`top_third`, `signature`). The other
   people on the pick frame are certainly not the subject, so their average
   signature is subtracted from every signature before comparing (`relative`,
   `similarity`): what is left is how a person differs from the people the
   subject was standing among. The best one at or above `match_threshold`
   seeds the shot; a shot with none is left empty. A shot is probed at its
   first frame and then every `PROBE_STRIDE` frames, so a subject who walks in
   late is still found.
4. Tracking: from the seed frame forward to the shot's end and backward to its
   start, each a tracker call with the seed mask as `initial_mask`.

**How far the matching can be trusted.** SAM 3's trunk is trained to say what
a thing is, not who, so people in the same clothes score close together. On
the owner's band clip (six people in one sweatshirt; measured on CPU with the
masks of an earlier track, `docs/research/masking/2026-10-04_mrhf.md`):

- pooled under the whole mask, a neighbour in the same frame scores as close
  to the pick as the lead in another shot does. Not usable.
- under the top third, plain similarity: the lead facing the camera 0.945 and
  above, everyone else 0.89 and below. A gap of about 0.06.
- under the top third, with the pick frame's other people subtracted: the lead
  facing the camera 0.77 and above, 0.88 and above on the first frame of each
  shot he is in; everyone else 0.65 and below. About three times the gap.
- seen from behind he scores inside the others' range either way. A shot that
  opens on his back is not found, and the report shows it as absent with its
  best value. On that clip every shot he is in opens on his face, and the
  tracker follows him as he turns.

When the pick frame shows nobody else there is nothing to subtract and the
plain similarity is used, on its own scale: other people then score about 0.8
to 0.9, so `match_threshold` wants to be near 0.91 there. The report says
which of the two was used. One clip; the threshold is an input for that
reason.

Everything that asks SAM for something is an input the user can read and
change (`subject_phrase`, `detection_threshold`, `match_threshold`,
`cut_threshold`), and the report says what was found: the cuts, the pick, each
shot's best similarity and whether it was taken, and the seconds it cost.

**Keeping the mask** is not this node's job. `MiniMaxH3MaskedSource` keeps a
finished mask across runs (`mask_store.py`) and on a hit never asks for its
`mask` input, so core does not run this node at all. Two things here serve
that. `MASK_VERSION` on the node goes into the kept mask's key: bump it when
a change would give a different mask from the same inputs and settings (the
cut score, the pick, the signature, how a shot is seeded and tracked), and
not for a tooltip or the report's wording. And the preview and the report are
shown as the node's own UI, so nothing has to be wired to see them: a preview
or save node on either output would make core run the tracker on every queue,
whatever the Masked Source decided. The outputs exist; a shipped graph leaves
them unwired.

Nothing here patches core. It calls core's own nodes (`SAM3_Detect`,
`SAM3_VideoTrack`, `SAM3_TrackToMask`) and the SAM 3 model's vision trunk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui

logger = logging.getLogger(__name__)

#: The size frames are reduced to for the cut score, width by height. Small on
#: purpose: a cut moves every edge, and grain or a gesture should not count.
#: Reasoned, not measured.
CUT_SIZE = (192, 108)
#: The default of `cut_threshold`. Measured on one clip: the band segment's
#: real cuts score 0.99 and above and no other frame scores above 0.77
#: (`docs/research/masking/2026-10-04_mrhf.md`). Not validated elsewhere.
CUT_THRESHOLD = 0.9
#: Frames between probes of a shot when its first frame has no match.
#: Reasoned: half a second at the pack's frame rate.
PROBE_STRIDE = 12
#: The default of `match_threshold`. Measured on one clip, the band segment,
#: with the pick frame's other people subtracted: between everyone else (0.65
#: and below) and the lead facing the camera (0.77 and above)
#: (`docs/research/masking/2026-10-04_mrhf.md`). Not validated elsewhere, and
#: on the wrong scale when the pick frame shows only the subject.
MATCH_THRESHOLD = 0.71
#: The default of `detection_threshold`. Inherited: core's node default
#: (`comfy_extras/nodes_sam3.py::SAM3_Detect.define_schema`).
DETECTION_THRESHOLD = 0.5
#: The default of `subject_phrase`.
SUBJECT_PHRASE = "person"
#: The side SAM 3's vision trunk takes. Inherited: core's detect node scales
#: every frame to it (`comfy_extras/nodes_sam3.py::SAM3_Detect.execute`).
TRUNK_SIDE = 1008

PICK_LARGEST = "largest"
PICK_CENTRAL = "most central"
PICK_SCORE = "best match for the phrase"
PICKS = (PICK_LARGEST, PICK_CENTRAL, PICK_SCORE)


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
    """The frames that start a new shot: every frame whose step in scores above `threshold`."""
    return [int(i) + 1 for i in (scores > float(threshold)).nonzero().flatten()]


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


@dataclass
class Shot:
    """What `follow` did with one shot, for the report and the preview."""
    start: int
    end: int
    seed: int | None = None      # the frame the subject was found on; None when absent
    best: float = -1.0           # the best similarity seen while probing
    candidates: int = 0          # detections on the frame that was taken, or on the last one probed


def follow(n_frames: int, cuts: list[int], pick_frame: int, pick: str, match_threshold: float,
           detect: Callable[[int], tuple[torch.Tensor, list[float]]],
           sign: Callable[[int, torch.Tensor], torch.Tensor | None],
           track: Callable[[int, int, int, torch.Tensor], torch.Tensor],
           stride: int = PROBE_STRIDE) -> tuple[dict[int, torch.Tensor], list[Shot], torch.Tensor | None, int]:
    """The subject's mask per frame, shot by shot. The model work is in three callables.

    `detect(frame)` returns that frame's detections, [N, H, W] and their
    scores. `sign(frame, mask)` returns a mask's signature on that frame.
    `track(start, end, seed, mask)` returns the masks of frames `start` to
    `end` exclusive, tracked from `mask` on `seed`, in frame order.

    Returns the tracked pieces as {first frame: [n, H, W]}, one `Shot` per
    shot, the picked mask (None when nothing was detected on the pick frame,
    in which case every shot is empty), and how many other people on the pick
    frame the comparison was made relative to (0 is the plain similarity).
    """
    ranges = shot_ranges(n_frames, cuts)
    shots = [Shot(s, e) for s, e in ranges]
    if not 0 <= int(pick_frame) < int(n_frames):
        raise ValueError(f"pick_frame {pick_frame} is outside the clip's {n_frames} frames")
    masks, scores = detect(int(pick_frame))
    which = choose(masks, scores, pick)
    if which is None:
        return {}, shots, None, 0
    picked = masks[which]
    others = [v for v in (sign(int(pick_frame), m) for i, m in enumerate(masks) if i != which) if v is not None]
    centre = torch.stack(others, dim=0).mean(dim=0) if others else None
    reference = relative(sign(int(pick_frame), picked), centre)
    pieces: dict[int, torch.Tensor] = {}
    for shot in shots:
        seed, seed_mask = None, None
        if shot.start <= int(pick_frame) < shot.end:
            seed, seed_mask = int(pick_frame), picked
            shot.best, shot.candidates = 1.0, int(masks.shape[0])
        else:
            for f in range(shot.start, shot.end, max(int(stride), 1)):
                found, _ = detect(f)
                shot.candidates = int(found.shape[0])
                sims = [similarity(reference, relative(sign(f, m), centre)) for m in found]
                if sims and max(sims) > shot.best:
                    shot.best = max(sims)
                if sims and max(sims) >= float(match_threshold):
                    seed, seed_mask = f, found[sims.index(max(sims))]
                    break
        if seed is None or seed_mask is None:
            continue
        shot.seed = seed
        pieces[shot.start] = track(shot.start, shot.end, seed, seed_mask)
    return pieces, shots, picked, len(others)


def assemble(n_frames: int, height: int, width: int, pieces: dict[int, torch.Tensor]) -> torch.Tensor:
    """[n_frames, height, width] of 0 or 1: the tracked pieces in place, zeros where the subject is absent."""
    out = torch.zeros((int(n_frames), int(height), int(width)), dtype=torch.float32)
    for start, piece in pieces.items():
        out[start:start + piece.shape[0]] = (piece > 0.5).to(torch.float32)
    return out


def report(shots: list[Shot], cuts: list[int], pick_frame: int, pick: str, phrase: str, picked: bool,
           match_threshold: float, seconds: float, others: int = 0) -> str:
    """What was found, in words: the cuts, the pick, each shot, the cost."""
    lines = [f"{len(shots)} shot(s); cuts at frame(s) {cuts if cuts else 'none'}"]
    if not picked:
        lines.append(f"nothing matched `{phrase}` on pick_frame {pick_frame}: no subject, every mask is empty")
    else:
        lines.append(f"subject: the {pick} `{phrase}` on frame {pick_frame}")
        lines.append(f"similarity is relative to the {others} other(s) on that frame" if others else
                     "nobody else on that frame: plain similarity, where other people score about 0.8 to 0.9")
    for n, s in enumerate(shots, 1):
        if s.seed is None:
            why = "no detection" if s.best < 0 else f"best similarity {s.best:.2f}, below {match_threshold:.2f}"
            lines.append(f"[{n}] frames {s.start}-{s.end - 1}: absent ({why})")
        elif s.start <= pick_frame < s.end and s.seed == pick_frame:
            lines.append(f"[{n}] frames {s.start}-{s.end - 1}: the picked shot, {s.candidates} detection(s) on the pick frame")
        else:
            lines.append(f"[{n}] frames {s.start}-{s.end - 1}: found on frame {s.seed}, similarity {s.best:.2f}, "
                         f"{s.candidates} detection(s) there")
    lines.append(f"{seconds:.0f} s")
    return "\n".join(lines)


def preview(frames: torch.Tensor, mask: torch.Tensor, shots: list[Shot]) -> torch.Tensor:
    """One frame per shot with the subject's mask tinted, [shots, H, W, 3]: the seed frame, or the shot's first when absent."""
    tiles = []
    tint = torch.tensor([1.0, 0.1, 0.1], dtype=torch.float32)
    for s in shots:
        f = s.seed if s.seed is not None else s.start
        img = frames[f, ..., :3].to(torch.float32).cpu()
        m = mask[f].unsqueeze(-1)
        tiles.append(img * (1.0 - 0.5 * m) + tint * (0.5 * m))
    return torch.stack(tiles, dim=0) if tiles else frames[:1, ..., :3].to(torch.float32).cpu()


def _sam_callables(segmenter, segmenter_clip, frames: torch.Tensor, phrase: str, detection_threshold: float):
    """`detect`, `sign` and `track` on core's SAM 3: its detect and track nodes, and the model's vision trunk."""
    import comfy.model_management
    import comfy.utils
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_TrackToMask, SAM3_VideoTrack  # core's nodes

    cond = segmenter_clip.encode_from_tokens_scheduled(segmenter_clip.tokenize(phrase))
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
        return signature(features[f], top_third(mask))

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


class MiniMaxH3SubjectTrack(io.ComfyNode):
    #: Part of a kept mask's key (`mask_store.mask_key`). Bump when the same
    #: inputs and settings would give a different mask. 2: the comparison became
    #: relative to the pick frame's other people.
    MASK_VERSION = 2

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SubjectTrack",
            display_name="MiniMax H3 Subject Track (one person across cuts)",
            category="model/latent/minimax",
            description=(
                "Follows one person through a clip that has cuts and other people in it, and gives one mask "
                "per frame, empty where they are not on screen. You pick the person once, on one frame; "
                "the node finds the same person in every other shot. Wire the mask into a Masked Source "
                "node. The report says where the cuts are, who was picked and what was found in each shot."),
            inputs=[
                io.Image.Input("frames", tooltip="The source video's frames, the same ones the Masked Source node gets."),
                io.Model.Input("segmenter", tooltip="The SAM 3 checkpoint's model."),
                io.Clip.Input("segmenter_clip", tooltip="The SAM 3 checkpoint's text encoder."),
                io.String.Input("subject_phrase", default=SUBJECT_PHRASE,
                                tooltip=("What SAM 3 is asked to find. `person` finds everyone and lets `pick` "
                                         "choose; a narrower phrase finds fewer. The same phrase is used on "
                                         "every shot.")),
                io.Int.Input("pick_frame", default=0, min=0, max=1_000_000,
                             tooltip=("The frame the subject is picked on. Choose one where they are easy to "
                                      "tell from the others by `pick`.")),
                io.Combo.Input("pick", options=list(PICKS), default=PICK_LARGEST,
                               tooltip=("Which of the people found on `pick_frame` is the subject: the one "
                                        "covering the most of the frame, the one nearest its centre, or the "
                                        "one SAM 3 scored highest for the phrase. The preview shows who it took.")),
                io.Float.Input("match_threshold", default=MATCH_THRESHOLD, min=-1.0, max=1.0, step=0.01,
                               tooltip=("How alike a person in another shot has to be to count as the subject, "
                                        "from -1 to 1. The report gives each shot's best value: raise this if "
                                        "it took someone else, lower it if it missed them. If the pick frame "
                                        "shows nobody else the scale is different and about 0.91 is the value "
                                        "to start from; the report says when that is so.")),
                io.Float.Input("cut_threshold", default=CUT_THRESHOLD, min=0.0, max=2.0, step=0.01,
                               tooltip=("How different two consecutive frames have to be to count as a cut. "
                                        "The report lists the cuts it found: lower this if it missed one, "
                                        "raise it if it split a shot.")),
                io.Float.Input("detection_threshold", default=DETECTION_THRESHOLD, min=0.0, max=1.0, step=0.01,
                               tooltip="SAM 3's score threshold for a detection of the phrase."),
            ],
            outputs=[
                io.Mask.Output(display_name="mask", tooltip="One mask per frame at the frames' size; empty where the subject is absent."),
                io.Image.Output(display_name="preview", tooltip="One frame per shot with the subject tinted."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, frames, segmenter, segmenter_clip, subject_phrase=SUBJECT_PHRASE, pick_frame=0,
                pick=PICK_LARGEST, match_threshold=MATCH_THRESHOLD, cut_threshold=CUT_THRESHOLD,
                detection_threshold=DETECTION_THRESHOLD) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        began = time.perf_counter()
        cuts = find_cuts(cut_scores(frames), cut_threshold)
        detect, sign, track = _sam_callables(segmenter, segmenter_clip, frames, subject_phrase, detection_threshold)
        with torch.no_grad():
            pieces, shots, picked, others = follow(n, cuts, int(pick_frame), pick, float(match_threshold),
                                                   detect, sign, track)
        mask = assemble(n, h, w, pieces)
        text = report(shots, cuts, int(pick_frame), pick, subject_phrase, picked is not None,
                      float(match_threshold), time.perf_counter() - began, others)
        logger.info("[h3] MiniMaxH3SubjectTrack: %s", text.replace("\n", "; "))
        tiles = preview(frames, mask, shots)
        shown = {**ui.PreviewImage(tiles, cls=cls).as_dict(), **ui.PreviewText(text).as_dict()}
        return io.NodeOutput(mask, tiles, text, ui=shown)
