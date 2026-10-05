"""A source video and a subject mask, for masked video-to-video.

The owner's case (2026-10-04): take a music video, keep its audio and
everything outside one subject, and regenerate that subject from a reference
still. Core already has the mechanism. A latent `noise_mask` is a per-token
timestep, not an attention mask: `comfy/ldm/minimax/model.py::mask_row_values`
pools it to the DiT's 2x2 patch grid and `MiniMaxH3Model._forward` pins a
preserved token at the conditioning timestep, with the clean latent
re-injected every step (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`).
This module supplies what core leaves to the graph: getting a per-frame pixel
mask onto that grid without losing the subject, and putting the untouched
pixels back afterwards.

**Why not stock `SetLatentNoiseMask`.** It writes a flat mask, so the sampler
pads ones for the audio stream and the track regenerates
(`audio_freeze.py`, "Trap this node guards"). And core resizes a mask with
trilinear interpolation (`comfy/utils.py::reshape_mask`), which samples it:
the video VAE packs frames in runs of `FRAME_PER_TOKEN`, and a subject that
moves inside a run can fall between samples and stay in the plate.

**The reduction** (`token_mask`), every step a max so no subject pixel is
dropped: grow in pixels, max-pool to the latent cells, max over each temporal
run, then max over each 2x2 patch, written back onto the cells. The last step
is core's own (`mask_row_values`), done here so that the region this module
reports, the region the sampler blends and the region the composite restores
are the same set of tokens. Binary on purpose: a fractional value is a
partial-strength row, which is a different experiment.

**The composite** (`pixel_alpha`, `composite`). A preserved token does not
decode back to the source exactly: the VAE round trip and the pinned-timestep
blend both move it (`bench/results/2026-09-25_distill_audio_s1.md` is the
record for a wholly frozen video). So outside the regenerated tokens the
source's own pixels go back, with the boundary feathered. `grow_pixels` is
what keeps the feather on background: the blend reaches `feather_pixels` into
the regenerated region, and the subject sits at least `grow_pixels` inside it.

**What gets replaced** (`replace`, owner 2026-10-04: "it should be a choice on
a node"). `whole subject` regenerates the tracked person. `head and hair`
keeps the body below the hair, its clothes and its movement as the source's
own pixels, so nothing about the action has to be prompted: the body turns
because it is the original's. The part is found by core's SAM 3 detector on
the frames the subject is in, from phrases the user can read and change
(`part_phrases`), and kept only where it lies on the subject's own mask, so a
neighbour's hair is not taken.

The region is **everything of the subject down to where the part ends**, not
the part's own outline. The first render regenerated the outline alone (head
plus a mass of long hair), and the model filled it with a head that size: "a
giant bobblehead" (owner). Nothing in a head-shaped hole pins the head's
scale. Drawing the neck and shoulders with it does, because they have to
meet the kept torso (a peer session's suggestion,
`docs/research/masking/2026-10-04_mrhf.md`). With short hair the region is
little more than the head.

A frame the detector misses would otherwise regenerate nothing and show the
original's head, so a miss takes the nearest found frame's cut, and the log
says how many did. The node's `mask` output is the mask it used, for a
preview.

**Painting the subject out before the encode** (`paint_out`; a peer session's
finding, `docs/research/masking/2026-10-04_mrhf.md`). The video VAE's encoder
is convolutional, so a kept token beside the mask was encoded with the old
subject inside its receptive field, and those tokens are re-injected clean on
every step. A faint remnant of the original measured beside the regenerated
subject is the suspected result. With `paint_out` the subject's pixels are
filled from their surroundings (`fill_subject`) in the frames that are
encoded; the composite still restores the true source. Core's own H3 inpaint
path hides the original before its encode too
(`comfy_extras/nodes_minimax_h3.py::MiniMaxH3FunControlPatch`). Off by
default until a matched render says it helps.

**What the composite keeps** (`composite`; owner's notes on the first clip
and a peer's measurements and design, `docs/research/masking/2026-10-04_mrhf.md`).
The regenerated tokens hold three kinds of pixel: the new subject, where the
old subject stood, and margin that is neither. The margin is background the
model had to invent, and it is where the flicker and the cut-out look were:
the grown border, the backdrop a moving original swept, and anything the
tracker marked by mistake. `whole region` keeps all three from the render.
`only what changed` keeps the render where it differs from the source by
more than `change_threshold` (the new subject and its shadow; the old
subject's place, which is also forced by its mask) and restores the source
where the model merely repainted the backdrop. No second detector pass and
no phrase: a neighbour the model reproduced faithfully simply stays source.
Then a generous `grow_pixels` costs nothing visible, so a replacement that
reaches past the original's outline is not cut off. It does not fix where
the old subject stood and the new one does not: that needs a plate with
nobody in it.

**Painting out, as built, made things worse** (owner, on the proof render:
"much worse than before"). It stays off. The likely reason (inferred): the
original's trace in the kept tokens also tells the model where the subject
is and how big, and a hole smaller than the original's hair leaves a fringe
the model continues as dark lines.

**A motion reference** (`motion_reference`, 2026-10-05). The per-token
mechanism carries no movement: the replaced subject faces the camera while
the original turns, and every way of putting the source into the target rows
(a late start, plain or softened) brought the original's look with its pose
(`bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`). The channel the
model was trained to take motion from is a video reference, so this node can
ask the song node to show each window to the model as `<Video 1>` as well:
the subject alone on grey, or the whole window, at a short edge that sets its
cost, with or without the video model's copy (`motion_vae`; off is the
encoder-only form, a few thousand text tokens per window). The song node
builds it per window from the source it already holds (`window_frames`,
`motion_reference`), so nothing is wired and no copy of the clip is kept. The
prompt names the relationship, never the action: the masking board's route
1, `docs/research/masking/2026-10-05_mryellow.md` section 7.

**The mask is kept across runs** (`reuse_mask`; owner, 2026-10-04: "save the
mask"). Tracking the subject, and finding the part for `head and hair`, cost
more than a window of sampling after every restart, and a clip's mask does
not change between the arms of a test. The node keeps its finished mask on
disk under a key made from everything that decides it, and asks core for
`mask`, `segmenter` and `segmenter_clip` only when nothing kept matches, so on
a hit the tracker and the detector never run. A hit is the same bytes as the
tracked mask. `mask_store.py` has the key, the format and the budget;
`MASK_KEY_SKIP` below is the list of this node's inputs that do not change
its mask.

Design from two third-party nodes, read and not run:
`coderef/comfyui_dagthomas/nodes/h3/mouth_guard.py` (pixel grow, max-pool,
max per run; it protects where this regenerates) and
`coderef/ComfyUI-H3-Motion-Context-MultiRef/h3_v2v_fractional.py::_mask_to_video_latent`
(the per-run reduction). Neither feathers a composite.

Nothing here patches core.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from comfy_api.latest import io

import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN

logger = logging.getLogger(__name__)

H3MaskedSource = io.Custom("H3_MASKED_SOURCE")

#: Frames per chunk for the pixel-space pools and blends: bounds the transient
#: memory of a window. Reasoned, not measured.
CHUNK = 48

#: The `replace` choices. The second finds a part with core's SAM 3 detector.
REPLACE_WHOLE = "whole subject"
REPLACE_PART = "head and hair"
#: The default of `grow_pixels`. The owner's choice on one clip, 2026-10-04:
#: twice a DiT token of canvas "preserved identity better" than one, which
#: was the first, reasoned value. One seed each.
GROW_PIXELS = 64
#: The `composite` choices. The second is the default: it is the composite of
#: the render the owner called the best (2026-10-04, one clip).
COMPOSITE_REGION = "whole region"
COMPOSITE_CHANGED = "only what changed"
#: The default of `change_threshold`: how far the render must differ from the
#: source, as a fraction of full scale averaged over the colour channels and a
#: small neighbourhood, to be kept. Reasoned from a peer's measurements on the
#: first clip (invented backdrop a few levels from the source, a subject tens
#: of levels), not tuned.
CHANGE_THRESHOLD = 0.05
#: The neighbourhood the difference is averaged over, in pixels to each side.
#: Reasoned, not measured.
CHANGE_BLUR = 4
#: The default of `part_phrases`: the union of the detections is the part.
#: "hair" as well as "head" because long hair lies on the chest and back,
#: outside any head. Reasoned, not measured.
PART_PHRASES = "hair, head"
#: The default of `part_margin`: a part is kept where it lies on the subject's
#: mask widened by this many pixels of the source frame, because the two masks
#: come from different passes and their edges do not coincide. Reasoned, not
#: measured.
PART_MARGIN = 8
#: The default of `part_threshold`, the detector's score threshold for a part.
#: Inherited: core's node default
#: (`comfy_extras/nodes_sam3.py::SAM3_Detect.define_schema`).
PART_THRESHOLD = 0.5
#: This node's inputs that do not change the mask it settles on: they act on
#: the mask afterwards (the grow, the feather, the composite) or on the frames.
#: Everything else on the node, and everything upstream of it, is in the key a
#: kept mask is found by (`mask_store.mask_key`). Reasoned, from `execute`: the
#: mask is final before any of these is read. `bench/check_mask_store.py`
#: holds both directions.
MASK_KEY_SKIP = ("grow_pixels", "feather_pixels", "paint_out", "composite", "change_threshold", "reuse_mask",
                 "motion_reference", "motion_short_edge", "motion_vae")
#: `motion_reference` choices: what of the source window, if anything, the
#: song node appends to the reference chain as a video, so the model is shown
#: the original's movement through the channel it was trained to take motion
#: from (`docs/research/masking/2026-10-05_mryellow.md`, section 2, route A).
#: Nothing in the per-token-timestep mechanism carries movement: the arms of
#: 2026-10-04 showed that whatever enters the target rows brings its look.
MOTION_NONE = "none"
MOTION_SUBJECT = "subject only"
MOTION_FRAME = "whole frame"
#: Shorter side of that reference, in pixels. Reasoned, 2026-10-05: core never
#: enlarges a reference video, so this sets its pixel area; with the VAE copy
#: on, a 384 short edge costs a quarter of the 768 canvas's rows
#: (`docs/h3_references.md`, "Budget by pixel area"). Encoder-only, it sets
#: how much the text encoder sees of the subject at two frames per second.
MOTION_SHORT_EDGE = 384
#: The inputs core is asked for only when no kept mask matches.
LAZY_FOR_MASK = ("mask", "segmenter", "segmenter_clip")


def run_lengths(latent_t: int) -> list[int]:
    """Pixel frames under each latent step: `FRAME_PER_TOKEN`, cyclic."""
    return [FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(int(latent_t))]


def fit_frames(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[N, H, W, C] to [N, height, width, 3], centre-cropped to the canvas's shape.

    The same call core's own H3 nodes fit a clip with
    (`comfy_extras/nodes_minimax_h3.py`, `common_upscale(..., "center")`).
    """
    out = comfy.utils.common_upscale(frames[..., :3].movedim(-1, 1), int(width), int(height), "bilinear", "center")
    return out.movedim(1, -1)


def fit_mask(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[N, h, w] to [N, height, width] through the crop `fit_frames` makes.

    The crop is `comfy.utils.common_upscale`'s centre crop, restated because
    that function then interpolates, and interpolation samples: a thin mask
    feature can fall between samples on the way down. Here a mask that is
    being made smaller is max-pooled, so nothing is lost; one being made
    larger is interpolated, where nothing can be. `bench/check_video_mask.py`
    holds this crop to core's.
    """
    width, height = int(width), int(height)
    old_h, old_w = int(mask.shape[-2]), int(mask.shape[-1])
    old_aspect, new_aspect = old_w / old_h, width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
    m = mask.to(torch.float32).narrow(-2, y, old_h - y * 2).narrow(-1, x, old_w - x * 2).unsqueeze(1)
    if m.shape[-2] >= height and m.shape[-1] >= width:
        return F.adaptive_max_pool2d(m, (height, width))[:, 0]
    return F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)[:, 0]


#: A dilation of at least this many pixels is done on a mask reduced by
#: `GROW_COARSE`, which costs a small fraction of the full-size pool. The
#: result covers at least the asked distance and at most `GROW_COARSE - 1`
#: pixels more. Reasoned: the reduction is far below a DiT token.
GROW_COARSE_FROM = 16
GROW_COARSE = 4


def grow(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Dilate a [N, H, W] mask by at least `pixels` in every direction (a square max-pool).

    Exact below `GROW_COARSE_FROM`. From there the mask is max-pooled down by
    `GROW_COARSE`, dilated there and brought back, so it never covers less
    than asked and the cost stays flat as the distance grows.
    """
    p = int(pixels)  # zero is a kernel of one, which returns the mask
    coarse = p >= GROW_COARSE_FROM
    k = -(-p // GROW_COARSE) if coarse else p
    parts = []
    for i in range(0, mask.shape[0], CHUNK):
        m = mask[i:i + CHUNK].unsqueeze(1)
        if coarse:
            m = F.max_pool2d(m, GROW_COARSE, ceil_mode=True)
        m = F.max_pool2d(m, (1, 2 * k + 1), stride=1, padding=(0, k))
        m = F.max_pool2d(m, (2 * k + 1, 1), stride=1, padding=(k, 0))
        if coarse:
            m = m.repeat_interleave(GROW_COARSE, dim=-2).repeat_interleave(GROW_COARSE, dim=-1)
            m = m[..., :mask.shape[-2], :mask.shape[-1]]
        parts.append(m[:, 0])
    return torch.cat(parts, dim=0)


def token_mask(mask: torch.Tensor, latent_t: int, lat_h: int, lat_w: int) -> torch.Tensor:
    """A [F, H, W] pixel mask (above 0.5 = regenerate) to [latent_t, lat_h, lat_w] of 0 or 1.

    F must be the window's frame count, the sum of `run_lengths(latent_t)`.
    Every value in a 2x2 patch is the same, which is what core's own pooling
    would make of it.
    """
    runs = run_lengths(latent_t)
    if int(mask.shape[0]) != sum(runs):
        raise ValueError(f"the mask has {int(mask.shape[0])} frames; a {latent_t}-step window is {sum(runs)}")
    binary = (mask > 0.5).to(torch.float32)
    cells = torch.cat([F.adaptive_max_pool2d(binary[i:i + CHUNK].unsqueeze(1), (int(lat_h), int(lat_w)))[:, 0]
                       for i in range(0, binary.shape[0], CHUNK)], dim=0)
    steps, at = [], 0
    for n in runs:
        steps.append(cells[at:at + n].amax(dim=0))
        at += n
    m = torch.stack(steps, dim=0)
    # core's patch pooling (`mask_row_values`): replicate-pad to even, max per 2x2
    m = F.pad(m.unsqueeze(1), (0, lat_w % 2, 0, lat_h % 2), mode="replicate")[:, 0]
    tokens = m.reshape(latent_t, m.shape[-2] // 2, 2, m.shape[-1] // 2, 2).amax(dim=(2, 4))
    return tokens.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)[:, :lat_h, :lat_w].contiguous()


def pixel_alpha(tokens: torch.Tensor, height: int, width: int, feather_pixels: int) -> torch.Tensor:
    """The blend weight per pixel frame, [F, height, width]: 1 keeps the render, 0 the source.

    Each latent step's mask covers the frames of its run. The feather is a
    box blur of `feather_pixels`, so the ramp is centred on the token edge
    and reaches that far to each side.
    """
    runs = torch.tensor(run_lengths(tokens.shape[0]), device=tokens.device)
    f = int(feather_pixels)
    out = []
    for i in range(0, tokens.shape[0], 12):
        a = F.interpolate(tokens[i:i + 12].unsqueeze(1), size=(int(height), int(width)), mode="nearest")
        # zero is a kernel of one, which returns the hard edge
        a = F.avg_pool2d(F.pad(a, (f, f, f, f), mode="replicate"), 2 * f + 1, stride=1)
        out.append(a[:, 0].repeat_interleave(runs[i:i + 12], dim=0))
    return torch.cat(out, dim=0)


def changed_alpha(images: torch.Tensor, source: torch.Tensor, tokens: torch.Tensor,
                  old_subject: torch.Tensor, feather_pixels: int, old_margin: int,
                  threshold: float) -> torch.Tensor:
    """The blend weight that keeps the render only where it changed the picture. [F, H, W].

    The change is the absolute difference from the source, averaged over the
    channels and over `CHANGE_BLUR` pixels, ramped from 0 at 0.6 of
    `threshold` to 1 at 1.4 of it, so a pixel hovering near the threshold
    fades and does not switch. The old subject's mask widened by `old_margin`
    is always kept. The result is held at its maximum over each temporal run
    (the mask's own grain), cut to the regenerated tokens, widened by the
    feather to close small holes and blurred by it, and never exceeds the
    whole-region weight.
    """
    f, t = int(feather_pixels), float(threshold)
    height, width = int(images.shape[1]), int(images.shape[2])
    region = pixel_alpha(tokens, height, width, 0) > 0.5
    b = CHANGE_BLUR
    parts = []
    for i in range(0, images.shape[0], CHUNK):
        d = (images[i:i + CHUNK] - source[i:i + CHUNK].to(images.device, images.dtype)).abs().mean(dim=-1).unsqueeze(1)
        d = F.avg_pool2d(F.pad(d, (b, b, b, b), mode="replicate"), 2 * b + 1, stride=1)[:, 0]
        parts.append(((d - 0.6 * t) / max(0.8 * t, 1e-6)).clamp(0.0, 1.0))
    keep = torch.maximum(torch.cat(parts, dim=0), (grow(old_subject.to(torch.float32), old_margin) > 0.5).float())
    keep = keep * region
    held, at = [], 0
    for n in run_lengths(tokens.shape[0]):
        held.append(keep[at:at + n].amax(dim=0, keepdim=True).expand(n, -1, -1))
        at += n
    wide = grow(torch.cat(held, dim=0), f)
    soft = [F.avg_pool2d(F.pad(wide[i:i + CHUNK].unsqueeze(1), (f, f, f, f), mode="replicate"), 2 * f + 1, stride=1)[:, 0]
            for i in range(0, wide.shape[0], CHUNK)]
    return torch.minimum(torch.cat(soft, dim=0), pixel_alpha(tokens, height, width, f))


def composite(images: torch.Tensor, source: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """`alpha * images + (1 - alpha) * source`, [F, H, W, 3] with alpha [F, H, W]."""
    if tuple(images.shape[:3]) != tuple(source.shape[:3]) or tuple(images.shape[:3]) != tuple(alpha.shape):
        raise ValueError(
            f"the decoded window {tuple(images.shape)}, the source {tuple(source.shape)} and the blend "
            f"weight {tuple(alpha.shape)} do not cover the same frames")
    out = torch.empty_like(images)
    for i in range(0, images.shape[0], CHUNK):
        a = alpha[i:i + CHUNK].unsqueeze(-1).to(images.device, images.dtype)
        out[i:i + CHUNK] = images[i:i + CHUNK] * a + source[i:i + CHUNK].to(images.device, images.dtype) * (1.0 - a)
    return out


def _push_pull(image: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fill where `weight` is 0 from where it is not: halve until the hole closes, then come back up.

    `image` is [N, C, H, W] already multiplied by `weight` [N, 1, H, W].
    """
    known = weight > 0
    estimate = image / weight.clamp(min=1e-6)
    if bool(known.all()) or min(image.shape[-2:]) <= 1:
        return estimate
    coarse = _push_pull(F.avg_pool2d(image, 2, ceil_mode=True), F.avg_pool2d(weight, 2, ceil_mode=True))
    up = F.interpolate(coarse, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return torch.where(known, estimate, up)


def fill_subject(pixels: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
    """[F, H, W, 3] with the pixels under `hole` [F, H, W] replaced by a fill from their surroundings.

    Low frequency on purpose: these pixels are regenerated and never shown,
    and the fill only has to stop a kept token's encoder seeing the subject.
    Each frame is filled alone. Outside the hole nothing changes.
    """
    out = pixels.clone()
    for i in range(0, pixels.shape[0], CHUNK):
        h = (hole[i:i + CHUNK] > 0.5).to(pixels.dtype).unsqueeze(1)
        if not bool(h.any()):
            continue
        img = pixels[i:i + CHUNK].movedim(-1, 1)
        fill = _push_pull(img * (1.0 - h), 1.0 - h)
        out[i:i + CHUNK] = (img * (1.0 - h) + fill * h).movedim(1, -1)
    return out


def select_part(subject: torch.Tensor, part: torch.Tensor, margin: int = PART_MARGIN) -> torch.Tensor:
    """The part of a subject: `part` where it lies on `subject` widened by `margin` pixels. [N, H, W] each."""
    return ((part > 0.5) & (grow(subject.to(torch.float32), margin) > 0.5)).to(torch.float32)


def part_bottom(part: torch.Tensor) -> torch.Tensor:
    """Per frame, the lowest row a part reaches, or -1 where the frame has none. [N] long."""
    rows = (part > 0.5).any(dim=-1)                       # [N, H]
    index = torch.arange(rows.shape[-1], device=part.device)
    return torch.where(rows, index, torch.full_like(index, -1)).amax(dim=-1)


def carry_missing(bottoms: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Give each frame with no part (-1) the nearest found frame's value; (filled, how many were carried)."""
    found = (bottoms >= 0).nonzero().flatten()
    missing = (bottoms < 0).nonzero().flatten()
    if not found.numel() or not missing.numel():
        return bottoms, 0
    nearest = found[(missing[:, None] - found[None, :]).abs().argmin(dim=1)]
    out = bottoms.clone()
    out[missing] = bottoms[nearest]
    return out, int(missing.numel())


def above(subject: torch.Tensor, bottoms: torch.Tensor) -> torch.Tensor:
    """The subject's mask down to row `bottoms[n]` inclusive, per frame. [N, H, W]."""
    rows = torch.arange(subject.shape[-2], device=subject.device)[None, :, None]
    return ((subject > 0.5) & (rows <= bottoms[:, None, None])).to(torch.float32)


def window_frames(source: dict, first_frame: int, frames: int, width: int, height: int):
    """One window of a source on the render canvas: its fitted frames, its fitted mask, frames held.

    The source can run out inside a loop's last window: the missing frames
    repeat the last one with nothing masked (`window` says why). A window that
    starts past the source's end is refused.
    """
    have = int(source["frames"].shape[0])
    first_frame, frames = int(first_frame), int(frames)
    if first_frame >= have:
        raise ValueError(
            f"the source video has {have} frames and this window starts at frame {first_frame}: "
            "load more of it (the loader's frame cap), or shorten the run")
    pixels = fit_frames(source["frames"][first_frame:first_frame + frames], width, height)
    mask = fit_mask(source["mask"][first_frame:first_frame + frames], width, height)
    short = frames - int(pixels.shape[0])
    if short > 0:
        pixels = torch.cat([pixels, pixels[-1:].expand(short, -1, -1, -1)], dim=0)
        mask = torch.cat([mask, torch.zeros((short,) + tuple(mask.shape[1:]), dtype=mask.dtype, device=mask.device)], dim=0)
    return pixels, mask, short


def motion_reference(pixels: torch.Tensor, mask: torch.Tensor, mode: str, short_edge: int, margin: int):
    """The window as the model is shown it as a video reference, [F, h, w, 3], or None for `none`.

    `subject only` keeps the pixels under the mask widened by `margin` and sets
    the rest to mid grey, so the reference carries how the subject moves and
    nothing of the scene the kept rows already hold. `whole frame` keeps the
    window as it is. Either is scaled so its shorter side is `short_edge`,
    rounded to the canvas multiple with the aspect kept; the reference
    compiler never enlarges a video, so this is what sets its cost.
    """
    if mode == MOTION_NONE:
        return None
    if mode not in (MOTION_SUBJECT, MOTION_FRAME):
        raise ValueError(f"unknown motion_reference {mode!r}; one of {[MOTION_NONE, MOTION_SUBJECT, MOTION_FRAME]}")
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE  # core's constant; imported here so a check needs no server
    n, h, w = int(pixels.shape[0]), int(pixels.shape[1]), int(pixels.shape[2])
    scale = float(short_edge) / float(min(h, w))
    th = max(CANVAS_MULTIPLE, int(round(h * scale / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    tw = max(CANVAS_MULTIPLE, int(round(w * scale / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    keep = grow(mask.to(torch.float32), int(margin)) if mode == MOTION_SUBJECT else None
    out = []
    for i in range(0, n, CHUNK):
        chunk = pixels[i:i + CHUNK, ..., :3].to(torch.float32)
        if keep is not None:
            m = (keep[i:i + CHUNK] > 0.5).to(chunk.dtype).unsqueeze(-1)
            chunk = chunk * m + 0.5 * (1.0 - m)
        if (th, tw) != (h, w):
            chunk = F.interpolate(chunk.movedim(-1, 1), size=(th, tw), mode="bilinear",
                                  align_corners=False, antialias=True).movedim(1, -1)
        out.append(chunk.clamp(0.0, 1.0))
    return torch.cat(out, dim=0)


def window(source: dict, first_frame: int, frames: int, width: int, height: int,
           latent_t: int, lat_h: int, lat_w: int):
    """One window of a source: its fitted frames, the frames to encode, its token mask, its fitted mask, frames held.

    The frames to encode are the fitted frames themselves, or with `paint_out`
    a copy with the subject filled in: the mask widened by half of
    `grow_pixels`, which takes the soft edge SAM leaves on a fast limb and
    keeps the other half of the margin real background for the kept tokens.
    That hole is inside the regenerated tokens, so no filled pixel is shown.

    A loop's last window ends at or past the end of its track
    (`loop_plan.py`, "Lengths"), so the source can run out inside it. The
    missing frames repeat the last one with nothing masked: they are past the
    track, the join cuts them, and a held plate is what a frozen row should see.
    A window that starts past the source's end is refused; that is a source
    that does not belong to this track.
    """
    pixels, mask, short = window_frames(source, first_frame, frames, width, height)
    tokens = token_mask(grow(mask, source["grow_pixels"]), latent_t, lat_h, lat_w)
    encode = pixels
    if source.get("paint_out"):
        encode = fill_subject(pixels, grow(mask, int(source["grow_pixels"]) // 2))
    return pixels, encode, tokens, mask, short


def detect_part(segmenter, segmenter_clip, frames: torch.Tensor, where: torch.Tensor,
                phrases: tuple[str, ...], threshold: float = PART_THRESHOLD) -> torch.Tensor:
    """The union of core's SAM 3 detections of `phrases` on frames `where`, one mask each, [len(where), H, W]."""
    from comfy_extras.nodes_sam3 import SAM3_Detect  # core's node; imported here so a check needs no SAM
    parts = []
    for i in range(0, where.shape[0], CHUNK):
        chunk = frames[where[i:i + CHUNK]]
        union = torch.zeros(tuple(chunk.shape[:3]), dtype=torch.float32)
        for phrase in phrases:
            cond = segmenter_clip.encode_from_tokens_scheduled(segmenter_clip.tokenize(phrase))
            out = SAM3_Detect.execute(segmenter, chunk, conditioning=cond, threshold=float(threshold))
            union = torch.maximum(union, getattr(out, "args", out)[0].to(union))
        parts.append(union)
    return torch.cat(parts, dim=0)


class MiniMaxH3MaskedSource(io.ComfyNode):
    #: Part of a kept mask's key (`mask_store.py`). Raise it when a change
    #: would give a different mask from the same inputs and settings: that is
    #: `_settle_mask` and what it calls. The grow, the feather and the
    #: composite act after the mask and do not count.
    MASK_VERSION = 1

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaskedSource",
            display_name="MiniMax H3 Masked Source (video to video)",
            category="model/latent/minimax",
            description=(
                "A source video and a mask over the subject to replace. Wire it into the song "
                "node's `source`: each window then starts from the source's own frames, "
                "regenerates only the masked subject, and puts the source's pixels back "
                "everywhere else. The mask is one frame per source frame, 1 on the subject, "
                "as SAM3 Track to Mask gives it."),
            inputs=[
                io.Image.Input("frames", tooltip="The source video's frames at 24 fps, from its start."),
                # lazy since 2026-10-04: not asked for when a kept mask matches (`check_lazy_status`)
                io.Mask.Input("mask", lazy=True, tooltip="One mask per frame, 1 on the subject to replace."),
                io.Int.Input("grow_pixels", default=GROW_PIXELS, min=0, max=512,
                             tooltip=("How far the mask is widened before it reaches the model, in pixels of "
                                      "the render canvas. Raise it when the replacement is cut off at its "
                                      "edge. The hole's shape is all that tells the model where the original "
                                      "stood, so a wider one lets the new subject stand somewhere else and "
                                      "uncover what the original hid.")),
                io.Int.Input("feather_pixels", default=8, min=0, max=128,
                             tooltip=("Width of the blend between the regenerated region and the source's "
                                      "own pixels, to each side of the boundary. Raise it if the boundary "
                                      "shows; it cannot exceed grow_pixels.")),
                # appended 2026-10-04
                io.Combo.Input("replace", options=[REPLACE_WHOLE, REPLACE_PART], default=REPLACE_WHOLE,
                               tooltip=("What is regenerated. `whole subject` replaces the person, clothes "
                                        "and movement included, so the prompt has to say what they do. "
                                        "`head and hair` replaces the subject from the top of the head down "
                                        "to where the hair ends and keeps the rest of the body, its clothes "
                                        "and its movement; it needs `segmenter` and `segmenter_clip`, and "
                                        "costs a detector pass per phrase on each frame the subject is in.")),
                io.Boolean.Input("paint_out", default=False,
                                 tooltip=("Fill the subject in with its surroundings before the source is "
                                          "encoded, so the kept picture around the mask carries no trace of "
                                          "them. Turn it on if a faint remnant of the original shows beside "
                                          "the new subject. Costs a fill per frame, no sampling time.")),
                io.Model.Input("segmenter", optional=True, lazy=True,
                               tooltip="The SAM 3 model that tracked the mask. Read only for `head and hair`."),
                io.Clip.Input("segmenter_clip", optional=True, lazy=True,
                              tooltip="The SAM 3 checkpoint's text encoder. Read only for `head and hair`."),
                io.String.Input("part_phrases", default=PART_PHRASES,
                                tooltip=("What SAM 3 is asked to find on the subject for `head and hair`, "
                                         "separated by commas; everything found is used together. Change "
                                         "it when the `mask` output shows the wrong region. Each phrase "
                                         "costs one more detector pass.")),
                io.Float.Input("part_threshold", default=PART_THRESHOLD, min=0.0, max=1.0, step=0.01,
                               tooltip=("How sure SAM 3 has to be before a detection of a phrase counts, for "
                                        "`head and hair`. Lower it when the log says many frames were "
                                        "carried from a neighbour; raise it when the `mask` output takes "
                                        "something that is not the part.")),
                io.Int.Input("part_margin", default=PART_MARGIN, min=0, max=256,
                             tooltip=("How far past the subject's own mask a detected part may reach and "
                                      "still count as the subject's, in pixels of the source frame. Raise it "
                                      "when the `mask` output clips the part at the subject's edge; lower it "
                                      "when it takes a neighbour's.")),
                io.Combo.Input("composite", options=[COMPOSITE_REGION, COMPOSITE_CHANGED], default=COMPOSITE_CHANGED,
                               tooltip=("What is kept from the render. `whole region` keeps everything that was "
                                        "regenerated, margin included. `only what changed` keeps the render "
                                        "where it differs from the source (the new subject, and where the old "
                                        "one stood) and puts the source back where the model only repainted "
                                        "the background: choose it when the area around the subject flickers "
                                        "or looks cut out. Costs a comparison per frame, no model.")),
                io.Float.Input("change_threshold", default=CHANGE_THRESHOLD, min=0.0, max=1.0, step=0.005,
                               tooltip=("For `only what changed`: how different the render must be from the "
                                        "source to be kept, as a fraction of full brightness. Lower it if parts "
                                        "of the new subject go missing; raise it if flicker around the subject "
                                        "remains.")),
                # appended 2026-10-04 (`mask_store.py`)
                io.Boolean.Input("reuse_mask", default=True, optional=True,
                                 tooltip=("On (default): the finished mask is kept on disk, and a later run with "
                                          "the same video and the same mask settings uses it without tracking "
                                          "again, also after a restart. Any change to the video or to a setting "
                                          "that affects the mask tracks afresh.\n\n"
                                          "Off: track every time and keep nothing. Costs the tracker, and the "
                                          "part detection for `head and hair`, on every run after a restart.")),
                # appended 2026-10-05, optional so a saved graph keeps running. The
                # defaults are the shipped render as it was; `subject only` with the
                # VAE copy off is the arm the masking board calls route 1. Read by
                # the song node, which builds the reference per window.
                io.Combo.Input("motion_reference", options=[MOTION_NONE, MOTION_SUBJECT, MOTION_FRAME],
                               default=MOTION_NONE, optional=True,
                               tooltip=("Also show the model the original's movement, as a video reference the "
                                        "prompt names as <Video 1>.\n\n"
                                        "none (default): the model sees the still and the prompt only.\n\n"
                                        "subject only: the source window with everything outside the subject "
                                        "grey, so the model sees how the subject moves and nothing of the "
                                        "scene.\n\n"
                                        "whole frame: the source window as it is.\n\n"
                                        "The prompt has to say what the video provides, for example that the "
                                        "subject's motion and timing come from <Video 1>. Costs text-encoder "
                                        "tokens on every window, and with motion_vae on, rows on every "
                                        "sampling step.")),
                io.Int.Input("motion_short_edge", default=MOTION_SHORT_EDGE, min=32, max=1024, step=32, optional=True,
                             tooltip=("Shorter side, in pixels, of the motion reference the model is shown, "
                                      "rounded to 32. Smaller is cheaper; raise it if the model cannot make "
                                      "out the subject.")),
                io.Boolean.Input("motion_vae", default=False, optional=True,
                                 tooltip=("Off (default): the motion reference reaches the model through the "
                                          "text encoder only, at two frames per second. Cheap.\n\n"
                                          "On: the video model also gets its own copy, which costs rows on "
                                          "every sampling step and may not fit the card at a long window. "
                                          "Read the song node's report before queueing.")),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[H3MaskedSource.Output(display_name="source"),
                     io.Mask.Output(display_name="mask",
                                    tooltip=("The mask this node used, one per source frame, before grow_pixels: "
                                             "preview it to see what will be replaced."))],
        )

    @classmethod
    def _mask_key(cls, frames, reuse_mask):
        """The key this node's mask is kept under, or None: turned off, or no queued prompt to read (a direct call)."""
        hidden = getattr(cls, "hidden", None)
        prompt, node_id = getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None)
        if not reuse_mask or frames is None or not isinstance(prompt, dict) or node_id is None:
            return None
        from . import mask_store
        return mask_store.mask_key(prompt, node_id, frames, skip=MASK_KEY_SKIP)

    @classmethod
    def check_lazy_status(cls, frames=None, replace=REPLACE_WHOLE, reuse_mask=True, **kwargs):
        # None is a connected input core has not run yet; an unconnected
        # optional input is absent. `frames` is not lazy, so it is here.
        wanted = LAZY_FOR_MASK if replace == REPLACE_PART else LAZY_FOR_MASK[:1]
        missing = [name for name in wanted if name in kwargs and kwargs[name] is None]
        if not missing:
            return []
        key = cls._mask_key(frames, reuse_mask)
        if key is not None:
            from . import mask_store
            if mask_store.has(key, tuple(frames.shape[:3])):
                return []           # a kept mask matches: the tracker and the detector do not run
        return missing

    @classmethod
    # `mask` has no default: it is required in the schema, and core hands a lazy input it was not asked to run
    # as None, which is what a hit on a kept mask looks like here.
    def execute(cls, frames, mask, grow_pixels=GROW_PIXELS, feather_pixels=8, replace=REPLACE_WHOLE, paint_out=False,
                segmenter=None, segmenter_clip=None, part_phrases=PART_PHRASES,
                part_threshold=PART_THRESHOLD, part_margin=PART_MARGIN, composite=COMPOSITE_CHANGED,
                change_threshold=CHANGE_THRESHOLD, reuse_mask=True, motion_reference=MOTION_NONE,
                motion_short_edge=MOTION_SHORT_EDGE, motion_vae=False) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError(f"frames must be [N, H, W, C]; got {tuple(frames.shape)}")
        if int(feather_pixels) > int(grow_pixels):
            raise ValueError(
                f"feather_pixels {int(feather_pixels)} is wider than grow_pixels {int(grow_pixels)}: the blend "
                "would reach the subject's own pixels and bring the original back at its edge")
        if replace not in (REPLACE_WHOLE, REPLACE_PART):
            raise ValueError(f"unknown replace {replace!r}; one of {[REPLACE_WHOLE, REPLACE_PART]}")
        if composite not in (COMPOSITE_REGION, COMPOSITE_CHANGED):
            raise ValueError(f"unknown composite {composite!r}; one of {[COMPOSITE_REGION, COMPOSITE_CHANGED]}")
        if motion_reference not in (MOTION_NONE, MOTION_SUBJECT, MOTION_FRAME):
            raise ValueError(f"unknown motion_reference {motion_reference!r}; one of {[MOTION_NONE, MOTION_SUBJECT, MOTION_FRAME]}")
        key = cls._mask_key(frames, reuse_mask)
        kept = None
        if key is not None:
            from . import mask_store
            kept = mask_store.load(key, tuple(frames.shape[:3]))
        if kept is not None:
            mask, note = kept, ", mask kept from an earlier run (nothing tracked)"
        else:
            if mask is None:
                # `check_lazy_status` found a kept mask and it did not read here: it has been removed
                raise ValueError(
                    "the mask kept for this video could not be read and has been removed: queue the "
                    "workflow again and it will be tracked afresh")
            mask, note = cls._settle_mask(frames, mask, replace, segmenter, segmenter_clip, part_phrases,
                                          part_threshold, part_margin)
            if key is not None:
                seconds = mask_store.save(key, mask)
                note += f", mask kept for the next run ({seconds:.0f} s to write)"
        covered = float((mask > 0.5).any(dim=0).float().mean())
        logger.info("[h3] MiniMaxH3MaskedSource: %d frames, replacing the %s%s, the mask touches %.1f%% of the "
                    "frame over the clip, grow %d px, feather %d px%s, composite keeps the %s", int(frames.shape[0]),
                    replace, note, 100.0 * covered, int(grow_pixels), int(feather_pixels),
                    ", subject painted out before the encode" if paint_out else "", composite)
        if motion_reference != MOTION_NONE:
            logger.info("[h3] MiniMaxH3MaskedSource: motion reference %s at a %d short edge, %s", motion_reference,
                        int(motion_short_edge), "with the video model's copy" if motion_vae else "text encoder only")
        return io.NodeOutput({"frames": frames, "mask": mask, "grow_pixels": int(grow_pixels),
                              "feather_pixels": int(feather_pixels), "paint_out": bool(paint_out),
                              "composite": composite, "change_threshold": float(change_threshold),
                              "motion_reference": motion_reference, "motion_short_edge": int(motion_short_edge),
                              "motion_vae": bool(motion_vae)},
                             mask.to(torch.float32))

    @classmethod
    def _settle_mask(cls, frames, mask, replace, segmenter, segmenter_clip, part_phrases, part_threshold,
                     part_margin):
        """The mask this node uses, from the tracked one: checked against the frames, and cut to the part for
        `head and hair`. Returns it with what the log line says about it."""
        if mask.ndim == 4 and int(mask.shape[-1]) == 1:
            mask = mask[..., 0]
        if mask.ndim != 3:
            raise ValueError(f"frames must be [N, H, W, C] and mask [N, H, W]; got {tuple(frames.shape)} and {tuple(mask.shape)}")
        if int(mask.shape[0]) != int(frames.shape[0]):
            raise ValueError(
                f"{int(frames.shape[0])} frames and {int(mask.shape[0])} masks: the mask must be tracked "
                "over the same frames the source node loaded")
        if tuple(mask.shape[1:]) != tuple(frames.shape[1:3]):
            raise ValueError(
                f"the mask is {int(mask.shape[2])}x{int(mask.shape[1])} and the frames are "
                f"{int(frames.shape[2])}x{int(frames.shape[1])}: each is cropped to the canvas by its own "
                "shape, so a mask of another shape would land shifted. Track the mask on these frames")
        note = ""
        if replace == REPLACE_PART:
            phrases = tuple(p.strip() for p in str(part_phrases).split(",") if p.strip())
            if not phrases:
                raise ValueError("part_phrases is empty: name what SAM 3 should find on the subject, e.g. `hair, head`")
            if segmenter is None or segmenter_clip is None:
                raise ValueError(
                    f"replace `{replace}` finds the part with SAM 3: wire the SAM 3 checkpoint's model into "
                    "`segmenter` and its text encoder into `segmenter_clip`")
            where = (mask > 0.5).flatten(1).any(dim=1).nonzero().flatten()
            region = torch.zeros_like(mask, dtype=torch.float32)
            carried = 0
            if where.numel():
                subject = mask[where].to(torch.float32)
                found = select_part(subject, detect_part(segmenter, segmenter_clip, frames, where, phrases,
                                                         part_threshold).to(mask.device), int(part_margin))
                bottoms, carried = carry_missing(part_bottom(found))
                if bool((bottoms < 0).all()):
                    raise ValueError(
                        f"SAM 3 found none of {list(phrases)} on the subject in any frame: change `part_phrases`, "
                        "or set `replace` to whole subject")
                region[where] = above(subject, bottoms)
            note = (f", from {list(phrases)} at threshold {float(part_threshold):g}, found on {int(where.numel()) - carried} of the {int(where.numel())} "
                    f"frames the subject is in" + (f" and carried from the nearest frame on {carried}" if carried else ""))
            mask = region
        return mask, note
