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


def grow(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Dilate a [N, H, W] mask by `pixels` in every direction (a square max-pool, one axis at a time)."""
    p = int(pixels)  # zero is a kernel of one, which returns the mask
    parts = []
    for i in range(0, mask.shape[0], CHUNK):
        m = F.max_pool2d(mask[i:i + CHUNK].unsqueeze(1), (1, 2 * p + 1), stride=1, padding=(0, p))
        parts.append(F.max_pool2d(m, (2 * p + 1, 1), stride=1, padding=(p, 0))[:, 0])
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


def window(source: dict, first_frame: int, frames: int, width: int, height: int,
           latent_t: int, lat_h: int, lat_w: int):
    """One window of a source: its fitted frames [frames, height, width, 3], its token mask, frames padded.

    A loop's last window ends at or past the end of its track
    (`loop_plan.py`, "Lengths"), so the source can run out inside it. The
    missing frames repeat the last one with nothing masked: they are past the
    track, the join cuts them, and a held plate is what a frozen row should see.
    A window that starts past the source's end is refused; that is a source
    that does not belong to this track.
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
    tokens = token_mask(grow(mask, source["grow_pixels"]), latent_t, lat_h, lat_w)
    return pixels, tokens, short


class MiniMaxH3MaskedSource(io.ComfyNode):
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
                io.Mask.Input("mask", tooltip="One mask per frame, 1 on the subject to replace."),
                io.Int.Input("grow_pixels", default=32, min=0, max=512,
                             tooltip=("How far the mask is widened before it reaches the model, in pixels of "
                                      "the render canvas. Room for a replacement with a different outline, "
                                      "and the margin the feather blends inside of.")),
                io.Int.Input("feather_pixels", default=8, min=0, max=128,
                             tooltip=("Width of the blend between the regenerated region and the source's "
                                      "own pixels, to each side of the boundary. Keep it below grow_pixels.")),
            ],
            outputs=[H3MaskedSource.Output(display_name="source")],
        )

    @classmethod
    def execute(cls, frames, mask, grow_pixels=32, feather_pixels=8) -> io.NodeOutput:
        if mask.ndim == 4 and int(mask.shape[-1]) == 1:
            mask = mask[..., 0]
        if frames.ndim != 4 or mask.ndim != 3:
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
        if int(feather_pixels) > int(grow_pixels):
            raise ValueError(
                f"feather_pixels {int(feather_pixels)} is wider than grow_pixels {int(grow_pixels)}: the blend "
                "would reach the subject's own pixels and bring the original back at its edge")
        covered = float((mask > 0.5).any(dim=0).float().mean())
        logger.info("[h3] MiniMaxH3MaskedSource: %d frames, the mask touches %.1f%% of the frame over the clip, "
                    "grow %d px, feather %d px", int(frames.shape[0]), 100.0 * covered,
                    int(grow_pixels), int(feather_pixels))
        return io.NodeOutput({"frames": frames, "mask": mask, "grow_pixels": int(grow_pixels),
                              "feather_pixels": int(feather_pixels)})
