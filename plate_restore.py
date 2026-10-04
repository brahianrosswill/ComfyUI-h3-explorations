"""Put the kept parts of a masked latent back between two samplers.

A masked H3 render keeps part of a latent (a source video outside one
subject, a frozen audio track) and regenerates the rest. Core does that with
a `noise_mask`: every step the kept rows are re-injected from the sampler's
`latent_image` and the step's output for them is overwritten with it
(`comfy/samplers.py::KSamplerX0Inpaint`,
`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`). `latent_image` is
the plate.

**What goes wrong with two samplers.** A split graph (the daily PDD8 then
FlashGen finish graphs) hands the first sampler's output to a second sampler
with no added noise. That output is the sampler's state divided by
`1 - sigma` (`comfy/model_sampling.py::CONST.inverse_noise_scaling`), which
is what makes the second sampler resume exactly where the first stopped. In
a kept row the state is the plate plus the noise still on it, so the second
sampler's `latent_image` is a noisy plate, and it pins the kept rows to that
for the rest of the render. Nobody has to wire the mask for this to happen:
`SamplerCustomAdvanced` returns a copy of the latent dict it was given, mask
included.

**The fix is one blend.** Per stream, `mask * first + (1 - mask) * plate`:
the first sampler's output where the model is generating, the plate where it
is kept. The generating rows resume exactly as before; the kept rows are
clean again, the model is shown the clean plate, and the render ends on it.

**Only for a mask of zeros and ones, uniform per patch.** A row kept in part
(a mask value strictly between) needs two different things from the second
sampler's one `latent_image`: its own half-denoised state to resume from, and
the clean plate to blend toward. No latent holds both, so this node refuses
such a mask and names the stream. The same goes for a mask that keeps one
latent cell of a 2x2 patch the model is regenerating: core then feeds the
model that cell's sampler state, not the plate. `video_mask.token_mask`
writes masks that pass both conditions; an `audio_mask` above zero on the
freeze nodes does not.

One thing differs from a single sampler, and it is small: core adds
`1 - VISUAL_COND_TIMESTEP` of the sampler's noise to a kept video row before
the model sees it, and the second sampler's noise is zero. The kept rows the
model sees in the second half are that much cleaner than in a one-sampler
run. `bench/check_plate_restore.py` bounds what it does to the generated rows
on a stub model.

Measured through core's own sampler classes with a stub model, on the CPU
(`bench/check_plate_restore.py`). Not yet run on H3.

Nothing here patches core.
"""

from __future__ import annotations

import inspect
import logging

import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui

import comfy.nested_tensor
import comfy.sampler_helpers
from comfy.ldm.minimax.model import patchify_video

logger = logging.getLogger(__name__)

#: The DiT's spatial patch, in latent cells. Inherited: the default of core's
#: own patchifier, read here so a change in core moves this with it.
PATCH_H, PATCH_W = inspect.signature(patchify_video).parameters["patch_size"].default[1:]
#: A mask value at or above this is a fully generating row to core
#: (`comfy/ldm/minimax/model.py::mask_row_values`). Inherited.
OPEN_ABOVE = 1.0 - 1e-3
#: A mask value at or below this is a kept row. Reasoned: float noise from a
#: resize of an exact zero, far under the smallest step a mask widget has.
KEPT_BELOW = 1e-6

STREAM_NAMES = ("video", "audio")


def _streams(samples) -> list[torch.Tensor]:
    return list(samples.unbind()) if getattr(samples, "is_nested", False) else [samples]


def stream_masks(noise_mask, shapes) -> list[torch.Tensor]:
    """Each stream's mask as the sampler will prepare it.

    `comfy/samplers.py::CFGGuider.sample`: a nested mask is taken apart, a
    stream with no mask of its own generates everywhere, and each mask is
    resized to its stream.
    """
    given = _streams(noise_mask)[:len(shapes)]
    out = []
    for i, shape in enumerate(shapes):
        if i < len(given):
            out.append(comfy.sampler_helpers.prepare_mask(given[i], shape, "cpu").float())
        else:
            out.append(torch.ones(shape))
    return out


def _uniform_per_patch(mask: torch.Tensor) -> bool:
    """True when every 2x2 patch of a [B, C, T, H, W] mask, and every channel, holds one value."""
    if not bool((mask == mask[:, :1]).all()):
        return False
    m = mask[:, 0]
    h, w = m.shape[-2:]
    m = F.pad(m.reshape((-1, 1, h, w)), (0, -w % PATCH_W, 0, -h % PATCH_H), mode="replicate")
    hi = F.max_pool2d(m, (PATCH_H, PATCH_W))
    lo = -F.max_pool2d(-m, (PATCH_H, PATCH_W))
    return bool((hi == lo).all())


def binary_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    """The mask as zeros and ones, or a refusal that says why a blend cannot be right."""
    kept, opened = mask <= KEPT_BELOW, mask >= OPEN_ABOVE
    if not bool((kept | opened).all()):
        between = mask[~(kept | opened)]
        raise ValueError(
            f"the {name} mask holds values between 0 and 1 (from {float(between.min()):g} to "
            f"{float(between.max()):g}). A partly kept row cannot be carried across two samplers "
            "by restoring a latent: it needs its own half-denoised state and the clean plate at "
            "once. Use a mask of 0 and 1 for this stream, or render it with one sampler.")
    if mask.ndim == 5 and not _uniform_per_patch(mask):
        raise ValueError(
            f"the {name} mask keeps part of a {PATCH_H}x{PATCH_W} patch the model regenerates. "
            "The model works on whole patches, so such a cell cannot be restored correctly. "
            "MiniMax H3 Masked Source writes masks on whole patches.")
    if mask.ndim == 4 and not bool((mask == mask[:, :1]).all()):
        raise ValueError(f"the {name} mask differs between channels; core keeps or regenerates an audio step whole")
    return opened.to(mask.dtype)


def restore(first, plate, noise_mask):
    """`mask * first + (1 - mask) * plate` per stream, and one report line per stream."""
    a, b = _streams(first), _streams(plate)
    if [tuple(t.shape) for t in a] != [tuple(t.shape) for t in b]:
        raise ValueError(
            f"the two latents differ in shape: {[tuple(t.shape) for t in a]} from the first sampler, "
            f"{[tuple(t.shape) for t in b]} as the plate. The plate is the latent the first sampler was given.")
    masks = stream_masks(noise_mask, [t.shape for t in a])
    out, lines = [], []
    for i, (x, p, m) in enumerate(zip(a, b, masks)):
        name = STREAM_NAMES[i] if i < len(STREAM_NAMES) else f"stream {i + 1}"
        m = binary_mask(m, name).to(device=x.device, dtype=x.dtype)
        out.append(x * m + p.to(device=x.device, dtype=x.dtype) * (1.0 - m))
        kept = 100.0 * float(1.0 - m.mean())
        lines.append(f"{name}: {kept:.1f}% restored from the plate" if kept else f"{name}: nothing kept, left as sampled")
    if getattr(first, "is_nested", False):
        return comfy.nested_tensor.NestedTensor(tuple(out)), lines
    return out[0], lines


class MiniMaxH3RestorePlate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RestorePlate",
            display_name="MiniMax H3 Restore Plate (between two samplers)",
            category="model/latent/minimax",
            description=(
                "For a masked render split across two samplers. Put it between "
                "them: it sets the kept (unmasked) parts of the latent back to "
                "what they were before the first sampler, so the second sampler "
                "keeps the real picture and audio instead of a noisy copy. The "
                "parts being generated pass through unchanged.\n\n"
                "The mask must be all 0 and 1. A partly kept stream, such as an "
                "audio_mask above 0, cannot be split across two samplers."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="The first sampler's output."),
                io.Latent.Input("plate", tooltip="The latent you gave the first sampler."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, plate) -> io.NodeOutput:
        # The first sampler's output still carries the mask it was given; the
        # plate's is the same one, and the fallback if a node in between dropped it.
        noise_mask = latent.get("noise_mask", plate.get("noise_mask"))
        out = dict(latent)
        if noise_mask is None:
            report = "no noise_mask on either latent: nothing is kept, so nothing was restored"
        else:
            out["samples"], lines = restore(latent["samples"], plate["samples"], noise_mask)
            out["noise_mask"] = noise_mask
            report = "\n".join(lines)
        logger.info("[h3] MiniMaxH3RestorePlate: %s", report.replace("\n", "; "))
        return io.NodeOutput(out, ui=ui.PreviewText(report))
