"""VOID's inpainting conditioning the way upstream's own inference builds it.

For the masked lane's clean plate (`docs/wiki/masked_v2v.md`, "What the body
does to the room"). Core ships `VOIDInpaintConditioning`
(`comfy_extras/nodes_void.py`); this node has the same inputs and outputs and
differs in two things, both read on 2026-10-05 from upstream's inference code
(`netflix/void-model`, `main` at e3914f8f551d: `predict_v2v.py`, its config
`quadmask_cogvideox.py`, and the pipeline's mask handling) and recorded with
the arms that led here in `bench/results/2026-10-05_void_plate_turn.md`.

1. **The pixel range the mask is encoded at.** Both sides hand `1 - mask` to
   the VAE as the mask channels. Upstream's mask processor does not normalise
   (`VaeImageProcessor(do_normalize=False)`), so its VAE, whose pixels run
   from -1 to 1, is given the mask in 0 to 1: 0.0 on the object to remove,
   about 0.5 on the affected area, 1.0 on what is kept. Core's node calls
   `VAE.encode`, which scales its input by `x * 2 - 1`, so core's VAE is given
   -1.0, about 0.0 and 1.0: the affected area lands on upstream's value for
   the object. On the band clip that drew a flat panel in the shape of the
   affected region; with the mask's range corrected the panel went
   (`mask_image`).
2. **What the model is shown under the mask.** Upstream's shipped config has
   `zero_out_mask_region = False`: the model is conditioned on the whole
   video, the object included. Core's node multiplies the video by
   `1 - mask`, which blacks the object out and dims the affected area by
   half. `video_under_mask` offers upstream's two paths: the whole video (its
   config), and the object blacked out with the affected area left as it is
   (`zero_out_mask_region = True`).

Also upstream's, and done here: the mask is reduced to three levels after it
is resized (`trimask`), because upstream's script passes `use_trimask=True`
to its pipeline whatever the config says.

Not done here, and still different from upstream: upstream samples in windows
of `temporal_window_size` frames, and core's decode returns three more frames
than it was given (input frame j is output frame j + 3). The second is the
decoder's, not the conditioning's; whoever composites a plate applies it.

The quadmask's convention is core's and upstream's: 1.0 is the object to
remove, about 0.5 the area its removal affects, 0.0 what is kept.

Nothing here patches core. It calls the VAE core loaded and uses core's own
length rule.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

import comfy.model_management
import comfy.utils
import node_helpers

logger = logging.getLogger(__name__)

#: The `video_under_mask` choices. The first is upstream's shipped config
#: (`zero_out_mask_region = False`), the second its other path. Inherited.
VIDEO_WHOLE = "the whole video"
VIDEO_OBJECT_BLACK = "the object blacked out"
#: Upstream's three levels and the thresholds between them, as its pipeline
#: quantises a mask in 0..1 (`use_trimask`). Inherited.
TRIMASK_OBJECT_ABOVE = 0.75
TRIMASK_KEPT_BELOW = 0.25
TRIMASK_AFFECTED = 127.0 / 255.0
#: Core's template values for the canvas and the length
#: (`utility_void_video_inpainting.json`), which are the model card's size. Inherited.
WIDTH, HEIGHT, LENGTH = 672, 384, 45


def trimask(mask: torch.Tensor) -> torch.Tensor:
    """A mask in 0..1 reduced to upstream's three levels: 1.0 object, 127/255 affected, 0.0 kept."""
    out = torch.where(mask > TRIMASK_OBJECT_ABOVE, torch.ones_like(mask), mask)
    out = torch.where((out <= TRIMASK_OBJECT_ABOVE) & (out >= TRIMASK_KEPT_BELOW), torch.full_like(out, TRIMASK_AFFECTED), out)
    return torch.where(out < TRIMASK_KEPT_BELOW, torch.zeros_like(out), out)


def mask_image(mask: torch.Tensor, process_input) -> torch.Tensor:
    """The picture to hand `VAE.encode` so that the encoder itself is given `1 - mask` in 0..1, as upstream's is.

    `process_input` is the VAE's own pixel scaling. It is `x * 2 - 1` for the VAE core loads for VOID, so the
    picture is `1 - mask / 2`; any other scaling is refused, since the inverse written here would be wrong for it.
    """
    probe = process_input(torch.tensor([0.0, 0.5, 1.0]))
    if not torch.allclose(probe.to(torch.float32), torch.tensor([-1.0, 0.0, 1.0])):
        raise ValueError(
            "this VAE does not scale its pixels by x * 2 - 1, which is what this node undoes to give the encoder "
            f"upstream's mask range; it gave {probe.tolist()} for 0, 0.5 and 1")
    wanted = 1.0 - mask                 # what upstream's encoder is given
    return (wanted + 1.0) / 2.0         # what core's scaling turns into that


def video_shown(video: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    """The conditioning video, [T, H, W, 3] in 0..1, for a mask [T, H, W] already on upstream's three levels."""
    if how == VIDEO_WHOLE:
        return video
    if how == VIDEO_OBJECT_BLACK:
        # upstream: init_video * (mask < 0.75) + (-1) * (mask > 0.75), in -1..1; black is 0 here
        return video * (mask < TRIMASK_OBJECT_ABOVE).to(video.dtype).unsqueeze(-1)
    raise ValueError(f"unknown video_under_mask {how!r}; one of {[VIDEO_WHOLE, VIDEO_OBJECT_BLACK]}")


def conditioning(vae, video: torch.Tensor, quadmask: torch.Tensor, width: int, height: int, length: int,
                 video_under_mask: str = VIDEO_WHOLE) -> tuple[torch.Tensor, int]:
    """The 32-channel concat conditioning and the length used: the mask's latents, then the video's, as core orders them."""
    from comfy_extras.nodes_void import TEMPORAL_COMPRESSION, _valid_void_length  # core's own length rule
    if quadmask.ndim == 4 and int(quadmask.shape[-1]) == 1:
        quadmask = quadmask[..., 0]
    if quadmask.ndim != 3:
        raise ValueError(f"quadmask must be [T, H, W]; got {tuple(quadmask.shape)}")
    if int(quadmask.shape[0]) < 1 or int(video.shape[0]) < 1:
        raise ValueError("video and quadmask need at least one frame each")
    length = _valid_void_length(min(int(length), int(video.shape[0]), int(quadmask.shape[0])))
    latent_t = ((length - 1) // TEMPORAL_COMPRESSION) + 1
    vid = comfy.utils.common_upscale(video[:length, ..., :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    mask = comfy.utils.common_upscale(quadmask[:length].unsqueeze(1).to(torch.float32), width, height, "bilinear", "center")[:, 0]
    mask = trimask(mask.clamp(0.0, 1.0))
    mask_latents = vae.encode(mask_image(mask, vae.process_input).unsqueeze(-1).expand(-1, -1, -1, 3))
    video_latents = vae.encode(video_shown(vid, mask, video_under_mask))

    def match(lat):
        if lat.shape[2] > latent_t:
            return lat[:, :, :latent_t]
        if lat.shape[2] < latent_t:
            return torch.cat([lat, lat[:, :, -1:].repeat(1, 1, latent_t - lat.shape[2], 1, 1)], dim=2)
        return lat

    return torch.cat([match(mask_latents), match(video_latents)], dim=1), length


class MiniMaxH3VoidConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VoidConditioning",
            display_name="MiniMax H3 VOID Conditioning (as upstream)",
            category="model/conditioning/void",
            description=(
                "Builds VOID's inpainting conditioning the way upstream's own inference does: the mask at "
                "upstream's levels and range, and the video shown whole. Use it in place of core's VOID "
                "Inpaint Conditioning, with the same wiring."),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae", tooltip="The CogVideoX VAE."),
                io.Image.Input("video", tooltip="The source frames."),
                io.Mask.Input("quadmask", tooltip=("One mask per frame: 1 on the object to remove, about 0.5 on the "
                                                   "area its removal affects (a shadow, a reflection), 0 on what "
                                                   "is kept. A plain mask of the object works.")),
                io.Int.Input("width", default=WIDTH, min=16, max=8192, step=8, tooltip="The width VOID works at."),
                io.Int.Input("height", default=HEIGHT, min=16, max=8192, step=8, tooltip="The height VOID works at."),
                io.Int.Input("length", default=LENGTH, min=1, max=8192,
                             tooltip=("How many frames to process. It is rounded down to a length VOID takes, "
                                      "and to the frames given.")),
                io.Combo.Input("video_under_mask", options=[VIDEO_WHOLE, VIDEO_OBJECT_BLACK], default=VIDEO_WHOLE,
                               tooltip=("What the model sees where the mask is. `the whole video` shows it the "
                                        "source untouched, the object included. `the object blacked out` hides "
                                        "the object and leaves the affected area as it is.")),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, video, quadmask, width=WIDTH, height=HEIGHT, length=LENGTH,
                video_under_mask=VIDEO_WHOLE) -> io.NodeOutput:
        concat, used = conditioning(vae, video, quadmask, int(width), int(height), int(length), video_under_mask)
        if used != int(length):
            logger.info("[h3] MiniMaxH3VoidConditioning: length %d used, of %d asked for", used, int(length))
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat})
        latent = torch.zeros([1, 16, concat.shape[2], int(height) // 8, int(width) // 8],
                             device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})
