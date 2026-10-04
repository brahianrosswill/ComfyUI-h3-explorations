#!/usr/bin/env python3
"""A masked render split across two samplers ends on the clean plate, and resumes where it stopped.

`plate_restore.py` is the module. A masked H3 render keeps part of a latent
by a `noise_mask`; split across two samplers as the daily graphs are, the
second one takes the first one's still-noisy output as the picture to keep.
Each item is one thing that has to hold for the node between them to be the
fix, or one way it could be wrong without anyone seeing.

1. **The core functions this leans on are the ones it was built against.**
   `KSamplerX0Inpaint.__call__`, `KSAMPLER.sample` and
   `MiniMaxH3.scale_latent_inpaint` are hashed against `PINNED`. A change in
   core fails here; the fix is to read the three again and re-run items 3 to
   5, never to move the pin alone.
2. **The mask reaches the second sampler unasked.** `SamplerCustomAdvanced`
   returns a copy of the latent dict it was given, so the node can read the
   mask off the first sampler's output and nobody has to rewire it.
3. **RED CONTROL: as wired, the kept rows end noisy** and the generated rows
   drift from a one-sampler render. If this ever passes without the node,
   core has changed and the node may no longer be needed.
4. **With the node, the kept rows end on the plate**, video and audio, and
   the model is shown the plate in the second half.
5. **With the node, the generated rows match a one-sampler render** to
   within `GENERATED_TOL`, with the red control far outside it. Not exact:
   see the module docstring on the noise core adds to a kept video row.
6. **What a blend cannot fix is refused, by name**: a mask value between 0
   and 1, and a mask that keeps part of a patch.
7. **A flat mask leaves the audio generating**, as core treats it, and no
   mask restores nothing. The node is in the pack's node list.

The sampler is core's own `KSAMPLER` with `sample_euler`, on the packed
video+audio latent, with `MiniMaxH3`'s own inpaint and audio-scale functions
bound to a small stand-in (no DiT is built). The model is a fixed linear map
of everything it is shown, so a wrong plate moves the generated rows too.
The lines around the sampler are `CFGGuider.sample` and `inner_sample`'s,
restated here because that class needs a loaded model.

No model, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_plate_restore.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # no CUDA context; the sibling checks do the same
import comfy.k_diffusion.sampling as k_sampling  # noqa: E402
import comfy.model_base  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.sampler_helpers  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP  # noqa: E402
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced  # noqa: E402

#: sha256 (first 16 hex) of the source of the core functions the node's
#: correctness rests on, read at the ComfyUI checkout of 2026-10-04.
PINNED = {
    "KSamplerX0Inpaint.__call__": "e66e8a287e7473aa",
    "KSAMPLER.sample": "228648800dca50b5",
    "MiniMaxH3.scale_latent_inpaint": "e9525d4819cc0c69",
}

#: Largest difference in a generated row, over the largest value there, allowed
#: between the split render with the node and a one-sampler render. Reasoned:
#: the only thing that differs is `1 - VISUAL_COND_TIMESTEP` of unit noise on
#: the kept video rows the model is shown, through a model of gain under one;
#: ten times that.
GENERATED_TOL = 10.0 * (1.0 - VISUAL_COND_TIMESTEP)

VIDEO_SHAPE = (1, 3, 4, 4, 6)    # [B, C, T, H, W], two patches high and three wide
AUDIO_SHAPE = (1, 2, 2, 5)       # [B, C, channels, T]
FULL = [1.0, 0.75, 0.5, 0.25, 0.0]
FIRST, SECOND = FULL[:3], FULL[2:]


def _load():
    spec = importlib.util.spec_from_file_location("_h3_plate_restore", REPO / "plate_restore.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3_plate_restore"] = module
    spec.loader.exec_module(module)
    return module


pr = _load()


def _sha(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


CORE = {
    "KSamplerX0Inpaint.__call__": comfy.samplers.KSamplerX0Inpaint.__call__,
    "KSAMPLER.sample": comfy.samplers.KSAMPLER.sample,
    "MiniMaxH3.scale_latent_inpaint": comfy.model_base.MiniMaxH3.scale_latent_inpaint,
}


class _Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass


class _H3:
    """`MiniMaxH3`'s own mask, inpaint and audio-scale functions on a stand-in with no DiT."""

    audio_scale = comfy.model_base.MiniMaxH3.audio_scale
    _scale_audio_slice = comfy.model_base.MiniMaxH3._scale_audio_slice
    _pool_masks_to_token_grid = comfy.model_base.MiniMaxH3._pool_masks_to_token_grid
    _token_grid_masks = comfy.model_base.MiniMaxH3._token_grid_masks
    scale_latent_inpaint = comfy.model_base.MiniMaxH3.scale_latent_inpaint

    def __init__(self):
        self.model_sampling = _Sampling()
        # an audio shift of its own, so the audio stream is carried scaled as on the real model
        self.model_sampling.set_parameters(shift=5.0, audio_shift=2.5)
        self.diffusion_model = types.SimpleNamespace(patch_size=(1, pr.PATCH_H, pr.PATCH_W))
        self.latent_shapes = None


class _Guider:
    """What `KSAMPLER.sample` is handed: an x0 model that reads everything it is shown."""

    cfg = 1.0

    def __init__(self, h3, size):
        self.inner_model = h3
        self.model_patcher = types.SimpleNamespace(model=object())
        g = torch.Generator().manual_seed(7)
        q, _ = torch.linalg.qr(torch.randn(size, size, generator=g))
        self.mix = 0.5 * q                       # gain 0.5 in every direction
        self.bias = torch.randn(size, generator=g)
        self.seen = []

    def __call__(self, x, sigma, model_options={}, seed=None):
        self.seen.append(x.clone())
        return (x.reshape(1, -1) @ self.mix + self.bias).reshape(x.shape)


def sample(noise, latent, mask, sigmas):
    """`CFGGuider.sample` and `inner_sample` around core's `KSAMPLER`, for nested (video, audio)."""
    h3 = _H3()
    latent_image, shapes = comfy.utils.pack_latents(list(latent.unbind()))
    packed_noise, _ = comfy.utils.pack_latents(list(noise.unbind()))
    denoise_mask = None
    if mask is not None:
        masks = pr.stream_masks(mask, shapes)
        denoise_mask, _ = comfy.utils.pack_latents(masks)
    h3.latent_shapes = shapes
    if torch.count_nonzero(latent_image) > 0:
        latent_image = h3._scale_audio_slice(latent_image, h3.audio_scale())           # process_latent_in
    guider = _Guider(h3, latent_image.shape[-1])
    sampler = comfy.samplers.KSAMPLER(k_sampling.sample_euler)
    samples = sampler.sample(guider, torch.tensor(sigmas), {"model_options": {}, "seed": 0}, None,
                             packed_noise, latent_image, denoise_mask, True)
    samples = h3._scale_audio_slice(samples.to(torch.float32), 1.0 / h3.audio_scale())  # process_latent_out
    out = comfy.nested_tensor.NestedTensor(tuple(comfy.utils.unpack_latents(samples, shapes)))
    return out, guider, shapes


def _case():
    g = torch.Generator().manual_seed(3)
    plate = comfy.nested_tensor.NestedTensor((torch.randn(VIDEO_SHAPE, generator=g),
                                              torch.randn(AUDIO_SHAPE, generator=g)))
    noise = comfy.nested_tensor.NestedTensor((torch.randn(VIDEO_SHAPE, generator=g),
                                              torch.randn(AUDIO_SHAPE, generator=g)))
    zero = comfy.nested_tensor.NestedTensor((torch.zeros(VIDEO_SHAPE), torch.zeros(AUDIO_SHAPE)))
    video_mask = torch.zeros((1, 1) + VIDEO_SHAPE[2:])
    video_mask[..., 2:, 2:] = 1.0                 # the lower right patches regenerate, on the patch grid
    video_mask[:, :, 0] = 0.0                     # and one latent step is kept whole
    audio_mask = torch.zeros((1, 1) + AUDIO_SHAPE[2:])   # the track is frozen
    mask = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return plate, noise, zero, mask


def _rel(a, b, where):
    return float((a - b)[where].abs().max() / b[where].abs().max())


def check_pins(problems):
    for name, fn in CORE.items():
        got = _sha(fn)
        if PINNED[name] != got:
            problems.append(f"{name} is not the source this node was built against (pinned {PINNED[name]}, now {got})")


def check_mask_rides(problems):
    source = inspect.getsource(SamplerCustomAdvanced.execute)
    if "out = latent.copy()" not in source:
        problems.append("SamplerCustomAdvanced no longer returns a copy of its latent dict; the mask may not reach the second sampler")


def check_two_samplers(problems):
    plate, noise, zero, mask = _case()
    one, _, shapes = sample(noise, plate, mask, FULL)
    first, _, _ = sample(noise, plate, mask, FIRST)
    wired, wired_guider, _ = sample(zero, first, mask, SECOND)
    restored, lines = pr.restore(first, plate, mask)
    fixed, fixed_guider, _ = sample(zero, restored, mask, SECOND)

    masks = pr.stream_masks(mask, shapes)
    keep_v, keep_a = masks[0] == 0, masks[1] == 0
    gen_v = ~keep_v
    pv, pa = plate.unbind()

    # the one-sampler render is the reference: it must itself end on the plate
    ov, oa = one.unbind()
    if not torch.allclose(ov[keep_v], pv[keep_v], atol=1e-5) or not torch.allclose(oa[keep_a], pa[keep_a], atol=1e-5):
        problems.append("one sampler does not end on the plate; the reference of this check is broken")

    wv, _wa = wired.unbind()
    wired_kept = _rel(wv, pv, keep_v)
    wired_gen = _rel(wv, ov, gen_v)
    if wired_kept < 100 * 1e-5:
        problems.append(f"RED CONTROL failed: as wired, the kept video rows end on the plate ({wired_kept:.1e})")
    if wired_gen < 10 * GENERATED_TOL:
        problems.append(f"RED CONTROL failed: as wired, the generated rows are within {wired_gen:.1e} of one sampler; "
                        "the check cannot tell a right plate from a wrong one")

    fv, fa = fixed.unbind()
    if not torch.allclose(fv[keep_v], pv[keep_v], atol=1e-5):
        problems.append(f"with the node, the kept video rows do not end on the plate ({_rel(fv, pv, keep_v):.1e})")
    if not torch.allclose(fa[keep_a], pa[keep_a], atol=1e-5):
        problems.append(f"with the node, the frozen audio does not end on the plate ({_rel(fa, pa, keep_a):.1e})")
    fixed_gen = _rel(fv, ov, gen_v)
    if fixed_gen > GENERATED_TOL:
        problems.append(f"with the node, the generated rows are {fixed_gen:.1e} from one sampler, over {GENERATED_TOL:.1e}")

    # what the model was shown in the second half: the plate at the cond strength, not a noisy one
    n_video = int(torch.tensor(shapes[0][1:]).prod())
    shown = fixed_guider.seen[0][..., :n_video].reshape(VIDEO_SHAPE)
    wrong = wired_guider.seen[0][..., :n_video].reshape(VIDEO_SHAPE)
    if not torch.allclose(shown[keep_v], VISUAL_COND_TIMESTEP * pv[keep_v], atol=1e-5):
        problems.append("with the node, the model is not shown the plate in the kept video rows")
    if torch.allclose(wrong[keep_v], VISUAL_COND_TIMESTEP * pv[keep_v], atol=1e-3):
        problems.append("RED CONTROL failed: as wired, the model is shown the clean plate")
    if not any("restored from the plate" in line for line in lines):
        problems.append(f"the node's report does not say what it restored: {lines}")
    return {"wired_kept": wired_kept, "wired_generated": wired_gen, "fixed_generated": fixed_gen}


def check_refusals(problems):
    plate, _noise, _zero, mask = _case()
    video_mask, audio_mask = mask.unbind()

    def refused(m, word):
        try:
            pr.restore(plate, plate, m)
        except ValueError as exc:
            return word in str(exc)
        return False

    partly = comfy.nested_tensor.NestedTensor((video_mask, torch.full_like(audio_mask, 0.25)))
    if not refused(partly, "audio"):
        problems.append("a partly kept audio stream (mask 0.25) is not refused by name")
    ragged = video_mask.clone()
    ragged[..., 3, 3] = 0.0                       # one cell of a regenerating patch kept
    if not refused(comfy.nested_tensor.NestedTensor((ragged, audio_mask)), "patch"):
        problems.append("a mask that keeps part of a patch is not refused")
    short = comfy.nested_tensor.NestedTensor((torch.zeros((1, 3, 2, 4, 6)), torch.zeros(AUDIO_SHAPE)))
    try:
        pr.restore(short, plate, mask)
        problems.append("two latents of different shape are not refused")
    except ValueError:
        pass


def check_flat_and_none(problems):
    plate, _noise, _zero, mask = _case()
    video_mask, _audio_mask = mask.unbind()
    first = comfy.nested_tensor.NestedTensor(tuple(t + 1.0 for t in plate.unbind()))
    out, _lines = pr.restore(first, plate, video_mask)       # a flat mask: core pads ones for the audio
    _ov, oa = out.unbind()
    if not torch.equal(oa, first.unbind()[1]):
        problems.append("a flat mask restored the audio; core lets a stream with no mask generate")
    node = pr.MiniMaxH3RestorePlate.execute({"samples": first}, {"samples": plate})
    node = getattr(node, "args", node)
    if node[0]["samples"] is not first:
        problems.append("with no mask on either latent the node changed the samples")
    with_mask = pr.MiniMaxH3RestorePlate.execute({"samples": first, "noise_mask": mask}, {"samples": plate})
    with_mask = getattr(with_mask, "args", with_mask)[0]
    wv, wa = with_mask["samples"].unbind()
    if "noise_mask" not in with_mask or not torch.equal(wa, plate.unbind()[1]):
        problems.append("the node did not restore the frozen audio or dropped the mask from its output")
    schema = pr.MiniMaxH3RestorePlate.define_schema()
    if [i.id for i in schema.inputs] != ["latent", "plate"] or any(not i.tooltip for i in schema.inputs):
        problems.append("the node's inputs are not `latent` and `plate`, each with a tooltip")
    listed = re.search(r"^\s+MiniMaxH3RestorePlate[,\]]", (REPO / "nodes.py").read_text(), re.M)
    if listed is None:
        problems.append("MiniMaxH3RestorePlate is not in the pack's node list (nodes.py)")


def main() -> int:
    problems: list[str] = []
    check_pins(problems)
    check_mask_rides(problems)
    numbers = check_two_samplers(problems) or {}
    check_refusals(problems)
    check_flat_and_none(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if numbers:
        print(f"      as wired: kept rows {numbers['wired_kept']:.1e} from the plate, generated rows "
              f"{numbers['wired_generated']:.1e} from one sampler; with the node: generated rows "
              f"{numbers['fixed_generated']:.1e} (bound {GENERATED_TOL:.1e})")
    if not problems:
        print("ok    a split masked render ends on the plate and resumes where it stopped; what a blend "
              "cannot fix is refused; the core functions it leans on are the pinned ones")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
