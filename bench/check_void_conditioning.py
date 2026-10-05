#!/usr/bin/env python3
"""Hold `void_conditioning.py` to upstream VOID's conditioning arithmetic, on a synthetic clip and a stand-in VAE.

The node exists because core's `VOIDInpaintConditioning` gives the VAE
something upstream's own inference does not, and a clean plate fails quietly
when it does: the render plays and the fill is wrong. The stand-in VAE records
what its encoder is given, after the VAE's own pixel scaling, so each case
compares that with upstream's arithmetic, written out here from
`netflix/void-model` at e3914f8f551d (`bench/results/2026-10-05_void_plate_turn.md`
has the reading). Core's own node, run on the same input, is the control.

1. **The encoder is given upstream's mask.** For a quadmask holding every
   level, the mask channels' encoder input is `1 - trimask(mask)` in 0..1:
   0.0 on the object, about 0.5 on the affected area, 1.0 on what is kept.
   **The control: core's node gives -1.0, 0.0 and 1.0**, so its affected area
   sits on upstream's value for the object. If core ever matches upstream,
   the control fails and this node has lost its first reason.
2. **The model is shown the whole video**, or, with the other choice, the
   object blacked out and the affected area untouched. **The control: core's
   node dims the affected area.**
3. **The outputs are core's**: mask latents then video latents, the same
   shapes as core's node on the same input, core's rule for the length.
4. **A VAE with another pixel scaling is refused**, since the inverse the
   node applies would be wrong for it; and an unknown choice is refused.
5. **The node is wired as core's**: the same inputs but for `batch_size`,
   plus `video_under_mask`, each with a tooltip where core's has one to give,
   and it is in the pack's node list.

No weights, no CUDA, no server, no network.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_void_conditioning.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from _lib import bootstrap, case, finish  # noqa: E402

bootstrap(cpu=True)

import torch  # noqa: E402


def _load():
    """`void_conditioning` as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.void_conditioning", REPO / "void_conditioning.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.void_conditioning"] = module
    spec.loader.exec_module(module)
    return module


vc = _load()

W, H, T = 64, 32, 13            # on the model's grid; 13 frames is a length VOID takes
# one column per level a quadmask can hold: object, overlap (core's preprocess level), affected, a faint value, kept
LEVELS = (1.0, 192.0 / 255.0, 0.5, 0.2, 0.0)
OBJECT, OVERLAP, AFFECTED, FAINT, KEPT = (slice(i * 12, i * 12 + 12) for i in range(5))


class StandInVAE:
    """Records what its encoder is given, after the pixel scaling core's VAE applies."""

    def __init__(self, scaling=lambda x: x * 2.0 - 1.0):
        self.process_input = scaling
        self.seen: list[torch.Tensor] = []

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.seen.append(self.process_input(image.to(torch.float32)).clone())
        t, h, w = image.shape[0], image.shape[1], image.shape[2]
        return torch.full((1, 16, (t - 1) // 4 + 1, h // 8, w // 8), float(len(self.seen)))


def clip():
    torch.manual_seed(0)
    video = torch.rand(T, H, W, 3) * 0.8 + 0.1
    mask = torch.zeros(T, H, W)
    for column, level in zip((OBJECT, OVERLAP, AFFECTED, FAINT, KEPT), LEVELS):
        mask[:, :, column] = level
    return video, mask


def upstream_mask_channels(mask: torch.Tensor) -> torch.Tensor:
    """What upstream's VAE is given as the mask, from its pipeline: quantise, then `1 - mask`, not normalised."""
    m = torch.where(mask > 0.75, 1.0, mask)
    m = torch.where((m <= 0.75) * (m >= 0.25), 127.0 / 255.0, m)
    m = torch.where(m < 0.25, 0.0, m)
    return 1.0 - m


def run_core(vae, video, mask):
    from comfy_extras.nodes_void import VOIDInpaintConditioning
    cond = [[torch.zeros(1, 1, 4), {}]]
    out = VOIDInpaintConditioning.execute(cond, cond, vae, video, mask, W, H, T, 1)
    return getattr(out, "args", out)


def run_ours(vae, video, mask, **kwargs):
    cond = [[torch.zeros(1, 1, 4), {}]]
    out = vc.MiniMaxH3VoidConditioning.execute(cond, cond, vae, video, mask, W, H, T, **kwargs)
    return getattr(out, "args", out)


def the_mask():
    video, mask = clip()
    ours, core = StandInVAE(), StandInVAE()
    run_ours(ours, video, mask)
    run_core(core, video, mask)
    given = ours.seen[0][..., 0]
    assert torch.allclose(given, upstream_mask_channels(mask), atol=1e-6), "the encoder is not given upstream's mask channels"
    levels = [round(float(given[0, 0, c].mean()), 3) for c in (OBJECT, AFFECTED, KEPT)]
    assert levels == [0.0, round(1.0 - 127.0 / 255.0, 3), 1.0], levels
    assert float(given[0, 0, OVERLAP].mean()) == 0.0, "upstream's pipeline turns the overlap level into the object; this did not"
    assert float(given[0, 0, FAINT].mean()) == 1.0, "a value under upstream's lower threshold was not read as kept"
    control = [round(float(core.seen[0][0, 0, c, 0].mean()), 3) for c in (OBJECT, AFFECTED, KEPT)]
    assert control == [-1.0, 0.0, 1.0], (
        f"control failed: core's node now gives its encoder {control} for the object, the affected area and what "
        "is kept, not -1, 0, 1. If core has moved to upstream's range, this node's first reason is gone")
    return f"encoder given {levels}; core's node gives {control}"


def the_video():
    video, mask = clip()
    ours = StandInVAE()
    run_ours(ours, video, mask)
    assert torch.allclose(ours.seen[1], video * 2.0 - 1.0, atol=1e-6), "the whole video is not what the encoder is given"
    black = StandInVAE()
    run_ours(black, video, mask, video_under_mask=vc.VIDEO_OBJECT_BLACK)
    given = black.seen[1]
    assert bool((given[:, :, OBJECT] == -1.0).all()) and bool((given[:, :, OVERLAP] == -1.0).all()), "the object is not blacked out"
    for column in (AFFECTED, FAINT, KEPT):
        assert torch.allclose(given[:, :, column], (video * 2.0 - 1.0)[:, :, column], atol=1e-6), \
            "the affected or the kept area of the video was changed"
    core = StandInVAE()
    run_core(core, video, mask)
    dimmed = torch.allclose(core.seen[1][:, :, AFFECTED], (video * 0.5 * 2.0 - 1.0)[:, :, AFFECTED], atol=1e-6)
    assert dimmed, ("control failed: core's node no longer shows the affected area at half brightness. If it shows "
                    "the whole video now, this node's second reason is gone")


def the_outputs():
    video, mask = clip()
    ours, core = StandInVAE(), StandInVAE()
    mine, theirs = run_ours(ours, video, mask), run_core(core, video, mask)
    a, b = mine[0][0][1]["concat_latent_image"], theirs[0][0][1]["concat_latent_image"]
    assert tuple(a.shape) == tuple(b.shape) and a.shape[1] == 32, (tuple(a.shape), tuple(b.shape))
    assert float(a[:, :16].mean()) == 1.0 and float(a[:, 16:].mean()) == 2.0, "the mask's latents are not first, the video's second"
    assert torch.equal(mine[1][0][1]["concat_latent_image"], a), "the negative does not carry the same conditioning"
    assert tuple(mine[2]["samples"].shape) == tuple(theirs[2]["samples"].shape), "the empty latent is not core's shape"
    from comfy_extras.nodes_void import _valid_void_length
    for asked in (49, 45, 13, 9):
        long_video, long_mask = torch.rand(asked, H, W, 3), torch.zeros(asked, H, W)
        _, used = vc.conditioning(StandInVAE(), long_video, long_mask, W, H, asked)
        assert used == _valid_void_length(asked), f"{asked} frames asked for: {used} used, core's rule gives {_valid_void_length(asked)}"
    _, used = vc.conditioning(StandInVAE(), video, mask, W, H, 197)
    assert used == _valid_void_length(T), "a length past the frames given was not brought back to them"


def the_refusals():
    video, mask = clip()
    for scaling in (lambda x: x, lambda x: x * 2.0):
        try:
            run_ours(StandInVAE(scaling), video, mask)
        except ValueError as exc:
            assert "x * 2 - 1" in str(exc), exc
        else:
            raise AssertionError("a VAE with another pixel scaling was accepted")
    try:
        run_ours(StandInVAE(), video, mask, video_under_mask="dimmed")
    except ValueError as exc:
        assert "video_under_mask" in str(exc), exc
    else:
        raise AssertionError("an unknown video_under_mask was accepted")
    try:
        run_ours(StandInVAE(), video, torch.zeros(T, H))
    except ValueError as exc:
        assert "quadmask" in str(exc), exc
    else:
        raise AssertionError("a quadmask that is not [T, H, W] was accepted")


def the_wiring():
    from comfy_extras.nodes_void import VOIDInpaintConditioning
    ours = vc.MiniMaxH3VoidConditioning.define_schema()
    core = VOIDInpaintConditioning.define_schema()
    mine, theirs = [i.id for i in ours.inputs], [i.id for i in core.inputs]
    assert mine == [i for i in theirs if i != "batch_size"] + ["video_under_mask"], (mine, theirs)
    assert [o.display_name for o in ours.outputs] == [o.display_name for o in core.outputs], "the outputs are not core's"
    untold = [i.id for i in ours.inputs if i.id not in ("positive", "negative") and not i.tooltip]
    assert not untold, f"no tooltip on {untold}"
    choice = next(i for i in ours.inputs if i.id == "video_under_mask")
    assert choice.default == vc.VIDEO_WHOLE, "the default is not upstream's shipped config, the whole video"
    assert re.search(r"[\s\[]MiniMaxH3VoidConditioning[,\]]", (REPO / "nodes.py").read_text()), \
        "MiniMaxH3VoidConditioning is not in the pack's node list (nodes.py)"


def main() -> int:
    case("the encoder is given upstream's mask, and core's node is not", the_mask)
    case("the model is shown the whole video, and core's node dims it", the_video)
    case("the outputs are core's", the_outputs)
    case("another pixel scaling, an unknown choice and a misshapen mask are refused", the_refusals)
    case("the node is wired as core's", the_wiring)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
