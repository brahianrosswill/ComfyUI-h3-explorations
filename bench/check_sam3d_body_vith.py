#!/usr/bin/env python3
"""Hold `sam3d_body_vith.py` to Meta's ViT-H SAM 3D Body: the key set, the six geometry rules, and the backbone's forward.

The ViT-H release differs from the DINOv3 one core runs in two ways: the
backbone, and a 512 by 384 view cut out of the same square crop
(`sam3d_body_vith.py`'s docstring lists the six places). A mistake in either
still loads and still draws a mesh, which is why each is held here.

Without weights, on the CPU:

  backbone_keys_are_metas      the converter's ViT-H patterns, expanded, are
                               exactly the module's `backbone.*` keys. Delete
                               and a renamed submodule loads nothing and the
                               loader's strict check is the first to say so,
                               at run time.
  rest_is_cores                every key and shape outside the backbone equals
                               core's `SAM3DBody`, so the DINOv3 converter's rig
                               map and its two fetched groups apply unchanged.
  other_layouts_are_refused    a DINOv3 backbone key and a ViT with a block
                               missing are refused by the converter, and a file
                               without `backbone.pos_embed` by the loader.
  image_and_rays_are_cut       the normalised image and the ray condition are
                               core's own, columns 64 to 448.
  dense_pe_is_the_squares      both decoders' position encodings are the 32 by
                               32 grid's, columns 4 to 28; with a control that
                               this is NOT the encoding of a 32 by 24 grid,
                               which is what core's model would compute.
  mask_embedding_is_cut        the mask embedding likewise.
  keypoints_sample_as_metas    core's keypoint-token update, given the padded
                               feature map, adds the features Meta's rule
                               samples (x scaled by 16/12 on the unpadded map);
                               with a control that the unpadded call differs.

The geometry cases compare against the expressions Meta wrote in
`sam_3d_body/models/meta_arch/sam3d_body.py` and `base_model.py`, transcribed
here beside each case. That is this check's assumption, stated: a
transcription can go stale if Meta changes the file, and it cannot be
imported because those methods live on a class that needs Meta's whole
stack.

With the converted file on disk (two cases, skipped without it, exit 2):

  file_loads_through_the_loader   `load_model` accepts the converted file, on
                                  the CPU: no missing key, none unexpected.
  backbone_forward_matches_metas  one seeded 512 by 384 input through this
                                  backbone and through Meta's own
                                  `backbones/vit.py::vit512_384`, both in
                                  float32, inside `FORWARD_TOLERANCE`.

**Meta's file is run, in a child process, and nothing from it enters this
one.** `AGENTS.md` says not to import Python from `coderef/`; the pack never
does, and this check's use of it as a numeric reference is what the owner
asked for when they put the checkout there (mryellow's brief, 2026-10-05).
The child loads that one file by path with two stand-ins: `timm`'s three
helpers, which are not installed here and are not Meta's code, and
`LayerNorm32`, three lines read from `models/modules/transformer.py`, whose
other imports the child has no use for. It loads Meta's ORIGINAL
`model.ckpt`, found beside the converted file, so the comparison also covers
what the converter wrote.

The converted file is `models/detection/sam_3d_body_vith.safetensors` under
the ComfyUI root, or `H3_SAM3D_VITH_FILE`.

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/check_sam3d_body_vith.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import COMFY, REPO, bootstrap, case, finish, skip  # noqa: E402

bootstrap(cpu=True)          # the model is built and run on the CPU here
sys.path.append(str(REPO))   # after ComfyUI: this pack's `nodes.py` must not shadow core's

import comfy.ops  # noqa: E402
from comfy.ldm.sam3d_body.model.model import SAM3DBody  # noqa: E402
from comfy.ldm.sam3d_body.model.prompt import PromptEncoder  # noqa: E402
from comfy.ldm.sam3.sam import PositionEmbeddingRandom  # noqa: E402

import convert_sam3d_body_checkpoint as C  # noqa: E402
import convert_sam3d_body_vith_checkpoint as CV  # noqa: E402
import sam3d_body_vith as V  # noqa: E402

META = REPO / "coderef" / "sam-3d-body"
META_VIT = META / "sam_3d_body" / "models" / "backbones" / "vit.py"

# measured 2026-10-05 on the CPU in float32: the largest absolute difference
# over the output, as a fraction of the output's largest magnitude, was
# 1.2e-06 on the one seeded input (two attention implementations, the same
# weights). The bound leaves room for another BLAS. Two mutations run the
# same day land far outside it: a patch convolution without its padding of
# 2, and the position embedding without its class slot added.
FORWARD_TOLERANCE = 1e-4

_state: dict = {}


def model():
    """One weightless `SAM3DBodyViTH`, with the layers the cases drive given values."""
    if "model" not in _state:
        torch.manual_seed(0)
        m = V.SAM3DBodyViTH(dtype=torch.float32, operations=comfy.ops.disable_weight_init)
        for module in (m.prompt_encoder.mask_downscaling, m.prompt_encoder.no_mask_embed,
                       m.keypoint_posemb_linear, m.keypoint_feat_linear, m.keypoint_embedding):
            for p in module.parameters():
                torch.nn.init.normal_(p, std=0.05)
        _state["model"] = m.eval()
    return _state["model"]


def original_backbone_keys() -> list[str]:
    """Every backbone key of the release's `model.ckpt`, by pattern (read 2026-10-05)."""
    keys = ["backbone.pos_embed", "backbone.patch_embed.proj.weight", "backbone.patch_embed.proj.bias",
            "backbone.last_norm.weight", "backbone.last_norm.bias"]
    for n in range(32):
        b = f"backbone.blocks.{n}"
        keys += [f"{b}.{part}.{kind}" for part in ("norm1", "norm2", "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
                 for kind in ("weight", "bias")]
    return keys


# ------------------------------------------------------------ without weights

def backbone_keys_are_metas():
    keys = original_backbone_keys()
    CV.check_backbone_keys(keys)
    ours = {k for k in model().state_dict() if k.startswith("backbone.")}
    assert ours == set(keys), (f"missing {sorted(set(keys) - ours)[:4]}, "
                               f"not Meta's {sorted(ours - set(keys))[:4]}")
    return f"{len(keys)} keys, none renamed"


def rest_is_cores():
    core = SAM3DBody(dtype=torch.float32, operations=comfy.ops.disable_weight_init).state_dict()
    ours = model().state_dict()
    rest_core = {k: tuple(v.shape) for k, v in core.items() if not k.startswith("backbone.")}
    rest_ours = {k: tuple(v.shape) for k, v in ours.items() if not k.startswith("backbone.")}
    assert rest_ours == rest_core, (
        f"keys or shapes outside the backbone differ from core's: "
        f"{sorted(set(rest_ours) ^ set(rest_core))[:6]}")
    rig = set(C.RIG_MAP) | set(C.RIG_ATTRS)
    assert {k for k in ours if k.startswith("mhr.")} == rig, "the rig map does not cover mhr.*"
    return f"{len(rest_ours)} tensors, the rig's {len(rig)} among them"


def other_layouts_are_refused():
    for bad, why in ((["backbone.encoder.blocks.0.attn.qkv.weight"], "ViT-H layout"),
                     ([k for k in original_backbone_keys() if ".blocks.31." not in k], "32-block")):
        try:
            CV.check_backbone_keys(bad)
        except ValueError as exc:
            assert why in str(exc), f"refused for the wrong reason: {exc}"
        else:
            raise AssertionError(f"the converter accepted a backbone that is not the {why}")
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "not_vith.safetensors"
        save_file({"backbone.embeddings.cls_token": torch.zeros(1, 1, 4)}, str(path))
        try:
            V.load_model(str(path))
        except ValueError as exc:
            assert "not the ViT-H release" in str(exc), f"refused for the wrong reason: {exc}"
        else:
            raise AssertionError("the loader accepted a file with no backbone.pos_embed")


def _batch():
    """A two-person batch of 512 by 512 crops with the fields the ray condition reads."""
    torch.manual_seed(1)
    affine = torch.tensor([[1.6, 0.0, -40.0], [0.0, 1.6, -25.0]]).expand(1, 2, 2, 3).clone()
    affine[0, 1, :, 2] += 13.0
    cam = torch.tensor([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])[None]
    return {"img": torch.rand(1, 2, 3, 512, 512), "affine_trans": affine, "cam_int": cam}


def image_and_rays_are_cut():
    m, batch = model(), _batch()
    flat = batch["img"].view(2, 3, 512, 512)
    # Meta, base_model.py: `batch_inputs = batch_inputs[:, :, :, 64:-64]`
    want = SAM3DBody.data_preprocess(m, flat)[:, :, :, 64:-64]
    got = m.data_preprocess(flat)
    assert got.shape == (2, 3, 512, 384) and torch.equal(got, want), f"image: {tuple(got.shape)}"
    # Meta, sam3d_body.py: `ray_cond = ray_cond[:, :, :, 64:-64]`
    want = SAM3DBody.get_ray_condition(m, batch)[..., 64:-64]
    got = m.get_ray_condition(batch)
    assert got.shape == (1, 2, 2, 512, 384) and torch.equal(got, want), f"rays: {tuple(got.shape)}"
    assert not torch.equal(got, SAM3DBody.get_ray_condition(m, batch)[..., :384]), \
        "control: the centre columns equal the left columns, so the cut is not being tested"


def dense_pe_is_the_squares():
    m = model()
    for name, got, layer in (("body", m.prompt_encoder.get_dense_pe((32, 24)), m.prompt_encoder.pe_layer),
                             ("hand", m.hand_pe_layer((32, 24)), m.hand_pe_layer)):
        # Meta, sam3d_body.py: `get_dense_pe((32, 32))[:, :, :, 4:-4]` and
        # `hand_pe_layer((32, 32)).unsqueeze(0)[:, :, :, 4:-4]`
        want = PositionEmbeddingRandom.forward(layer, (32, 32))[:, :, :, 4:-4]
        assert got.shape == (1, 1280, 32, 24) and torch.equal(got, want), f"{name}: {tuple(got.shape)}"
        narrow = PositionEmbeddingRandom.forward(layer, (32, 24))
        assert not torch.allclose(got, narrow, atol=1e-3), \
            f"control, {name}: equals the encoding of a 32 by 24 grid, so the override does nothing"


def mask_embedding_is_cut():
    m = model()
    torch.manual_seed(2)
    masks = torch.rand(2, 1, 512, 512)
    got, no_mask = m.prompt_encoder.get_mask_embeddings(masks, 2, (32, 24))
    # Meta, sam3d_body.py `_get_mask_prompt`: `mask_embeddings[:, :, :, 4:-4]`
    want = PromptEncoder.get_mask_embeddings(m.prompt_encoder, masks, 2, (32, 24))[0][:, :, :, 4:-4]
    assert got.shape == (2, 1280, 32, 24) and torch.equal(got, want), f"{tuple(got.shape)}"
    assert no_mask.shape == got.shape, "the no-mask embedding does not match the cut one"
    # and the whole prompt path runs on these shapes, which core's would not:
    m._max_num_person, m._batch_size = 2, 1
    batch = {"mask": masks[None], "mask_score": torch.tensor([[1.0, 0.0]])}
    out = m._get_mask_prompt(batch, torch.zeros(2, 1280, 32, 24))
    assert out.shape == (2, 1280, 32, 24)


def keypoints_sample_as_metas():
    m = model()
    torch.manual_seed(3)
    n_kp, start = m.keypoint_embedding.weight.shape[0], 5
    emb = torch.randn(2, 1280, 32, 24)
    tokens = torch.randn(2, start + n_kp + 3, 1024)
    augment = torch.zeros_like(tokens)
    # some keypoints outside the crop and one behind the camera, as in a real pass
    kp = torch.rand(2, n_kp, 2) * 1.2 - 0.6
    depth = torch.ones(2, n_kp)
    depth[0, 7] = -1.0
    pose = {"pred_keypoints_2d_cropped": kp, "pred_keypoints_2d_depth": depth}

    got = m._keypoint_token_update("body", start, emb, tokens, augment, pose, 0)[0]

    # Meta, sam3d_body.py `keypoint_token_update_fn`:
    #   sample_points = pred_keypoints_2d_cropped * 2
    #   sample_points[:, :, 0] = sample_points[:, :, 0] / 12 * 16
    #   feats = F.grid_sample(image_embeddings, sample_points[:, :, None, :],
    #                         mode="bilinear", padding_mode="zeros", align_corners=False)
    #   feats = feats * (~invalid_mask[:, :, None])
    #   token_embeddings[:, start:start + n] += keypoint_feat_linear(feats)
    points = kp * 2
    points[:, :, 0] = points[:, :, 0] / 12 * 16
    feats = F.grid_sample(emb, points[:, :, None, :], mode="bilinear", padding_mode="zeros",
                          align_corners=False).squeeze(3).permute(0, 2, 1)
    kp01 = kp + 0.5
    invalid = ((kp01[:, :, 0] < 0) | (kp01[:, :, 0] > 1) | (kp01[:, :, 1] < 0) | (kp01[:, :, 1] > 1)
               | (depth < 1e-5))
    want = tokens.clone()
    want[:, start:start + n_kp] += m.keypoint_feat_linear(feats * (~invalid[:, :, None]))

    scale = float(want.abs().max())
    worst = float((got - want).abs().max()) / scale
    assert worst < 1e-5, f"differs from Meta's rule by {worst:.2e} of the largest value"
    unpadded = SAM3DBody._keypoint_token_update(m, "body", start, emb, tokens, augment, pose, 0)[0]
    control = float((unpadded - want).abs().max()) / scale
    assert control > 1e-2, f"control: core's unpadded call differs by only {control:.2e}"
    return f"within {worst:.1e}; the unpadded call is off by {control:.2f}"


# --------------------------------------------------------------- with weights

def converted_file() -> Path:
    named = os.environ.get("H3_SAM3D_VITH_FILE")
    path = Path(named).expanduser() if named else COMFY / "models" / "detection" / "sam_3d_body_vith.safetensors"
    if not path.is_file():
        skip("needs the converted file, models/detection/sam_3d_body_vith.safetensors under the "
             "ComfyUI root or H3_SAM3D_VITH_FILE (bench/convert_sam3d_body_vith_checkpoint.py writes it)")
    return path


def file_loads_through_the_loader():
    patcher = V.load_model(str(converted_file()))
    inner = patcher.model
    assert isinstance(inner, V.SAM3DBodyViTH) and isinstance(inner.backbone, V.ViTHBackbone)
    assert tuple(inner.image_size) == (512, 512), "core's predictor crops to this; it must stay the square"
    dtypes = {str(p.dtype) for n, p in inner.backbone.named_parameters() if ".norm" in n or "last_norm" in n}
    assert dtypes == {"torch.float32"}, f"the backbone's norms are {dtypes}, Meta keeps them float32"
    return f"{sum(1 for _ in inner.state_dict())} tensors, none missing, none unexpected"


#: Run by the child. Meta's file, loaded by path; see the module docstring.
_META_CHILD = r'''
import importlib.util, sys, types
import torch, torch.nn as nn

vit_path, ckpt_path, in_path, out_path = sys.argv[1:5]

timm = types.ModuleType("timm"); models = types.ModuleType("timm.models"); layers = types.ModuleType("timm.models.layers")
def drop_path(x, drop_prob=0.0, training=False):
    assert not training, "the reference runs in eval mode, where stochastic depth is the identity"
    return x
layers.drop_path = drop_path
layers.to_2tuple = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)
layers.trunc_normal_ = nn.init.trunc_normal_
sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})

class LayerNorm32(nn.LayerNorm):          # sam_3d_body/models/modules/transformer.py, read 2026-10-05
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
for name in ("sam_3d_body", "sam_3d_body.models", "sam_3d_body.models.backbones", "sam_3d_body.models.modules"):
    pkg = types.ModuleType(name); pkg.__path__ = []; sys.modules[name] = pkg
tr = types.ModuleType("sam_3d_body.models.modules.transformer"); tr.LayerNorm32 = LayerNorm32
sys.modules[tr.__name__] = tr

spec = importlib.util.spec_from_file_location("sam_3d_body.models.backbones.vit", vit_path)
vit = importlib.util.module_from_spec(spec); sys.modules[spec.name] = vit
spec.loader.exec_module(vit)

class _Backbone(dict):
    pass
class _Cfg:
    class MODEL:
        BACKBONE = _Backbone()
net = vit.vit512_384(_Cfg)
net.eval()                                # Meta's `train` override returns None
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
net.load_state_dict(sd, strict=True)
with torch.no_grad():
    out = net(torch.load(in_path))
torch.save(out, out_path)
print("meta backbone:", tuple(out.shape), out.dtype)
'''


def backbone_forward_matches_metas():
    path = converted_file()
    if not META_VIT.is_file():
        skip("needs Meta's code at coderef/sam-3d-body (the numeric reference)")
    ckpt = path.resolve().parent / "model.ckpt"
    if not ckpt.is_file():
        skip("needs Meta's original model.ckpt beside the converted file, for Meta's side of the comparison")

    from safetensors.torch import load_file
    ours = V.ViTHBackbone(dtype=torch.float32, operations=comfy.ops.manual_cast).eval()
    sd = {k[len("backbone."):]: v for k, v in load_file(str(path)).items() if k.startswith("backbone.")}
    ours.load_state_dict(sd, strict=True)

    torch.manual_seed(0)
    x = torch.randn(1, 3, *V.VIT_INPUT)
    with torch.no_grad():
        got = ours.forward_features(x)

    with tempfile.TemporaryDirectory() as tmp:
        x_path, y_path = Path(tmp) / "x.pt", Path(tmp) / "y.pt"
        torch.save(x, x_path)
        done = subprocess.run(
            [sys.executable, "-c", _META_CHILD, str(META_VIT), str(ckpt), str(x_path), str(y_path)],
            capture_output=True, text=True)
        assert done.returncode == 0, f"Meta's backbone did not run: {done.stderr.strip()[-600:]}"
        want = torch.load(y_path)

    assert got.shape == want.shape == (1, 1280, 32, 24), f"{tuple(got.shape)} vs {tuple(want.shape)}"
    assert torch.isfinite(got).all()
    scale = float(want.abs().max())
    worst = float((got - want).abs().max()) / scale
    assert worst < FORWARD_TOLERANCE, (
        f"largest difference is {worst:.2e} of the output's largest magnitude, "
        f"over the bound {FORWARD_TOLERANCE:.0e}")
    return f"largest difference {worst:.1e} of the largest magnitude ({scale:.2f})"


def main() -> int:
    print("bench/check_sam3d_body_vith.py -- CPU\n")
    torch.set_grad_enabled(False)
    for fn in (backbone_keys_are_metas, rest_is_cores, other_layouts_are_refused,
               image_and_rays_are_cut, dense_pe_is_the_squares, mask_embedding_is_cut,
               keypoints_sample_as_metas, file_loads_through_the_loader,
               backbone_forward_matches_metas):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
