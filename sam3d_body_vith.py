"""SAM 3D Body's ViT-H release behind core's heads, predictor and renderers.

Meta publishes SAM 3D Body with two backbones. Core's
`comfy/ldm/sam3d_body/model/model.py::SAM3DBody` builds the DINOv3 one inside
its constructor, so core's loader cannot open `facebook/sam-3d-body-vith`.
Everything outside the backbone has the same names and shapes in both
releases (read off the two checkpoints on 2026-10-05), so this module is the
ViT-H backbone, a subclass of core's model that puts it behind core's own
heads, and a loader node that returns the type core's predict, smooth and
render nodes take. Nothing in core is patched.

## The geometry, which is the whole difference besides the backbone

The release's config names `BACKBONE.TYPE: vit_hmr_512_384` and still says
`IMAGE_SIZE: [512, 512]`. Meta's estimator crops each person to that square
for either backbone (`sam_3d_body_estimator.py`: `TopdownAffine(input_size=
cfg.MODEL.IMAGE_SIZE)`), so core's predictor feeds this model unchanged. The
ViT then sees only the centre 384 columns, and Meta's model class
(`sam_3d_body/models/meta_arch/sam3d_body.py` and `base_model.py`) special-
cases the type in six places, all of which keep the SQUARE crop's coordinate
frame and cut the backbone's view out of it:

  the normalised image       `[:, :, :, 64:-64]`, in `data_preprocess`
  the ray condition          `[:, :, :, 64:-64]`, in `forward_pose_branch`
  the decoder's dense PE     over a 32 by 32 grid, then `[:, :, :, 4:-4]`,
                             for the body (the prompt encoder's) and the hand
                             (its own layer)
  the mask embedding         `[:, :, :, 4:-4]`, in `_get_mask_prompt`
  keypoint feature sampling  x scaled by 16/12 before `grid_sample`, because
                             keypoints are normalised over the square and the
                             feature map covers three quarters of its width

Each is one override below, a few lines, calling core's own method. The last
is done by padding the feature map back to the square grid with zero columns
instead of scaling x: with `padding_mode="zeros"` and `align_corners=False`,
which both Meta and core pass, the two sample the same values (a sample at x
on the padded 32-wide map lands on pixel 16x + 15.5, and at 4x/3 on the
24-wide map on pixel 16x + 11.5, the same pixel four columns over).
`bench/check_sam3d_body_vith.py` holds each of the six against the formula
Meta wrote, and the backbone's forward against Meta's own file.

## The backbone

A plain ViT-H as Meta's `sam_3d_body/models/backbones/vit.py::vit512_384`
builds it: a 16 by 16 patch convolution **with padding 2** (the ViTPose
habit; the output grid is still 32 by 24), a learned position embedding with
a class slot that is added to every patch (`pos_embed[:, 1:] +
pos_embed[:, :1]`) and never interpolated, 32 pre-norm blocks with a fused
biased `qkv`, and a final norm. The norms compute in float32 and cast back
(`LayerNorm32`), and the checkpoint stores their weights in float32 beside
bf16 linears. Module names are Meta's, so the file needs no backbone rename.

Read from `coderef/sam-3d-body` (checked out 2026-10-05) and re-implemented;
nothing is imported from it.

Not measured here: accuracy against the DINOv3 release. `memory_used_forward`
is inherited from core, calibrated on the DINOv3 backbone, which has more
tokens per crop, so it over-reserves for this one.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.storage
import comfy.utils
import folder_paths
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.ldm.sam3.sam import PositionEmbeddingRandom
from comfy.ldm.sam3d_body.model.model import N_KEYPOINTS, SAM3DBody
from comfy.ldm.sam3d_body.model.prompt import PromptEncoder
from comfy_api.latest import io

logger = logging.getLogger(__name__)

# inherited, all of it: `vit512_384` in Meta's backbones/vit.py.
VIT_INPUT = (512, 384)          # (H, W) the backbone sees
VIT_PATCH = 16
VIT_PATCH_PADDING = 2           # `4 + 2 * (ratio // 2 - 1)` at ratio 1
VIT_DIM = 1280
VIT_DEPTH = 32
VIT_HEADS = 16
VIT_MLP_RATIO = 4
VIT_NORM_EPS = 1e-5             # `LayerNorm32` is `nn.LayerNorm` at its default eps

#: The key that says a file is this release: only the ViT backbone has it.
VITH_MARKER_KEY = "backbone.pos_embed"

# The same custom type core's nodes declare (`comfy_extras/nodes_sam3d_body.py`).
SAM3DBodyModel = io.Custom("SAM3D_BODY_MODEL")


# ------------------------------------------------------------------ geometry

def centre_columns(x: torch.Tensor, width: int) -> torch.Tensor:
    """The centre `width` columns of a (..., H, W) tensor. Meta's `[..., c:-c]`."""
    extra = x.shape[-1] - width
    if extra < 0 or extra % 2:
        raise ValueError(f"cannot take the centre {width} columns of a {x.shape[-1]}-wide tensor")
    side = extra // 2
    return x[..., side:x.shape[-1] - side] if side else x


class _SquareGridPE(PositionEmbeddingRandom):
    """The position encoding of the square crop's grid, cut to the backbone's view.

    Asked for an (H, W) grid with W < H it encodes (H, H) and returns the
    centre W columns, so a feature keeps the position it has in the crop.
    """

    def forward(self, size, device=None):
        h, w = size
        return centre_columns(super().forward((h, h), device=device), w)


class _SquareGridPromptEncoder(PromptEncoder):
    """Core's prompt encoder with its dense outputs cut to the backbone's view."""

    def get_dense_pe(self, size):
        h, w = size
        return centre_columns(self.pe_layer((h, h)), w)

    def get_mask_embeddings(self, masks, bs: int = 1, size=(16, 16)):
        mask_embeddings, no_mask_embeddings = super().get_mask_embeddings(masks, bs, size)
        return centre_columns(mask_embeddings, size[1]), no_mask_embeddings


# ------------------------------------------------------------------ backbone

def _norm32(dim: int, device, operations) -> nn.Module:
    """A LayerNorm that computes in float32 and casts back, as Meta's LayerNorm32.

    Its weights are float32 whatever the model's dtype, as in the checkpoint.
    """
    class LayerNorm32(operations.LayerNorm):
        def forward(self, x):
            return super().forward(x.float()).type(x.dtype)

    return LayerNorm32(dim, eps=VIT_NORM_EPS, device=device, dtype=torch.float32)


class _Attention(nn.Module):
    def __init__(self, dim, heads, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.qkv = operations.Linear(dim, dim * 3, bias=True, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attend = optimized_attention_for_device(x.device, mask=False)
        out = attend(q, k, v, self.heads, None, skip_reshape=True, skip_output_reshape=True,
                     low_precision_attention=False)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden, device, dtype, operations):
        super().__init__()
        self.fc1 = operations.Linear(dim, hidden, device=device, dtype=dtype)
        self.fc2 = operations.Linear(hidden, dim, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, heads, device, dtype, operations):
        super().__init__()
        self.norm1 = _norm32(dim, device, operations)
        self.attn = _Attention(dim, heads, device, dtype, operations)
        self.norm2 = _norm32(dim, device, operations)
        self.mlp = _Mlp(dim, dim * VIT_MLP_RATIO, device, dtype, operations)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, device, dtype, operations):
        super().__init__()
        self.proj = operations.Conv2d(3, VIT_DIM, kernel_size=VIT_PATCH, stride=VIT_PATCH,
                                      padding=VIT_PATCH_PADDING, device=device, dtype=dtype)

    def forward(self, x):
        return self.proj(x)


class ViTHBackbone(nn.Module):
    """Meta's `vit512_384`, with the attributes core's model reads off a backbone."""

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.patch_size = VIT_PATCH
        self.embed_dim = self.embed_dims = VIT_DIM
        grid = (VIT_INPUT[0] // VIT_PATCH, VIT_INPUT[1] // VIT_PATCH)
        self.patch_embed = _PatchEmbed(device, dtype, operations)
        # One slot more than the grid: the pretraining model's class token.
        self.pos_embed = nn.Parameter(torch.empty(1, grid[0] * grid[1] + 1, VIT_DIM, device=device, dtype=dtype))
        self.blocks = nn.ModuleList(_Block(VIT_DIM, VIT_HEADS, device, dtype, operations) for _ in range(VIT_DEPTH))
        self.last_norm = _norm32(VIT_DIM, device, operations)

    def forward_features(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        """(B, 3, 512, 384) -> (B, 1280, 32, 24)."""
        if tuple(x.shape[-2:]) != VIT_INPUT:
            raise ValueError(f"the ViT-H backbone takes {VIT_INPUT[0]} by {VIT_INPUT[1]} crops, got "
                             f"{tuple(x.shape[-2:])}: its position embedding is not interpolated")
        x = self.patch_embed(x)
        b, _c, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        pos = comfy.ops.cast_to_input(self.pos_embed, x)
        x = x + pos[:, 1:] + pos[:, :1]
        for block in self.blocks:
            x = block(x)
        x = self.last_norm(x)
        return x.permute(0, 2, 1).reshape(b, -1, hp, wp).contiguous()

    def forward(self, x, **kwargs):
        return self.forward_features(x, **kwargs)


# --------------------------------------------------------------------- model

class SAM3DBodyViTH(SAM3DBody):
    """Core's SAM 3D Body with the ViT-H backbone and its 512 by 384 view."""

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__(device=device, dtype=dtype, operations=operations)
        embed = self.backbone.embed_dims
        if embed != VIT_DIM or self.backbone.patch_size != VIT_PATCH:
            raise RuntimeError(
                f"core's SAM3DBody is built for a backbone of width {embed}, patch "
                f"{self.backbone.patch_size}; the ViT-H release needs {VIT_DIM} and {VIT_PATCH}, "
                "so core's heads no longer fit it")
        # Replaces what core's constructor built; the names are the same, so
        # the state dict's keys outside `backbone.*` do not move.
        self.backbone = ViTHBackbone(device=device, dtype=dtype, operations=operations)
        self.prompt_encoder = _SquareGridPromptEncoder(
            embed_dim=embed, num_body_joints=N_KEYPOINTS, device=device, dtype=dtype, operations=operations)
        self.hand_pe_layer = _SquareGridPE(embed // 2)

    def data_preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        return centre_columns(super().data_preprocess(inputs), VIT_INPUT[1])

    def get_ray_condition(self, batch):
        return centre_columns(super().get_ray_condition(batch), VIT_INPUT[1])

    def _keypoint_token_update(self, branch, kps_emb_start_idx, image_embeddings, *args):
        # Keypoints are normalised over the square crop; give core's sampler a
        # feature map that covers it (the module docstring has the arithmetic).
        side = (image_embeddings.shape[-2] - image_embeddings.shape[-1]) // 2
        return super()._keypoint_token_update(
            branch, kps_emb_start_idx, F.pad(image_embeddings, (side, side)), *args)


# -------------------------------------------------------------------- loader

def loader_view(sd: dict) -> dict:
    """A file's state dict as core's loader hands it to the model.

    The same two steps as `SAM3DBody_Loader.execute`: one rename, one pop.
    """
    sd = {k.replace(".layers.0.0.", ".layers.0."): v for k, v in sd.items()}
    sd.pop("hand_cls_embed.weight", None)
    sd.pop("hand_cls_embed.bias", None)
    return sd


def load_model(path: str):
    """The model in a patcher, as core's `SAM3DBody_Loader` builds its own."""
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if VITH_MARKER_KEY not in sd:
        raise ValueError(
            "this file has no `backbone.pos_embed`, so it is not the ViT-H release. A DINOv3 "
            "file loads with core's `Load SAM3D Body Model` node.")
    sd = loader_view(sd)

    load_device = comfy.model_management.get_torch_device()
    weight_dtype = comfy.utils.weight_dtype(sd)
    torch_dtype = comfy.model_management.unet_dtype(device=load_device, model_params=-1, weight_dtype=weight_dtype)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(torch_dtype, load_device)
    operations = comfy.ops.pick_operations(torch_dtype, manual_cast_dtype, load_device=load_device,
                                           disable_fast_fp8=True)

    model = SAM3DBodyViTH(dtype=torch_dtype, operations=operations)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"SAM 3D Body ViT-H checkpoint key mismatch: missing={sorted(missing)}, "
                           f"unexpected={sorted(unexpected)}")
    model.backbone_dtype = torch_dtype
    logger.info("[h3] MiniMaxH3SAM3DBodyViTHLoader: %d tensors, backbone dtype %s", len(sd), torch_dtype)

    return comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
        size=comfy.model_management.module_size(model),
        fast_disk=comfy.storage.state_dict_fast_disk(sd),
    )


class MiniMaxH3SAM3DBodyViTHLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SAM3DBodyViTHLoader",
            display_name="MiniMax H3 SAM 3D Body Loader (ViT-H)",
            category="model/latent/minimax",
            description=(
                "Loads the ViT-H release of SAM 3D Body from models/detection/. The output goes "
                "into core's Run SAM3D Body Prediction node like core's own loader's. The file is "
                "the one bench/convert_sam3d_body_vith_checkpoint.py writes from Meta's originals."),
            inputs=[
                io.Combo.Input(
                    "model_file", options=folder_paths.get_filename_list("detection"),
                    tooltip=("The converted ViT-H file under models/detection/. A DINOv3 file is "
                             "refused here; it loads with core's Load SAM3D Body Model node.")),
            ],
            outputs=[SAM3DBodyModel.Output(display_name="sam3d_body_model")],
        )

    @classmethod
    def execute(cls, model_file) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("detection", model_file)
        return io.NodeOutput(load_model(path))
