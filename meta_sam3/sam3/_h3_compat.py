# Not Meta's file. Written for ComfyUI-h3-explorations; distributed with the copy under the SAM License.
"""Stand-ins for the two timm names Meta's SAM 3 inference code imports: `DropPath` and `trunc_normal_`."""

import torch
from torch import nn

#: timm's `trunc_normal_` initialises a parameter in place from a truncated normal; torch ships the same
#: initialiser. Every parameter it touches here is then overwritten by the checkpoint. Reasoned, not compared
#: draw for draw with timm's.
trunc_normal_ = nn.init.trunc_normal_


class DropPath(nn.Module):
    """Stochastic depth: in training, drop a whole residual branch per sample; in eval, the identity.

    It holds no parameter or buffer, so it changes no checkpoint key. Meta's builder constructs the ViT
    with a non-zero rate (`model_builder.py::_create_vit_backbone`, `drop_path_rate`), so the class has to
    exist; inference runs in eval mode, where this returns its input.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        if keep > 0.0 and self.scale_by_keep:
            mask.div_(keep)
        return x * mask
