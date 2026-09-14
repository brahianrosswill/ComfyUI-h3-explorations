#!/usr/bin/env python3
"""Check `MiniMaxH3ChannelBalance`: the fold is exact, RoPE-safe, off by default,
and picks the blocks the shipped checkpoint's weights say it should.

What each case defends, and what breaks it:

  fold_is_exact         `(kw/s) * (qw*s) == kw*qw` per channel, from the very
                        diffs the node hands to `add_patches`. Delete and a
                        sign or a reciprocal slips through while every render
                        still runs, silently changing q.k.
  fold_is_rope_safe     the factor is equal within every (i, i+48) pair and
                        leaves 96..127 free. Delete and a per-channel factor
                        that is not pair-equal no longer commutes with the
                        split-half rotation, and the "exact" claim is false.
  factor_is_scale_free  geometric mean one. Delete and the fold changes the
                        magnitude of q.k rather than its channel balance.
  off_adds_no_patch     the default mode returns a clone with no patches.
                        Delete and a graph that wires the node and leaves it
                        alone is no longer a control.
  named_blocks_syntax   "named blocks" goes through `block_spec.parse_blocks`,
                        the same grammar as `dense_blocks`.
  shipped_ranking       on the shipped unet, "loud blocks" at the default
                        threshold selects exactly {45, 48, 49}, and no other
                        block is within a factor of two of the threshold.
                        Skipped without the checkpoint. Delete and a weights
                        change (or a threshold edit) reroutes the fold with no
                        record. The set is a measurement of the release, not
                        a belief: ranking printed for the reader.

Needs ComfyUI importable for `comfy_api` (the checkout two directories up);
no server, no GPU, no model. The checkpoint case reads only the norm-weight
tensors (bf16 [128]) through safetensors.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))          # custom_nodes, so the pack imports as a package
sys.path.insert(0, str(ROOT.parent.parent))   # ComfyUI root, for comfy_api (no server, no GPU)
PACK = ROOT.name

cb = __import__(f"{PACK}.channel_balance", fromlist=["*"])

EXPECTED_LOUD = (45, 48, 49)   # measured 2026-09-14 on minimax_h3_fl2va_pruned_int8_convrot


def _norm_weights(seed=0, spike=None):
    g = torch.Generator().manual_seed(seed)
    kw = (1.0 + 0.2 * torch.randn(128, generator=g)).abs().to(torch.bfloat16)
    qw = (1.0 + 0.2 * torch.randn(128, generator=g)).abs().to(torch.bfloat16)
    if spike:
        for c in spike:
            kw[c] = kw[c] * 16
    return kw, qw


def test_fold_is_exact():
    kw, qw = _norm_weights(spike=(82, 19))
    dk, dq = cb.weight_patches(kw, qw, 0.5)
    before = kw.float() * qw.float()
    after = (kw.float() + dk.float()) * (qw.float() + dq.float())
    rel = ((after - before).abs() / before.abs()).max().item()
    assert rel < 2e-2, f"fold changes kw*qw by up to {rel:.3%}; must be bf16 rounding only"


def test_fold_is_rope_safe():
    kw, qw = _norm_weights(spike=(82, 19, 100))
    s = cb.balance_factor(kw, qw, 0.5)
    assert torch.allclose(s[:48], s[48:96]), "factor differs within a RoPE pair"
    assert not torch.allclose(s[96:], torch.ones(32)), "unrotated channels must be free to differ"


def test_factor_is_scale_free():
    kw, qw = _norm_weights(spike=(5,))
    s = cb.balance_factor(kw, qw, 0.5)
    assert abs(torch.log(s).mean().item()) < 1e-5, "geometric mean must be one"
    assert s.min() < 0.9 < 1.1 < s.max(), "a spiked channel must move the factor"


class _FakePatcher:
    def __init__(self, blocks):
        self.blocks = blocks
        self.patches = {}

    def clone(self):
        c = _FakePatcher(self.blocks)
        c.patches = dict(self.patches)
        return c

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return types.SimpleNamespace(blocks=self.blocks)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.patches.update(patches)
        return list(patches)


def _fake_model(n=50, loud=()):
    blocks = []
    for i in range(n):
        kw, qw = _norm_weights(seed=i, spike=(82, 19) if i in loud else None)
        attn = types.SimpleNamespace(k_norm=types.SimpleNamespace(weight=kw), q_norm=types.SimpleNamespace(weight=qw))
        blocks.append(types.SimpleNamespace(attn=attn))
    return _FakePatcher(blocks)


def test_off_adds_no_patch():
    m = _fake_model(loud=(49,))
    out = cb.MiniMaxH3ChannelBalance.execute(m, balance="off")
    assert out.args[0].patches == {}, "off must add no patch"
    assert out.args[0] is not m, "must return a clone"


def test_named_blocks_syntax():
    m = _fake_model()
    out = cb.MiniMaxH3ChannelBalance.execute(m, balance="named blocks", blocks="45,48,-1")
    keys = sorted(out.args[0].patches)
    assert keys == sorted(
        f"diffusion_model.blocks.{i}.attn.{n}_norm.weight" for i in (45, 48, 49) for n in ("k", "q")
    ), keys


def test_shipped_ranking():
    unet = _shipped_unet()
    if unet is None:
        print("  skipped: shipped unet not found")
        return
    from safetensors import safe_open
    with safe_open(str(unet), framework="pt", device="cpu") as sf:
        shares = [cb.loud_share(sf.get_tensor(f"blocks.{i}.attn.k_norm.weight")) for i in range(50)]
    default = 0.15
    picked = tuple(i for i, s in enumerate(shares) if s >= default)
    assert picked == EXPECTED_LOUD, f"loud blocks at {default}: {picked}, expected {EXPECTED_LOUD}"
    near = [i for i, s in enumerate(shares) if default / 2 <= s < default * 2 and i not in picked]
    assert not near, f"blocks within 2x of the threshold, selection is fragile: {near}"
    print("  top-4 K-norm energy share by block: " + ", ".join(f"{i}:{s:.0%}" for i, s in enumerate(shares) if s > 0.1))


def _shipped_unet():
    try:
        sys.path.insert(0, str(ROOT))
        from workflows import h3_config
        name = h3_config.MODELS["unet_fl2va"]
    except Exception:
        return None
    for root in (ROOT.parent.parent / "models", Path.home() / "ComfyUI" / "models"):
        hits = list(root.glob(f"**/{name}")) if root.exists() else []
        if hits:
            return hits[0]
    return None


CASES = [test_fold_is_exact, test_fold_is_rope_safe, test_factor_is_scale_free,
         test_off_adds_no_patch, test_named_blocks_syntax, test_shipped_ranking]


def main() -> int:
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
