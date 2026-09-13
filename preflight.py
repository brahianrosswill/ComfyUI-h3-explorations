"""What this render will actually cost, before you queue it.

Every other node in the graph knows one piece. `MiniMaxH3Resolution` knows the
video and nothing about references. `MiniMaxH3ReferenceFit` knows one
reference and nothing about the video. The total only exists once conditioning
is assembled, which is where this sits.

It reads the same `PackedLayout` the model builds, so the sequence length here
is the one attention will run at rather than an estimate. Four things it
reports that nothing else can:

**Whether you are inside the trained family.** Core's conditioning nodes take
width and height as plain ints and never call `adapt_canvas` on the target
canvas (`comfy_extras/nodes_minimax_h3.py` calls it only to size reference
videos), so `BASE_SHORT_EDGE` and `MAX_PIXELS` there constrain nothing you
type. 1024x1024 is legal, 32-divisible, renders, costs more per frame than
16:9, and is outside the family the checkpoint was trained on. Nothing else
says so.

**What the segments cost relative to each other.** Reference tokens ride every
sampling step exactly as video tokens do, so the share of the sequence the
references take is the number that decides whether to resize them. The
reference rows are whatever the append chain compiled into `minimax_refs`
(`latent_h`, `latent_w`, `ref_audio_t` per block), so a change to reference
sizing policy shows up here without this file knowing the policy.

**What a different aspect ratio would cost.** A tradeoff you cannot act on is
not a tradeoff. The alternatives are computed at the same length and
conditioning, so the comparison is honest.

**Where sage's integer-offset ceilings sit for the layout H3 actually uses.**
H3 builds q, k and v as three views of one fused projection
(`comfy/ldm/minimax/model.py`, `qkv_proj(x).split`), so each view carries
`stride_seq = 3*heads*head_dim` (`_FUSED_STRIDE` below), not the contiguous
`heads*head_dim` that KJNodes' `MiniMaxH3TokenCounter` warns from; that
warning stays silent through the range that matters. Two of the sage fork's
quantizers address the caller's tensors through that stride, and both run on
this box: on sm89 `sageattn_consume` takes the fp8 CUDA path with
`qk_quant_gran` left at the kernel's `per_thread` default
(`sageattention/core.py`), so q and k are quantized by the Triton kernels in
`sageattention/triton/quant_per_thread.py`, whose int32 offsets cross at
`_INT32_FUSED` and which carry the fork's `USE_I64` fix, and v is quantized
by `sageattention/quant.py::per_channel_fp8` through the CUDA kernels in
`csrc/fused/fused.cu`, whose uint32 strides wrap at `_CSRC_FUSED` and are NOT
fixed. The fork's `CHANGELOG.md` records both, as "int32 element-offset
overflow in the INT8 quant kernels" under v0.7.0 and "The CUDA quant kernels
form global offsets in uint32" under known issues. The int32 fix reaches every
user of this pack's sage node because `attention.py::build_kernel` refuses a
sageattention without `sageattn_consume`, which arrived in that same v0.7.0.
Length alone does not reach the uint32 wrap (the changelog entry gives the
frame equivalent; `h3_rules.MAX_LENGTH` is the ceiling) but references can,
so the report prints the headroom rather than asserting clearance, and both
boundaries are stated so an absent warning is not mistaken for one. These are
the `MiniMaxH3SageAttention` kernels' limits: with Sol-Attn wired as well,
Sol's dense calls chain to the sage override (`sol_attn_h3.py`, `dense()`),
and a graph with no sage node never runs them.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

try:
    from .h3_rules import describe_length
except ImportError:  # pragma: no cover
    from h3_rules import describe_length  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Inherited: heads and head_dim are `num_attention_heads=56,
# attention_head_dim=128` in `comfy/ldm/minimax/model.py`, and the fused
# stride follows from `qkv_proj(x).split(...)` there handing attention three
# views of one buffer. Reasoned: the two ceilings are the first row whose
# element offset no longer fits the offset type the kernel forms it in, int32
# for the Triton q/k quantizers and uint32 for the CUDA v quantizer (module
# docstring). The sage fork's CHANGELOG.md states the same two figures under
# its own derivation; this file computes rather than copies them.
_FUSED_STRIDE = 3 * 56 * 128
_CONTIGUOUS_STRIDE = 56 * 128
_INT32_FUSED = 2**31 // _FUSED_STRIDE
_INT32_CONTIGUOUS = 2**31 // _CONTIGUOUS_STRIDE
_CSRC_FUSED = 2**32 // _FUSED_STRIDE

_ALTERNATIVES = ("1:1", "4:3", "3:2", "16:9", "9:16")


def _bar(fraction, width=20):
    filled = max(0, min(width, round(fraction * width)))
    return "#" * filled + "." * (width - filled)


class MiniMaxH3Preflight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Preflight",
            display_name="MiniMax H3 Preflight",
            category="model/conditioning/minimax",
            description=(
                "Reports what this render will cost before you queue it: "
                "sequence length broken down by segment, whether the "
                "resolution is inside the trained family, what other aspect "
                "ratios would cost at the same length, and where the sage "
                "fork's quantizer ceilings sit for H3's fused qkv layout. "
                "Pass-through; it changes nothing. Wire it between "
                "conditioning and the sampler."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("samples"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Latent.Output(display_name="samples"),
                io.Int.Output(display_name="sequence_length"),
                io.String.Output(display_name="report"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, conditioning, samples) -> io.NodeOutput:
        from comfy.ldm.minimax.model import PackedLayout
        from comfy_extras.nodes_minimax_h3 import adapt_canvas

        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            video, audio = latent.unbind()[:2]
            audio_t = audio.shape[-1]
        else:
            video, audio_t = latent, 0
        if video.ndim != 5:
            raise RuntimeError(
                f"Expected an H3 video latent of shape [B, C, T, H, W]; got "
                f"{tuple(video.shape)}. Wire this to the latent the H3 "
                f"conditioning node produced.")

        latent_t = video.shape[2]
        # h/w round up to the DiT's 2x2 patch, matching model_base's extra_conds
        lat_h = (video.shape[3] + 1) // 2 * 2
        lat_w = (video.shape[4] + 1) // 2 * 2
        width, height = lat_w * 16, lat_h * 16

        # Scheduled conditioning can differ in text length; report the largest,
        # because the peak is what has to fit.
        layout = max(
            # `frame_count=` was dropped from PackedLayout upstream (the
            # signature is now text_len, latent_t, latent_h, latent_w, audio_t,
            # keyframes, refs). Passing it raised TypeError and failed EVERY
            # graph in this repo at the Preflight node -- not a degraded
            # number, no render at all. Caught within minutes of a `git pull`
            # only because a render was already queued; a static check would
            # not have found it, because the break is in a call into upstream.
            (PackedLayout(cond.shape[1], latent_t, lat_h, lat_w, audio_t,
                          keyframes=cd.get("minimax_keyframes"),
                          refs=cd.get("minimax_refs"))
             for cond, cd in conditioning),
            key=lambda l: l.seq_len)

        by_kind: dict[str, int] = {}
        for a, b, kind in layout.segments:
            by_kind[kind] = by_kind.get(kind, 0) + (b - a)
        total = layout.seq_len

        tokens_per_frame = (width // 32) * (height // 32)
        in_family = adapt_canvas(width, height) == (width, height)
        # No node writes `minimax_frame_count` today: core's
        # `nodes_minimax_h3.py` sets only `minimax_keyframes` and
        # `minimax_refs`, and this pack's conditioning nodes set the same two
        # (grep either tree for the key; this file is its only reader). An
        # earlier core did write it on the keyframe path alone, and sourcing
        # the duration line from it made the line vanish on every graph
        # without keyframes, which is exactly where the frame ceiling matters
        # most. So the key is honoured if something supplies it and `latent_t`,
        # already in hand, is the source otherwise, rather than printing
        # nothing and letting absence read as "fine".
        frames = None
        for _cond, cd in conditioning:
            if cd.get("minimax_frame_count"):
                frames = cd["minimax_frame_count"]
                break
        if frames is None and latent_t:
            # inverse of video_latent_t: latent_t = ((n - 5) // 17) * 5 + 2,
            # with one temporal step meaning one frame rather than five. That
            # case is reachable only on a stack that accepts one frame; on a
            # stock core, which clamps to a 5-frame floor, `latent_t == 1`
            # cannot happen and the else-branch's 5 is right for every latent
            # that exists. Kept because this reads a latent it did not build:
            # the pack's floor shim is parked (`archive/single_frame.py`) but a
            # patched core or a hand-built graph still produces the case, and
            # pricing it at 5 would be wrong by 5x on the one that matters.
            frames = (1 if latent_t == 1 else
                      ((latent_t - 2) // 5) * 17 + 5 if latent_t > 2 else 5)

        # Every kind `PackedLayout` emits (`comfy/ldm/minimax/model.py`, the
        # "kinds:" comment beside `self.segments`). An unlisted kind still
        # prints, under its raw name.
        label = {"text": "text", "cond": "keyframes",
                 "cond_audio": "keyframe audio", "ref_img": "references",
                 "ref_audio": "audio refs", "audio": "audio", "video": "video"}
        lines = [
            f"{width}x{height}  "
            f"{'trained family' if in_family else 'OUTSIDE trained family'}"
            f"  {tokens_per_frame} video tokens/frame",
        ]
        if frames:
            lines.append(f"{describe_length(frames)}  {latent_t} latent frames")
        lines.append(f"sequence length {total:,}")
        for kind, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {label.get(kind, kind):<15}{n:>8,}  "
                         f"{_bar(n / total)}  {100 * n / total:5.1f}%")

        # Alternatives at the same length and conditioning: only the video
        # segment moves, so the comparison is exact rather than modelled.
        video_tokens = by_kind.get("video", 0)
        rest = total - video_tokens
        lines.append("if the aspect ratio changed, same length:")
        for name in _ALTERNATIVES:
            aw, ah = (int(x) for x in name.split(":"))
            cw, ch = adapt_canvas(aw * 1000, ah * 1000)
            alt_video = (cw // 32) * (ch // 32) * latent_t
            alt = rest + alt_video
            mark = "  <- current" if (cw, ch) == (width, height) else ""
            lines.append(f"  {name:<6}{cw}x{ch:<6}{alt:>9,}  "
                         f"{(alt - total) / total:+6.0%}{mark}")

        # These lines describe the sage fork's quantizers (module docstring)
        # and mean nothing on a graph without `MiniMaxH3SageAttention`.
        if total >= _CSRC_FUSED:
            lines.append(f"sage: {total:,} is past the csrc/fused uint32 wrap "
                         f"at {_CSRC_FUSED:,} (the CUDA v quantizer). This "
                         f"one is NOT fixed.")
        elif total >= _INT32_FUSED:
            # "unreachable at any length" was true of LENGTH alone and false
            # once references are in play, which is exactly when this line is
            # read: core's reference node accepts up to three reference
            # videos (`ref_video_` in `nodes_minimax_h3.py`), and a full-length
            # clip plus three of them crosses the wrap. So the old wording
            # reassured the user about a ceiling they can actually hit.
            # Report the headroom instead of asserting there is enough.
            head = _CSRC_FUSED - total
            lines.append(
                f"sage: past the fused int32 crossing at {_INT32_FUSED:,} "
                f"(Triton q/k quantizers), fixed in every fork build that "
                f"has sageattn_consume. Next ceiling {_CSRC_FUSED:,} (CUDA v "
                f"quantizer) is NOT fixed and is {head:,} away "
                f"({100 * total / _CSRC_FUSED:.0f}% of it). Length alone "
                f"cannot reach it; references can.")
        else:
            lines.append(f"sage: under the fused int32 crossing at "
                         f"{_INT32_FUSED:,} (a contiguous layout would say "
                         f"{_INT32_CONTIGUOUS:,}).")

        report = "\n".join(lines)
        logger.info("[h3] preflight %s", report.replace("\n", " | "))

        unique_id = getattr(cls.hidden, "unique_id", None)
        if unique_id:
            try:
                from server import PromptServer
                PromptServer.instance.send_progress_text(report, unique_id)
            except Exception:
                pass

        return io.NodeOutput(conditioning, samples, total, report)
