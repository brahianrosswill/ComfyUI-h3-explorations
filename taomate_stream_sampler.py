"""TaoMate-H3's streaming runtime as a ComfyUI SAMPLER.

`docs/h3_taomate.md` section 7 is the design and `taomate_streaming.py` holds
the parts that need no ComfyUI. This file is the loop: it runs inside one
`SamplerCustomAdvanced` call with a `BasicGuider`, on the plain fl2va
checkpoint with the TaoMate LoRA, and generates the video chunk by chunk.

## What one chunk does

1. Slice the chunk's video noise and its frozen track.
2. Run three steps at the adapter's sigmas. Before each forward the chunk's
   audio rows hold, in turn, noise and then the track mixed with noise at the
   audio teacher's states 3 and 6. The adapter's audio output is ignored,
   because upstream's student never owns audio.
3. Match the chunk's colour to the first chunk.
4. Run a clean forward with the clean track, recording every block's K and V
   for the chunk's audio and video rows.
5. Trim the cache to the first chunk's video plus the two most recent commits.

Each forward goes through `BaseModel.apply_model` with a chunk-sized layout.
Its positions are sliced from the run's full `PackedLayout`, and its
`audio_scale` is 1, so the model sees the stream's own audio. Attention is
swapped per block through `patches_replace["dit"]`: text attends to text, and
audio and video attend to text, the cache and the chunk. Nothing in core is
patched.

## What it refuses, and why

- **Other attention patches** (sage, Sol, VSA, exact blocks) and **other
  object patches** (PDD heads). The adapter was trained under dense bf16
  attention. The block hook replaces the attention those patches install for
  every call it intercepts. And Sol's Morton and sink resolve spans by the
  identity of `position_ids`, which sliced positions do not carry.
- **References and keyframes.** Upstream is text to audio-video; the chunk
  layout carries text, audio and video only.
- **A run length that is not TaoMate's**: 124 frames, then 119 per request.
- **Sigmas other than the adapter's grid**, and no frozen track in stream
  mode, until the base-model audio teacher is built.

## Modes

`stream` is the port. `verify_whole_clip` is the equality check from section
7.3 step 2, not a way to render. It runs the stock Euler loop over the whole
clip through the same block hook, with every row attending to every row and no
cache. It must reproduce the stock sampler's latent.
"""

from __future__ import annotations

import functools
import json
import logging
import time

import torch

import comfy.model_management
import comfy.quant_ops
import comfy.samplers
import comfy.utils
from comfy.ldm.minimax.model import patchify_video, unpatchify_video
from comfy_api.latest import io

from . import taomate_streaming as tm

#: Append only: a saved graph stores the chosen string, so new modes go last.
#: `control_text_only` is `verify_whole_clip` with upstream's text-only routing
#: in the hooked loop. It must NOT match core, which is what shows the exact
#: match in `verify_whole_clip` comes from the hook running and not from it
#: being bypassed.
MODES = ("stream", "verify_whole_clip", "control_text_only")
CACHE_DEVICES = ("cpu_pinned", "cpu", "gpu")
#: Reasoned: the graph's `ManualSigmas` string keeps six decimals.
SIGMA_TOL = 1e-5


class _ChunkLayout:
    """A `PackedLayout` stand-in for one chunk: `[text | audio | video]`, positions given."""

    def __init__(self, text_len, chunk, frame_rows, positions, lat_h, lat_w):
        audio_rows = 2 * chunk.audio_latents
        video_rows = chunk.video_latents * frame_rows
        self.seq_len = text_len + audio_rows + video_rows
        if positions.shape[0] != self.seq_len:
            raise RuntimeError(f"chunk positions {positions.shape[0]} rows, layout {self.seq_len}")
        self.position_ids = positions
        self.audio_pos = torch.arange(text_len, text_len + audio_rows)
        self.audio_update = torch.ones(audio_rows, dtype=torch.bool)
        self.img_pos = torch.arange(text_len + audio_rows, self.seq_len)
        self.img_update = torch.ones(video_rows, dtype=torch.bool)
        self.signature = (text_len, chunk.video_latents, lat_h, lat_w, chunk.audio_latents)
        self.segments = [(0, text_len, "text"),
                         (text_len, text_len + audio_rows, "audio"),
                         (text_len + audio_rows, self.seq_len, "video")]


def _stream_attention(attn, block, state, x, rope_freqs=None, transformer_options={}):
    """`Attention.forward` (`comfy/ldm/minimax/model.py`) with upstream's routing and cache."""
    s = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    q, k, v = attn.qkv_proj(x).split(heads * head_dim, dim=-1)
    v = v.view(s, heads, head_dim)
    q = q.view(1, s, heads, head_dim)
    k = k.view(1, s, heads, head_dim)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    comfy.quant_ops.ck.rms_rope_split_half_(
        q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    q, k = q[0], k[0]
    text_rows = state["text_rows"]
    cache = state["cache"]
    if cache is not None and cache.committing:
        cache.stage(block, k[text_rows:], v[text_rows:])
    if state["text_sees_all"] or cache is None:
        history_k = history_v = None
    else:
        history_k, history_v = cache.history(block, x.device, k.dtype)
    out = tm.stream_attention(q, k, v, text_rows, history_k, history_v,
                              text_sees_all=state["text_sees_all"])
    return attn.out_proj(out.reshape(s, heads * head_dim))


def _block_patch(block, diffusion_model, state, args, extra):
    state["hook_calls"] = state.get("hook_calls", 0) + 1
    attention = functools.partial(_stream_attention, diffusion_model.blocks[block].attn, block, state)
    return extra["original_block"]({**args, "attention": attention})


def _deviation(reference, candidate, shapes) -> dict:
    """Per stream: the largest absolute difference and the RMS difference over the reference's RMS."""
    out = {}
    for name, ref, got in zip(("video", "audio"), comfy.utils.unpack_latents(reference.float(), shapes),
                              comfy.utils.unpack_latents(candidate.float(), shapes)):
        diff = got - ref
        out[name] = {"max_abs": float(diff.abs().max()),
                     "rel_rms": float(diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)),
                     "exact": bool(torch.equal(got, ref))}
    return out


def _refusals(model_wrap, transformer_options) -> list[str]:
    problems = []
    patcher = model_wrap.model_patcher
    object_keys = [k for k in patcher.object_patches if k.startswith("diffusion_model.")]
    if object_keys:
        problems.append(
            f"the model carries object patches ({object_keys[:3]}{'...' if len(object_keys) > 3 else ''}): "
            "sage, exact blocks or PDD heads. Remove those nodes; the adapter was trained under "
            "dense bf16 attention and this sampler supplies its own")
    for key in ("optimized_attention_override", "sol_compose"):
        if key in transformer_options:
            problems.append(f"transformer_options carries `{key}` (Sol or sage). Remove the node")
    if transformer_options.get("patches_replace", {}).get("dit"):
        problems.append("another block replace patch is installed (VSA or a control)")
    dm = patcher.model.diffusion_model
    if "_forward" in vars(dm) or "rope_freqs" in vars(dm):
        problems.append("the loaded model's forward was replaced by Sol's Morton install, which "
                        "outlives the node; restart the server")
    positive = model_wrap.conds.get("positive") or []
    if len(positive) != 1:
        problems.append(f"expected one positive conditioning, got {len(positive)}")
    if model_wrap.conds.get("negative"):
        problems.append("a negative conditioning is wired; the adapter runs without CFG, use BasicGuider")
    return problems


class TaoMateStreamSampler(comfy.samplers.Sampler):
    def __init__(self, mode: str = "stream", cache_device: str = "cpu_pinned"):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {MODES}")
        if cache_device not in CACHE_DEVICES:
            raise ValueError(f"cache_device {cache_device!r} is not one of {CACHE_DEVICES}")
        self.mode = mode
        self.cache_device = cache_device

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        inner = model_wrap.inner_model
        model_options = extra_args.get("model_options", {})
        base_to = dict(model_options.get("transformer_options", {}))
        problems = _refusals(model_wrap, base_to)
        shapes = inner.latent_shapes
        if shapes is None or len(shapes) != 2:
            problems.append("the latent is not H3's nested video + audio latent")
        want = tm.student_sigmas(tm.SHIFT_VIDEO)
        got = [float(s) for s in sigmas.flatten().tolist()]
        if len(got) != len(want) or max(abs(a - b) for a, b in zip(got, want)) > SIGMA_TOL:
            problems.append(f"sigmas {got} are not the adapter's grid {want} (ManualSigmas "
                            f"{tm.MANUAL_SIGMAS!r})")
        sampling = inner.model_sampling
        if (float(sampling.shift), float(sampling.audio_shift or 0.0)) != (tm.SHIFT_VIDEO, tm.SHIFT_AUDIO):
            problems.append(f"model shift {sampling.shift}/{sampling.audio_shift}, the adapter's is "
                            f"{tm.SHIFT_VIDEO}/{tm.SHIFT_AUDIO}")
        if problems:
            raise ValueError("TaoMate stream sampler refuses this graph:\n  - " + "\n  - ".join(problems))

        positive = model_wrap.conds["positive"][0]["model_conds"]
        context = positive["c_crossattn"].cond
        payload = dict(positive["minimax_payload"].cond)
        if payload.get("keyframes") or payload.get("refs"):
            raise ValueError("TaoMate stream sampler is text to audio-video only: remove references and keyframes")

        state = {"cache": None, "text_rows": int(context.shape[1]),
                 "text_sees_all": self.mode == "verify_whole_clip", "hook_calls": 0}
        dm = inner.diffusion_model
        replace = dict(base_to.get("patches_replace", {}))
        dit = dict(replace.get("dit", {}))
        for i in range(len(dm.blocks)):
            dit[("double_block", i)] = functools.partial(_block_patch, i, dm, state)
        replace["dit"] = dit
        base_to["patches_replace"] = replace

        if self.mode in ("verify_whole_clip", "control_text_only"):
            # The reference is core's own sampler on the same inputs, run first
            # and without the hook, so the expectation comes from outside this file.
            reference = comfy.samplers.ksampler("euler").sample(
                model_wrap, sigmas, dict(extra_args), None, noise, latent_image, denoise_mask, True)
            hooked = self._whole_clip(model_wrap, inner, sigmas, noise, latent_image, context, payload,
                                      shapes, base_to, callback)
            self.last_report = _deviation(reference, hooked, shapes)
            self.last_report["hook_calls"] = state["hook_calls"]
            logging.info("[taomate] %s %s", self.mode, json.dumps(self.last_report, sort_keys=True))
            return hooked
        return self._stream(model_wrap, inner, sigmas, noise, latent_image, context, payload,
                            shapes, base_to, state, callback)

    # -- verify: the stock Euler loop over the whole clip, through the same hook ----
    def _whole_clip(self, model_wrap, inner, sigmas, noise, latent_image, context, payload,
                    shapes, base_to, callback):
        s_in = noise.new_ones([noise.shape[0]])
        x = inner.model_sampling.noise_scaling(sigmas[0], noise, latent_image,
                                               self.max_denoise(model_wrap, sigmas))
        steps = len(sigmas) - 1
        for i in range(steps):
            to = dict(base_to)
            to["sigmas"] = sigmas[i] * s_in
            denoised = inner.apply_model(x, sigmas[i] * s_in, c_crossattn=context,
                                         transformer_options=to, latent_shapes=shapes,
                                         minimax_payload=payload)
            if callback is not None:
                callback(i, denoised, x, steps)
            d = (x - denoised) / sigmas[i]
            x = x + d * (sigmas[i + 1] - sigmas[i])
        return inner.model_sampling.inverse_noise_scaling(sigmas[-1], x)

    # -- stream: upstream's chunk loop ----------------------------------------------
    def _stream(self, model_wrap, inner, sigmas, noise, latent_image, context, payload,
                shapes, base_to, state, callback):
        video_shape, audio_shape = shapes
        latent_t, audio_t = int(video_shape[2]), int(audio_shape[-1])
        requests = tm.requests_for(latent_t)
        if requests is None:
            raise ValueError(
                f"{latent_t} video latents is not a TaoMate run: lengths are "
                f"{[tm.frame_count(r) for r in range(1, 4)]} frames, 124 plus 119 per request")
        plan = tm.run_plan(requests)
        if plan[-1].a1 != audio_t:
            raise ValueError(f"audio latent {audio_t} steps, the run plan ends at {plan[-1].a1}")
        full = payload.get("layout")
        if full is None:
            raise ValueError("the conditioning carries no packed layout")
        text_len, _, lat_h, lat_w, _ = full.signature
        if lat_h != video_shape[3] or lat_w != video_shape[4]:
            raise ValueError("odd latent height or width; use a canvas divisible by 32")
        frame_rows = (lat_h // 2) * (lat_w // 2)

        device = noise.device
        video_noise, audio_noise = comfy.utils.unpack_latents(noise, shapes)
        video_image, audio_image = comfy.utils.unpack_latents(latent_image, shapes)
        audio_scale = float(inner.audio_scale())
        if not bool(audio_image.abs().gt(0).any()):
            raise ValueError("stream mode needs a frozen track in the latent's audio rows "
                             "(MiniMaxH3FreezeAudio): the base-model audio teacher is not built")
        track = audio_image / audio_scale

        store = {"cpu_pinned": ("cpu", True), "cpu": ("cpu", False), "gpu": (str(device), False)}
        store_device, pin = store[self.cache_device]
        cache = tm.StreamCache(len(inner.diffusion_model.blocks), store_device=store_device, pin=pin)
        state["cache"] = cache

        chunk_payload_base = dict(payload)
        chunk_payload_base["audio_scale"] = 1.0
        teacher = tm.teacher_sigmas(tm.SHIFT_AUDIO)
        video_out = video_noise.clone()
        anchor = None
        s_in = noise.new_ones([noise.shape[0]])

        def forward(xv, xa, sigma, layout, commit):
            chunk_payload = dict(chunk_payload_base)
            chunk_payload["layout"] = layout
            shapes_c = [xv.shape, xa.shape]
            x = comfy.utils.pack_latents([xv, xa])[0]
            to = dict(base_to)
            to["sigmas"] = sigma * s_in
            if commit:
                cache.begin_commit()
            try:
                denoised = inner.apply_model(x, sigma * s_in, c_crossattn=context, transformer_options=to,
                                             latent_shapes=shapes_c, minimax_payload=chunk_payload)
            except BaseException:
                cache.abort_commit()
                raise
            if commit:
                cache.finish_commit(2 * xa.shape[-1], xv.shape[2] * frame_rows)
            return comfy.utils.unpack_latents(denoised, shapes_c)[0]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for n, chunk in enumerate(plan):
            started = time.perf_counter()
            if chunk.index == 0 and chunk.request > 0 and chunk.request % tm.AUDIO_RESET_REQUESTS == 0:
                cache.drop_audio()
            positions = tm.chunk_positions(full.position_ids, text_len, audio_t, frame_rows, chunk)
            layout = _ChunkLayout(text_len, chunk, frame_rows, positions, lat_h, lat_w)
            xv = video_noise[:, :, chunk.v0:chunk.v1].clone()
            track_c = track[..., chunk.a0:chunk.a1]
            noise_c = audio_noise[..., chunk.a0:chunk.a1]
            audio_states = [noise_c] + tm.frozen_track_states(track_c, noise_c, teacher)
            for step in range(tm.STEPS):
                sigma, sigma_next = sigmas[step], sigmas[step + 1]
                denoised = forward(xv, audio_states[step], sigma, layout, commit=False)
                xv = denoised + (sigma_next / sigma) * (xv - denoised)
            rows, anchor = tm.match_to_anchor(patchify_video(xv.to(torch.float32)), anchor)
            xv = unpatchify_video(rows, chunk.video_latents, lat_h // 2, lat_w // 2,
                                  int(video_shape[1])).to(xv.dtype)
            forward(xv, audio_states[-1], sigmas[-1], layout, commit=True)
            cache.retain()
            video_out[:, :, chunk.v0:chunk.v1] = xv
            peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            logging.info("[taomate] request %d chunk %d: latents %d-%d, audio %d-%d, cache %d tokens, "
                         "%.1f s, peak allocated %.2f GiB", chunk.request, chunk.index, chunk.v0, chunk.v1,
                         chunk.a0, chunk.a1, cache.tokens, time.perf_counter() - started, peak)
            if callback is not None:
                packed = comfy.utils.pack_latents([video_out, audio_image])[0]
                callback(n, packed, packed, len(plan))
        state["cache"] = None
        return comfy.utils.pack_latents([video_out, audio_image])[0].to(device)


class MiniMaxH3TaoMateStreamSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TaoMateStreamSampler",
            display_name="MiniMax H3 TaoMate Stream Sampler",
            category="sampling/custom_sampling/samplers",
            description=(
                "TaoLiveAIGC's TaoMate-H3 streaming runtime on one card: the video is generated in "
                "causal chunks against a cache of earlier chunks, with a frozen track as the audio. "
                "Wire into SamplerCustomAdvanced with BasicGuider, the TaoMate LoRA on the plain fl2va "
                "checkpoint, the adapter's ManualSigmas, and no sage, Sol or PDD nodes. "
                "docs/h3_taomate.md section 7."),
            inputs=[
                io.Combo.Input("mode", options=list(MODES), default="stream",
                               tooltip=("stream: the chunked, cached runtime. verify_whole_clip: the stock "
                                        "Euler loop over the whole clip through the same attention hook, "
                                        "for the equality check; not a way to render.")),
                io.Combo.Input("cache_device", options=list(CACHE_DEVICES), default="cpu_pinned",
                               tooltip=("Where committed K/V live between chunks. cpu_pinned: host memory, "
                                        "streamed to the card per block, the only choice that fits at a "
                                        "trained canvas. gpu: small canvases only.")),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, mode, cache_device) -> io.NodeOutput:
        return io.NodeOutput(TaoMateStreamSampler(mode, cache_device))
