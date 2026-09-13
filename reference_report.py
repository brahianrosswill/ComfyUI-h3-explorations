"""What an ordered H3 reference list costs, before anything is encoded.

Every reference is read twice. The video VAE encodes one copy into reference
latent rows that the DiT attends on every sampling step; Qwen3-VL reads a copy
as vision tokens that sit in the text segment ahead of the prompt, and the
hidden states of those positions ride the DiT's text segment on every step
too. `MiniMaxH3AppendRefImage.size_policy` sizes the first copy and its
`qwen_view` the second, and nothing in the graph showed either until this
node: the numbers lived in a server log line and in `bench/preflight_graph.py`,
which nobody runs from the UI.

`MiniMaxH3ReferenceReport` takes the same references, prompt and canvas the
conditioner takes, prices both copies of every reference with the SAME sizing
functions the conditioner calls (`reference_geometry.fit_reference_image`,
`reference_conditioning.qwen_view_size`, `reference_geometry.qwen_image_size`),
builds the packed sequence the model will build (`comfy.ldm.minimax.model
.PackedLayout`, from shapes alone) and returns a text report and a picture of
it. No VAE, no encoder forward: the only model it touches is the tokenizer,
for the prompt's token count.

**What is exact and what is not.** Every pixel size, every DiT row count and
every vision-token count is the geometry the conditioner will produce, from
the same functions. The prompt and label token counts come from the installed
tokenizer. Reference audio rows are an estimate: the aligned encode pads to
the audio VAE's hop, which this node does not load, so a soundtrack's rows can
be one hop off. The report says so on the line.

**Two ceilings apply to the encoder's copy, in order.** The selected
`image_policy` pre-applies its own bounds (none for `comfy`, the release's
declaration for `release`), and then the loaded encoder's own processor
applies core's `process_qwen2vl_images` defaults inside `preprocess_embed`.
The report shows the copy after both, because that is what lands in the
sequence, and says on the line when the second stage moved it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
from comfy_api.latest import io, ui
from comfy_extras.nodes_minimax_h3 import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    adapt_canvas,
    temporal_shape,
    video_latent_t,
)

from .reference_geometry import (
    IMAGE_POLICIES,
    fit_reference_image,
    latent_rows,
    qwen_image_size,
)

logger = logging.getLogger(__name__)

#: Merged patch: `patch_size * merge_size` of the installed Qwen3-VL processor.
#: Inherited from `comfy/text_encoders/qwen3vl.py`'s call site (16) and
#: `process_video_block`'s defaults (merge 2); one merged token per 32x32.
MERGED_PATCH = 32


@dataclass
class StillPricing:
    index: int
    label: str
    source: tuple[int, int]
    vae: tuple[int, int]
    dit_rows: int
    qwen: tuple[int, int]
    qwen_tokens: int
    separate: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class VideoPricing:
    index: int
    label: str
    source: tuple[int, int]
    prepared_frames: int
    vae: tuple[int, int]
    latent_t: int
    dit_rows: int
    qwen: tuple[int, int]
    sampled: int
    qwen_tokens: int
    has_soundtrack: bool
    audio_rows: int
    notes: list[str] = field(default_factory=list)


@dataclass
class AudioPricing:
    index: int
    label: str
    audio_rows: int
    notes: list[str] = field(default_factory=list)


@dataclass
class SequencePricing:
    canvas: tuple[int, int]
    frames: int
    latent_t: int
    image_policy: str
    video_policy: str
    encoder_bounds: tuple[int, int]
    items: list
    prompt_tokens: int | None
    label_tokens: int
    vision_tokens: int
    text_len: int
    segments: dict
    total: int

    @property
    def dit_reference_rows(self) -> int:
        return sum(getattr(i, "dit_rows", 0) for i in self.items)

    @property
    def prompt_share(self) -> float | None:
        if self.prompt_tokens is None or self.text_len == 0:
            return None
        return self.prompt_tokens / self.text_len


def _native_encoder_bounds(clip) -> tuple[int, int]:
    """The pixel bounds the loaded encoder's own processor applies to a still.

    `MiniMaxH3EncoderLoader` records them on the transformer as
    `_h3_image_bounds`; a CLIP from core's `CLIPLoader` carries nothing and
    runs the same code path, so the defaults are read out of core. Never
    typed here.
    """
    model = clip
    for attribute in ("cond_stage_model", "qwen3vl_32b", "transformer"):
        model = getattr(model, attribute, None)
        if model is None:
            break
    bounds = getattr(model, "_h3_image_bounds", None) if model is not None else None
    if bounds is not None:
        return int(bounds[0]), int(bounds[1])
    import inspect
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images
    params = inspect.signature(process_qwen2vl_images).parameters
    return int(params["min_pixels"].default), int(params["max_pixels"].default)


def _smart_resize(width: int, height: int, bounds: tuple[int, int]) -> tuple[int, int]:
    """The encoder's own resize, imported rather than restated."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    h, w = smart_resize(height=height, width=width, factor=MERGED_PATCH,
                        min_pixels=bounds[0], max_pixels=bounds[1])
    return int(w), int(h)


def merged_tokens(width: int, height: int) -> int:
    return (width // MERGED_PATCH) * (height // MERGED_PATCH)


def _pair_grid(width: int, height: int, bounds: tuple[int, int]) -> tuple[int, int]:
    """Core's per-pair video block sizing (`comfy/text_encoders/minimax.py`
    `process_video_block`), on shapes alone."""
    factor = MERGED_PATCH
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > bounds[1]:
        beta = math.sqrt((height * width) / bounds[1])
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < bounds[0]:
        beta = math.sqrt(bounds[0] / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar, h_bar


def _count_text_tokens(clip, text: str) -> int:
    tokens = clip.tokenize(text)
    batches = tokens["qwen3vl_32b"]
    return sum(len(batch) for batch in batches)


def price_references(records, width: int, height: int, length: int,
                     image_policy: str = "comfy", video_policy: str = "comfy",
                     clip=None, prompt: str = "") -> SequencePricing:
    """Price every record the way `_compile_reference_records` will size it."""
    from . import reference_conditioning as rc
    from .reference_order import assign_labels

    if image_policy not in IMAGE_POLICIES:
        raise ValueError(f"unknown image_policy {image_policy!r}")
    if video_policy not in rc.VIDEO_POLICIES:
        raise ValueError(f"unknown video_policy {video_policy!r}")

    frame_count, latent_t, audio_t = temporal_shape(length)
    duration = frame_count / FPS
    bounds = _native_encoder_bounds(clip) if clip is not None else (3136, 12845056)
    labels = assign_labels(rc._order_records(records))
    items = []
    blocks = []
    label_texts = []
    vision_tokens = 0

    for index, record in enumerate(records):
        label = labels[index] if index < len(labels) else f"#{index + 1}"
        if isinstance(record, rc.RuntimeImageReference):
            _, sh, sw = rc._image_shape(record.image, f"reference image {index + 1}")
            role_w, role_h = fit_reference_image(
                sw, sh, size_policy=record.size_policy,
                short_edge=record.short_edge, allow_upscale=record.allow_upscale,
                canvas_w=width, canvas_h=height)
            notes = []
            separate = bool(record.qwen_short_edge)
            if not separate:
                vae_w, vae_h = role_w, role_h
                if image_policy != "comfy":
                    vae_w, vae_h = qwen_image_size(role_w, role_h, image_policy)
                    if (vae_w, vae_h) != (role_w, role_h):
                        notes.append(f"{image_policy} bounds moved both copies "
                                     f"from {role_w}x{role_h}")
                qwen_w, qwen_h = vae_w, vae_h
            else:
                vae_w, vae_h = role_w, role_h
                qwen_w, qwen_h = rc.qwen_view_size(sw, sh, record.qwen_short_edge)
                if image_policy != "comfy":
                    pre = qwen_image_size(qwen_w, qwen_h, image_policy)
                    if pre != (qwen_w, qwen_h):
                        notes.append(f"{image_policy} bounds moved the encoder "
                                     f"copy from {qwen_w}x{qwen_h}")
                    qwen_w, qwen_h = pre
            # The encoder's own processor, after the policy.
            enc_w, enc_h = _smart_resize(qwen_w, qwen_h, bounds)
            if (enc_w, enc_h) != (qwen_w, qwen_h):
                notes.append(f"encoder's own bounds moved its copy from "
                             f"{qwen_w}x{qwen_h}")
                qwen_w, qwen_h = enc_w, enc_h
            if record.size_policy == "max":
                scale = record.short_edge / min(sw, sh)
                if scale > 1.0 and not record.allow_upscale:
                    notes.append(f"dit_short_edge {record.short_edge} inert: "
                                 f"source short edge {min(sw, sh)} is under it "
                                 f"and allow_upscale is off")
                elif scale > 1.0:
                    notes.append(f"upscaled x{scale:.2f} on each side "
                                 f"(x{scale * scale:.1f} rows)")
                elif scale < 1.0:
                    notes.append(f"shrunk to the {record.short_edge} short edge")
            else:
                notes.append("match: capped at the canvas area, never enlarged")
            if not separate:
                notes.append("shared: the encoder reads the video model's copy")
            rows = latent_rows(vae_w, vae_h)
            tokens = merged_tokens(qwen_w, qwen_h)
            vision_tokens += tokens
            items.append(StillPricing(index, label, (sw, sh), (vae_w, vae_h), rows,
                                      (qwen_w, qwen_h), tokens, separate, notes))
            blocks.append({"kind": "image", "latent_h": vae_h // 16,
                           "latent_w": vae_w // 16})
            label_texts.append(f"{label}: ")
            continue

        if isinstance(record, rc.RuntimeVideoReference):
            frames = rc._prepare_reference_video(record.frames, record.loaded_fps,
                                                 frame_count)
            n = int(frames.shape[0])
            _, sh, sw = rc._image_shape(frames, f"reference video {index + 1}")
            cw, ch = adapt_canvas(sw, sh)
            notes = []
            if video_policy == "comfy" and sw * sh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(sw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(sh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                notes.append("comfy: a video under the canvas is not enlarged")
            elif (cw, ch) != (sw, sh):
                notes.append(f"{video_policy}: fitted to the release canvas rule")
            lt = video_latent_t(n)
            rows = lt * latent_rows(cw, ch)
            sampled = len(range(0, n, FPS // 2))
            pairs = (sampled + 1) // 2
            if video_policy == "release":
                qw, qh = rc._configured_qwen_video_size(sampled, cw, ch, "release")
                tokens = pairs * merged_tokens(qw, qh)
                notes.append("release: the encoder's clip-wide budget sized the "
                             "2 fps samples")
            else:
                qw, qh = _pair_grid(cw, ch, bounds)
                tokens = pairs * merged_tokens(qw, qh)
            vision_tokens += tokens
            audio_rows = 0
            soundtrack = record.soundtrack
            has_sound = soundtrack is not None
            if soundtrack is not None:
                wave = soundtrack["waveform"]
                rate = int(soundtrack["sample_rate"])
                seconds = min(float(wave.shape[-1]) / rate, duration)
                audio_rows = int(round(seconds * AUDIO_LATENT_FPS)) * 2
                notes.append("soundtrack rows estimated to the target duration; "
                             "the aligned encode can add one VAE hop")
            items.append(VideoPricing(index, label, (sw, sh), n, (cw, ch), lt, rows,
                                      (qw, qh), sampled, tokens, has_sound,
                                      audio_rows, notes))
            blocks.append({"kind": "video_audio" if audio_rows else "video",
                           "latent_t": lt, "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": audio_rows // 2})
            label_texts.append(f"{label}: ")
            if has_sound:
                label_texts.append("<Audio 1>: ")
            label_texts.extend(f"<{(i + 0.5):.1f} seconds>" for i in range(pairs))
            continue

        if isinstance(record, rc.RuntimeAudioReference):
            wave = record.audio["waveform"]
            rate = int(record.audio["sample_rate"])
            seconds = min(float(wave.shape[-1]) / rate, duration)
            audio_rows = int(round(seconds * AUDIO_LATENT_FPS)) * 2
            items.append(AudioPricing(index, label, audio_rows, [
                "rows estimated to the target duration; the aligned encode can "
                "add one VAE hop",
                "the encoder sees only the label; audio needs the audio VAE to "
                "reach the DiT"]))
            blocks.append({"kind": "audio", "ref_audio_t": audio_rows // 2})
            label_texts.append(f"{label}: ")
            continue
        raise TypeError(f"not a runtime reference record: {record!r}")

    prompt_tokens = None
    label_tokens = 0
    if clip is not None:
        prompt_tokens = _count_text_tokens(clip, prompt) if prompt else 0
        label_tokens = sum(_count_text_tokens(clip, t) for t in label_texts)
    # <|vision_start|> and <|vision_end|> around every vision block.
    vision_blocks = sum(1 for t in label_texts if t.startswith("<Picture")) + sum(
        1 for t in label_texts if t.endswith("seconds>"))
    text_len = (prompt_tokens or 0) + label_tokens + vision_tokens + 2 * vision_blocks

    from comfy.ldm.minimax.model import PackedLayout
    lat_h = (height // 16 + 1) // 2 * 2
    lat_w = (width // 16 + 1) // 2 * 2
    layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t, refs=blocks)
    segments: dict[str, int] = {}
    for a, b, kind in layout.segments:
        segments[kind] = segments.get(kind, 0) + (b - a)

    return SequencePricing(
        canvas=(width, height), frames=frame_count, latent_t=latent_t,
        image_policy=image_policy, video_policy=video_policy,
        encoder_bounds=bounds, items=items, prompt_tokens=prompt_tokens,
        label_tokens=label_tokens, vision_tokens=vision_tokens,
        text_len=text_len, segments=segments, total=layout.seq_len)


def format_report(p: SequencePricing) -> str:
    """The text form. Monospace; one block per reference, then the sequence."""
    out = [f"MiniMax H3 reference report  --  {p.canvas[0]}x{p.canvas[1]}, "
           f"{p.frames} frames ({p.latent_t} latent), image_policy={p.image_policy}, "
           f"video_policy={p.video_policy}",
           f"encoder still bounds {p.encoder_bounds[0]:,}..{p.encoder_bounds[1]:,} px",
           ""]
    for it in p.items:
        if isinstance(it, StillPricing):
            out.append(f"{it.label}  still {it.source[0]}x{it.source[1]}")
            out.append(f"    video model sees  {it.vae[0]}x{it.vae[1]:<6} "
                       f"{it.dit_rows:>7,} DiT rows, attended every step")
            out.append(f"    text encoder sees {it.qwen[0]}x{it.qwen[1]:<6} "
                       f"{it.qwen_tokens:>7,} vision tokens, ahead of the prompt")
        elif isinstance(it, VideoPricing):
            out.append(f"{it.label}  video {it.source[0]}x{it.source[1]}, "
                       f"{it.prepared_frames} frames after 24 fps prep")
            out.append(f"    video model sees  {it.vae[0]}x{it.vae[1]} x {it.latent_t} "
                       f"latent frames  {it.dit_rows:>7,} DiT rows")
            out.append(f"    text encoder sees {it.qwen[0]}x{it.qwen[1]} x {it.sampled} "
                       f"samples at 2 fps  {it.qwen_tokens:>7,} vision tokens")
            if it.has_soundtrack:
                out.append(f"    soundtrack        {it.audio_rows:>7,} audio rows (estimate)")
        else:
            out.append(f"{it.label}  audio  {it.audio_rows:,} rows (estimate)")
        for note in it.notes:
            out.append(f"      - {note}")
        out.append("")
    out.append("packed sequence the DiT attends on every step:")
    order = [("video", "target video"), ("audio", "target audio"),
             ("text", "text segment"), ("ref_img", "reference stills (latent rows)"),
             ("ref_video", "reference video (latent rows)"),
             ("ref_audio", "reference audio")]
    seen = set()
    for kind, name in order:
        if kind in p.segments:
            seen.add(kind)
            out.append(f"    {name:<34} {p.segments[kind]:>9,}")
    for kind, n in p.segments.items():
        if kind not in seen:
            out.append(f"    {kind:<34} {n:>9,}")
    out.append(f"    {'TOTAL':<34} {p.total:>9,}")
    out.append("")
    if p.prompt_tokens is None:
        out.append("text segment: prompt not counted (no CLIP wired); "
                   f"{p.vision_tokens:,} vision tokens + {p.label_tokens:,} label tokens")
    else:
        out.append(f"text segment: {p.prompt_tokens:,} prompt tokens + "
                   f"{p.label_tokens:,} label tokens + {p.vision_tokens:,} vision "
                   f"tokens (+ delimiters) = {p.text_len:,}")
        share = p.prompt_share or 0.0
        out.append(f"the prompt is {share:.0%} of what the text encoder reads; the "
                   f"rest is references placed ahead of it")
    return "\n".join(out)


# --------------------------------------------------------------------------
# The picture


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: bitmap default only
        return ImageFont.load_default()


def _thumb(image_tensor, box_w: int, box_h: int):
    """A PIL thumbnail of a [1,H,W,C] float tensor, fitted inside the box."""
    from PIL import Image
    arr = (image_tensor[0, ..., :3].clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy())
    im = Image.fromarray(arr)
    im.thumbnail((max(1, box_w), max(1, box_h)))
    return im


def render_report_image(p: SequencePricing, records) -> torch.Tensor:
    """Two columns, one row per reference, then the sequence bar."""
    from PIL import Image, ImageDraw

    W = 1400
    margin = 28
    col_w = (W - 3 * margin) // 2
    row_gap = 26
    max_box = 300
    f_title = _font(30)
    f_head = _font(20)
    f_body = _font(17)
    f_small = _font(14)

    ink = (24, 24, 28)
    muted = (110, 110, 118)
    paper = (248, 247, 244)
    col_dit = (66, 133, 244)
    col_enc = (219, 68, 55)
    col_video = (170, 170, 178)
    col_audio = (120, 190, 140)
    col_text = (244, 180, 0)

    # Scale thumbnails by the square root of pixel area, against the largest
    # copy on the page, so a copy twice the rows draws about 1.4x the side.
    areas = []
    for it in p.items:
        if isinstance(it, (StillPricing, VideoPricing)):
            areas.append(it.vae[0] * it.vae[1])
            areas.append(it.qwen[0] * it.qwen[1])
    ref_area = max(areas) if areas else 1

    def box_for(w, h):
        s = math.sqrt((w * h) / ref_area)
        side = max(48, int(max_box * s))
        if w >= h:
            return side, max(24, int(side * h / w))
        return max(24, int(side * w / h)), side

    # Pre-compute row heights.
    rows = []
    for it, rec in zip(p.items, records):
        if isinstance(it, StillPricing):
            bw1, bh1 = box_for(*it.vae)
            bw2, bh2 = box_for(*it.qwen)
            text_h = 24 * 2 + 18 * len(it.notes)
            rows.append(max(bh1, bh2) + text_h + row_gap)
        elif isinstance(it, VideoPricing):
            bw1, bh1 = box_for(*it.vae)
            bw2, bh2 = box_for(*it.qwen)
            text_h = 24 * 3 + 18 * len(it.notes)
            rows.append(max(bh1, bh2) + text_h + row_gap)
        else:
            rows.append(24 * 2 + 18 * len(it.notes) + row_gap)
    header_h = 130
    bar_h = 200
    H = header_h + sum(rows) + bar_h + margin
    im = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(im)

    d.text((margin, 20), "MiniMax H3 reference report", fill=ink, font=f_title)
    d.text((margin, 60), f"{p.canvas[0]}x{p.canvas[1]}, {p.frames} frames, "
                         f"image_policy={p.image_policy}, video_policy={p.video_policy}",
           fill=muted, font=f_body)
    x1 = margin
    x2 = 2 * margin + col_w
    d.text((x1, 92), "VIDEO MODEL (DiT) sees   -- latent rows, every sampling step",
           fill=col_dit, font=f_head)
    d.text((x2, 92), "TEXT ENCODER (Qwen3-VL) sees   -- vision tokens, ahead of the prompt",
           fill=col_enc, font=f_head)
    d.line((x1, 120, W - margin, 120), fill=(200, 200, 200), width=1)

    y = header_h
    for it, rec, rh in zip(p.items, records, rows):
        d.text((x1, y), it.label, fill=ink, font=f_head)
        if isinstance(it, (StillPricing, VideoPricing)):
            src = rec.image if isinstance(it, StillPricing) else rec.frames[:1]
            bw1, bh1 = box_for(*it.vae)
            bw2, bh2 = box_for(*it.qwen)
            top = y + 26
            t1 = _thumb(src, bw1, bh1)
            im.paste(t1, (x1, top))
            d.rectangle((x1, top, x1 + t1.width, top + t1.height), outline=col_dit, width=2)
            t2 = _thumb(src, bw2, bh2)
            im.paste(t2, (x2, top))
            d.rectangle((x2, top, x2 + t2.width, top + t2.height), outline=col_enc, width=2)
            base = top + max(t1.height, t2.height) + 6
            if isinstance(it, StillPricing):
                d.text((x1, base), f"{it.vae[0]}x{it.vae[1]}   {it.dit_rows:,} rows",
                       fill=ink, font=f_body)
                d.text((x2, base), f"{it.qwen[0]}x{it.qwen[1]}   {it.qwen_tokens:,} tokens",
                       fill=ink, font=f_body)
                ny = base + 24
            else:
                d.text((x1, base), f"{it.vae[0]}x{it.vae[1]} x {it.latent_t} latent "
                                   f"frames   {it.dit_rows:,} rows", fill=ink, font=f_body)
                d.text((x2, base), f"{it.qwen[0]}x{it.qwen[1]} x {it.sampled} samples"
                                   f"   {it.qwen_tokens:,} tokens", fill=ink, font=f_body)
                ny = base + 24
                if it.has_soundtrack:
                    d.text((x1, ny), f"soundtrack   {it.audio_rows:,} audio rows",
                           fill=ink, font=f_body)
                ny += 24
            d.text((x1, ny - 2), f"source {it.source[0]}x{it.source[1]}", fill=muted,
                   font=f_small)
            ny += 18
        else:
            d.text((x1, y + 26), f"audio   {it.audio_rows:,} rows", fill=ink, font=f_body)
            ny = y + 50
        for note in it.notes:
            d.text((x2, ny), f"- {note}", fill=muted, font=f_small)
            ny += 18
        y += rh
        d.line((x1, y - 10, W - margin, y - 10), fill=(225, 225, 225), width=1)

    # The packed sequence.
    y += 4
    d.text((x1, y), f"packed sequence, {p.total:,} rows attended on every step",
           fill=ink, font=f_head)
    y += 32
    bar_w = W - 2 * margin
    parts = [("target video", p.segments.get("video", 0), col_video),
             ("target audio", p.segments.get("audio", 0), col_audio),
             ("text: prompt + labels", (p.prompt_tokens or 0) + p.label_tokens, col_text),
             ("text: reference vision tokens", p.vision_tokens, col_enc),
             ("reference latent rows",
              p.segments.get("ref_img", 0) + p.segments.get("ref_video", 0), col_dit),
             ("reference audio", p.segments.get("ref_audio", 0), (90, 150, 110))]
    total = max(1, sum(n for _, n, _ in parts))
    x = x1
    for name, n, color in parts:
        w = int(bar_w * n / total)
        if w > 0:
            d.rectangle((x, y, x + w, y + 34), fill=color)
        x += w
    y += 44
    x = x1
    for name, n, color in parts:
        if n <= 0:
            continue
        d.rectangle((x, y + 3, x + 12, y + 15), fill=color)
        label = f"{name} {n:,} ({n / total:.0%})"
        d.text((x + 18, y), label, fill=ink, font=f_small)
        x += 18 + int(d.textlength(label, font=f_small)) + 22
        if x > W - 260:
            x = x1
            y += 20
    y += 30
    if p.prompt_tokens is not None and p.text_len:
        share = p.prompt_share or 0.0
        d.text((x1, y), f"the prompt's share of what the text encoder reads: {share:.0%}",
               fill=ink, font=f_body)
        y += 26
        d.rectangle((x1, y, x1 + bar_w, y + 16), fill=(225, 225, 225))
        d.rectangle((x1, y, x1 + int(bar_w * share), y + 16), fill=col_text)
    else:
        d.text((x1, y), "wire the CLIP to count the prompt's tokens", fill=muted,
               font=f_body)

    arr = torch.from_numpy(__import__("numpy").asarray(im)).to(torch.float32) / 255.0
    return arr.unsqueeze(0)


class MiniMaxH3ReferenceReport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from .reference_conditioning import H3References, VIDEO_POLICIES
        return io.Schema(
            node_id="MiniMaxH3ReferenceReport",
            display_name="MiniMax H3 Reference Report",
            category="MiniMaxH3/references",
            description=(
                "What the appended references will cost, before anything is "
                "encoded. Every still is read twice: the video model (DiT) "
                "attends its latent rows on every sampling step, and the text "
                "encoder (Qwen3-VL) reads it as vision tokens placed ahead of "
                "your prompt. This node prices both copies of every reference "
                "with the conditioner's own sizing, builds the packed sequence "
                "the model will build, and shows the prompt's share of what the "
                "encoder reads. Wire the same references, prompt, canvas and "
                "policies you give the conditioner; connect report_image to a "
                "Preview Image node."
            ),
            inputs=[
                H3References.Input("references"),
                io.Clip.Input(
                    "clip", optional=True,
                    tooltip="The text encoder, for the prompt's token count. "
                            "Without it the prompt is not counted."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                optional=True, default=""),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "image_policy", options=list(IMAGE_POLICIES), default="comfy",
                    optional=True,
                    tooltip="Match the conditioner's image_policy."),
                io.Combo.Input(
                    "video_policy", options=list(VIDEO_POLICIES), default="comfy",
                    optional=True,
                    tooltip="Match the conditioner's video_policy."),
            ],
            outputs=[
                io.Image.Output(display_name="report_image"),
                io.String.Output(display_name="report_text"),
            ],
        )

    @classmethod
    def execute(cls, references, clip=None, prompt="", width=1344, height=768,
                length=124, image_policy="comfy", video_policy="comfy"):
        from .reference_conditioning import _reference_tuple
        records = _reference_tuple(references)
        if not records:
            raise ValueError("MiniMaxH3ReferenceReport needs at least one appended reference")
        pricing = price_references(records, width, height, length,
                                   image_policy=image_policy,
                                   video_policy=video_policy,
                                   clip=clip, prompt=prompt or "")
        text = format_report(pricing)
        logger.info("[h3] reference report:\n%s", text)
        image = render_report_image(pricing, records)
        return io.NodeOutput(image, text, ui=ui.PreviewText(text))
