"""Drop a song and a prompt; the windows come from the track.

`docs/h3_audio_freeze.md` section 4 step 6, the loop: the LTX pack's
music-video mode on H3. One prompt (or a few, separated by `---`), one track,
and the node plans the windows from the track's length, runs them one after
another inside itself, writes each window's new frames to a file as it goes,
and joins the files with the full track at the end. Nothing here holds more
than one window of frames. The shot-per-window workflows that chained window
nodes by hand retired unrendered on 2026-09-14; prompt blocks with `frames:`
lines do what they did.

**The plan.** Windows are `window_frames` long with `context_frames` of the
previous window frozen at their head, so each adds `window - context` frames.
The last window is the smallest length on both clocks (141, 192, 243, 294,
345 frames) that reaches the end of the track; if the track runs out inside
it, the freeze pads silence and the report says how much. `extent` is the
whole track, or its first N seconds for a quick look.

**The prompt.** One block is used for every window (`uniform`). Several
blocks separated by a line of `---` are used in order with the last one
repeating (`cycle`, the default when there are several), or one is drawn per
window from the seed (`random`); a block may start with a line `frames: N` to
set that window's length (on both clocks).

**Lists.** A `__name__` in the prompt takes a value per window from a Prompt
List chained into `lists`, or from a wildcard file of that name
(`prompt_lists.py`, which holds the owner's rules). A list advances only for a
window whose text uses the name, and filling happens during planning, so every
key and encode below sees the filled-in text.

**Everything is encoded before anything samples.** The track once, and each
distinct prompt once. A window's conditioning is a function of its text and
its references alone -- not its seed, start or the previous window -- because
continuity reaches the next window as the previous window's latent tail frozen
in, never through the encoder. So Qwen3-VL and the DiT trade places on the
card once per run rather than once per window, and a song on one prompt
encodes one prompt. With references the frame count joins the key: the
reference compiler is handed it, and a shorter last window costs at most one
more encode. With resume, only the windows that render are encoded for, and a
run that reuses every window never loads the encoder.

**References.** `references` takes an Append Ref Image chain and presents it
with every window's prompt through `MiniMaxH3ReferenceConditioning`'s own
execute, so the stills are fitted, VAE-encoded and labelled exactly as on the
reference graphs. The prompt then follows the reference prompt format
(`docs/prompting.md` section 2.2). The fl2va checkpoint takes references
(owner, 2026-09-14).

**Sampling** is what SamplerCustomAdvanced does, per window: a BasicGuider on
the model with the window's conditioning, prepared noise at `seed + i`, the
given sampler and sigmas, the nested noise mask from the window node.

**Files and resume.** `loop_output.py` and `loop_resume.py`: window files in
`<prefix>_windows/`, each video with its sampled latent beside it, overwritten
in place by the next run of the graph. A run reuses the stored windows whose
inputs have not changed, in order, and renders from the first that has
(`reuse_windows`); `loop_resume.py` says what a window's key covers and what it
cannot see. The seed holds after each queue so a re-queue can reuse.
`keep_windows` off removes this run's window files after the join. The
finished `<prefix>_NNNNN.mp4` carries the prompt and workflow;
`save_metadata_png` adds the first frame as a PNG with the same.

First run: `bench/results/2026-09-12_audio_freeze_song_smoke.jsonl`, one short
window; `bench/results/2026-09-14_audio_freeze_song_stage1_smoke.jsonl`, two
short windows with and without references. Resume arrived on 2026-09-14 after
that; its first run is a throwaway.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import random
import subprocess

import torch
from comfy_api.latest import io

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Guider_Basic
from comfy_extras.nodes_minimax_h3 import FPS, _empty_av_latent

from .audio_freeze import (MiniMaxH3EncodeTrack, MiniMaxH3FreezeAudioWindow,
                           _ffmpeg, _stereo, audio_grid)
from .conditioning import MiniMaxH3Conditioning
from . import loop_resume
from .prompt_lists import H3PromptLists, find_wildcard, lists_for, resolve_texts
from .loop_output import join_and_mux, saved_outputs, window_dir, write_metadata_png
from .reference_conditioning import H3References, MiniMaxH3ReferenceConditioning

logger = logging.getLogger(__name__)

# Lengths on both clocks: video runs (17k + 5) whose frame count is a multiple
# of 3, so the audio slice is whole latent steps. 39 + 51k.
CHAIN_LENGTHS = (141, 192, 243, 294, 345)


def plan_windows(total_frames: int, window_frames: int, context_frames: int, rng=None) -> list[int]:
    """Frame counts per window covering `total_frames`, each on both clocks.

    With `rng`, every window but the last draws its length from the lengths
    on both clocks at or below `window_frames` (and longer than the context),
    so the same prompt can be tested under uniform and non-uniform windows.
    """
    if window_frames not in CHAIN_LENGTHS:
        raise ValueError(f"window_frames {window_frames} is not on both clocks; use one of {CHAIN_LENGTHS}")
    if context_frames % 17 != 5 or (context_frames * 5) % 3 != 0 or context_frames >= window_frames:
        raise ValueError(f"context_frames {context_frames} must be 39, 90 or 141 and shorter than the window")
    if total_frames <= 0:
        raise ValueError("the track is empty")
    choices = [n for n in CHAIN_LENGTHS if n <= window_frames and n > context_frames]
    first = rng.choice(choices) if rng is not None else window_frames
    plan = [first]
    covered = first
    while covered < total_frames:
        remaining = total_frames - covered
        fit = [n for n in CHAIN_LENGTHS if n > context_frames and n - context_frames >= remaining]
        if fit:
            n = min(fit)
        elif rng is not None:
            n = rng.choice(choices)
        else:
            n = window_frames
        plan.append(n)
        covered += n - context_frames
    return plan


def parse_prompt_blocks(text: str) -> list[tuple[int | None, str]]:
    """Blocks separated by a `---` line; an optional leading `frames: N` line per block."""
    blocks = []
    for raw in text.replace("\r\n", "\n").split("\n---\n"):
        body = raw.strip("\n")
        if not body.strip():
            continue
        frames = None
        first, _, rest = body.partition("\n")
        if first.strip().lower().startswith("frames:"):
            frames = int(first.split(":", 1)[1].strip())
            body = rest
        blocks.append((frames, body.strip("\n")))
    if not blocks:
        raise ValueError("the prompt is empty")
    return blocks


def _write_frames_mp4(path: str, images: torch.Tensor, crf: int) -> int:
    """[T, H, W, 3] float in [0, 1] to an H.264 mp4 through ffmpeg's rawvideo pipe."""
    t, h, w, _ = images.shape
    cmd = [_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for i in range(0, t, 24):
            chunk = (images[i:i + 24].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy().tobytes()
            proc.stdin.write(chunk)
    finally:
        proc.stdin.close()
        err = proc.stderr.read()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {err.decode(errors='replace')[-400:]}")
    return int(t)


class MiniMaxH3AudioFreezeSong(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioFreezeSong",
            is_output_node=True,
            display_name="MiniMax H3 Audio Freeze Song (whole track)",
            category="model/latent/minimax",
            description=(
                "Drop a track and a prompt: the node plans windows from the track's length, "
                "encodes the track and every distinct prompt once, then runs the windows in "
                "sequence with the previous window's tail frozen as context and the track's slice "
                "frozen in each, writes each window's new frames to <prefix>_windows/ as it goes, "
                "and joins the files with the full track, reusing unchanged windows from the last "
                "run. One prompt for every window, or blocks "
                "separated by a `---` line; optional reference stills go with every window. "
                "docs/h3_audio_freeze.md."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="Video VAE."),
                io.Vae.Input("audio_vae"),
                io.Audio.Input("audio", tooltip="The whole track."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="One prompt for every window, or blocks separated by a line of ---; a block may start with `frames: N`."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("window_frames", default=345, min=141, max=345, step=51,
                             tooltip="Frames per window, on both clocks: 141, 192, 243, 294 or 345."),
                io.Int.Input("context_frames", default=39, min=39, max=141, step=51,
                             tooltip="Frames of the previous window frozen at the head of the next: 39, 90 or 141."),
                # A DynamicCombo, not a Float whose 0 meant "the whole track":
                # the owner's rule (2026-09-13) is that a number never means a
                # mode. Selection first, then the option's own widget.
                io.DynamicCombo.Input(
                    "extent",
                    options=[
                        io.DynamicCombo.Option("whole", []),
                        io.DynamicCombo.Option("first_seconds", [
                            io.Float.Input("seconds", default=30.0, min=0.5, max=36000.0, step=0.5,
                                           tooltip="Cover only the first N seconds of the track."),
                        ]),
                    ],
                    tooltip=("How much of the track to cover. whole: every window the plan needs "
                             "to reach the end. first_seconds: only the first N seconds, for a "
                             "quick look at the seams before committing to the whole song.")),
                # control_after_generate declared, not left to the frontend: it
                # draws a control widget for any INT named `seed` whether or not
                # the schema asks (`useIntWidget.ts`), and until 2026-09-14 the
                # shipped UI graphs wrote no value for it, shifting every widget
                # after this one. Fixed, not the frontend's randomize: a new seed
                # after every queue would re-render every window, which resume
                # (`reuse_windows`) exists to avoid (owner, 2026-09-14).
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=io.ControlAfterGenerate.fixed),
                io.Float.Input("audio_mask", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("level", options=["clip_guard", "peak", "none"], default="clip_guard"),
                io.String.Input("filename_prefix", default="Video/h3_song"),
                io.Int.Input("crf", default=19, min=0, max=51),
                io.Combo.Input("prompt_mode", options=["cycle", "uniform", "random"], default="cycle",
                               tooltip=("How prompt blocks map to windows. cycle: in order, the last repeating. "
                                        "uniform: the first block for every window. random: one block per window, "
                                        "drawn from the seed. With one block all three are the same.")),
                io.Combo.Input("window_mode", options=["uniform", "random"], default="uniform",
                               tooltip=("uniform: every window is window_frames (the last sized to reach the end). "
                                        "random: each window's length drawn from the seed among the lengths on both "
                                        "clocks at or below window_frames, so one prompt can be tested under "
                                        "uniform and non-uniform windows.")),
                # Appended from here on: saved graphs address inputs by position.
                io.Boolean.Input("save_metadata_png", default=True,
                                 tooltip=("Also write <prefix>_NNNNN.png, the first frame carrying the prompt and "
                                          "workflow, beside the video. The video carries both either way.")),
                io.Boolean.Input("keep_windows", default=True,
                                 tooltip=("Keep this run's window files (the videos and the latents resume reads) "
                                          "in <prefix>_windows/ after the join. Off removes them, so the next run "
                                          "renders every window.")),
                H3References.Input("references", optional=True,
                                   tooltip=("Reference stills from an Append Ref Image chain, presented with every "
                                            "window's prompt and encoded once per distinct prompt. The prompt then "
                                            "names them as <Picture N>.")),
                io.Boolean.Input("reuse_windows", default=True,
                                 tooltip=("Reuse the stored windows whose inputs have not changed, in order, and "
                                          "render from the first that has. Off renders every window: use it after "
                                          "replacing a model, LoRA or reference file under the same name.")),
                H3PromptLists.Input("lists", optional=True,
                                    tooltip=("Prompt List nodes filling __name__ placeholders in the prompt, one "
                                             "value per window that uses the name. A placeholder with no list here "
                                             "is read from the wildcards folder under the input directory.")),
            ],
            outputs=[
                io.String.Output(display_name="path"),
                io.String.Output(display_name="report"),
                io.Custom("VHS_FILENAMES").Output(display_name="Filenames"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, audio, sampler, sigmas, prompt, width, height,
                window_frames, context_frames, extent, seed, audio_mask, level,
                filename_prefix, crf, prompt_mode="cycle", window_mode="uniform",
                save_metadata_png=True, keep_windows=True, references=None,
                reuse_windows=True, lists=None) -> io.NodeOutput:
        import folder_paths
        # A DynamicCombo arrives as one nested dict (the selection under its own
        # id, the option's inputs beside it) or, from an API prompt that sets
        # only the selection, as a bare string.
        choice = extent if isinstance(extent, str) else extent["extent"]
        if choice not in ("whole", "first_seconds"):
            raise ValueError(f"unknown extent {choice!r}")
        max_seconds = (float(extent["seconds"]) if choice == "first_seconds" and not isinstance(extent, str)
                       else (30.0 if choice == "first_seconds" else None))
        waveform, rate, _ = _stereo(audio)
        seconds = waveform.shape[-1] / rate
        if max_seconds is not None:
            seconds = min(seconds, max_seconds)
        total_frames = int(math.ceil(seconds * FPS))
        rng = random.Random(int(seed))
        plan = plan_windows(total_frames, int(window_frames), int(context_frames),
                            rng=rng if window_mode == "random" else None)
        blocks = parse_prompt_blocks(prompt)
        if prompt_mode == "uniform":
            pick = [0] * len(plan)
        elif prompt_mode == "random":
            pick = [rng.randrange(len(blocks)) for _ in plan]
        else:
            pick = [min(i, len(blocks) - 1) for i in range(len(plan))]
        # a block's own `frames:` wins for its window; the plan's last window still must reach the end
        frames_per_window = [blocks[pick[i]][0] or n for i, n in enumerate(plan)]
        texts = [blocks[pick[i]][1] for i in range(len(plan))]
        # filled before anything is keyed or encoded: resume and the encode
        # cache see the text a window renders, never its placeholders
        texts, list_picks = resolve_texts(texts, lists_for(texts, lists, find_wildcard))
        n_windows = len(frames_per_window)

        out_dir = folder_paths.get_output_directory()
        full_out, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        work_dir = window_dir(full_out, filename)
        os.makedirs(work_dir, exist_ok=True)
        stem = f"{filename}_{counter:05d}"
        graph = getattr(cls.hidden, "prompt", None)
        extra = getattr(cls.hidden, "extra_pnginfo", None)

        # The stored windows that still hold: a prefix whose keys match. With no
        # queued prompt (a direct call) there is no root, nothing is reused and
        # nothing is stored for reuse.
        root = loop_resume.root_key(
            loop_resume.graph_signature(graph, getattr(cls.hidden, "unique_id", None),
                                        skip=loop_resume.SONG_PER_WINDOW),
            loop_resume.track_hash(waveform, rate))
        reused, keys, start = [], [], 0.0
        if reuse_windows and root is not None:
            for i, frames in enumerate(frames_per_window):
                key = loop_resume.window_key(root, i + 1, texts[i], frames, start, int(seed) + i,
                                             keys[-1] if keys else None)
                stored = loop_resume.read_window(work_dir, filename, i + 1)
                if stored is None or stored["key"] != key:
                    break
                reused.append(stored)
                keys.append(key)
                start = stored["next_start"]
        first = len(reused)

        # Every rendering window's conditioning before any window samples; see
        # the module docstring for why the key is the text (and, with
        # references, the frame count) and nothing else.
        track_latent, conds, cond_keys = None, {}, {}
        reports = [f"[{i + 1}] " + ", ".join(f"__{name}__ = {value!r}" for name, value in chosen.items())
                   for i, chosen in enumerate(list_picks) if chosen]
        if first < n_windows:
            enc = MiniMaxH3EncodeTrack.execute(audio_vae, audio, level)
            enc = getattr(enc, "args", enc)
            track_latent = enc[0]
            reports.append(enc[1])
            for i in range(first, n_windows):
                ck = (texts[i], int(frames_per_window[i]) if references is not None else None)
                cond_keys[i] = ck
                if ck in conds:
                    continue
                comfy.model_management.throw_exception_if_processing_interrupted()
                if references is None:
                    out = MiniMaxH3Conditioning.execute(clip, vae, texts[i], width, height,
                                                        frames_per_window[i], canvas="explicit")
                else:
                    out = MiniMaxH3ReferenceConditioning.execute(clip, references, texts[i], width, height,
                                                                frames_per_window[i], vae=vae, audio_vae=audio_vae)
                conds[ck] = getattr(out, "args", out)[0]
        reports.append((f"reused windows 1-{first} of {n_windows}" if first else "no stored window reused")
                       + (f"; {len(conds)} conditioning(s) encoded for {n_windows - first} rendered window(s)"
                          if first < n_windows else "; nothing rendered")
                       + (" with references" if references is not None and first < n_windows else ""))

        files = [s["video"] for s in reused]
        stored_latents = [s["latent"] for s in reused]
        prev = loop_resume.load_window_latent(reused[-1]["latent"]) if reused and first < n_windows else None
        for i in range(first, n_windows):
            frames = frames_per_window[i]
            comfy.model_management.throw_exception_if_processing_interrupted()
            latent, _count = _empty_av_latent(width, height, frames)
            # Always the real value: the window node freezes nothing when
            # `previous` is None and keeps the widget for what the NEXT window
            # takes. Passing 0 for the first window was the zero-as-mode this
            # session removed, and it raised on the first run after (2026-09-13).
            win = MiniMaxH3FreezeAudioWindow.execute(
                latent, audio_vae, audio, start, int(context_frames),
                previous=prev, audio_mask=audio_mask, level=level, track_latent=track_latent)
            win = getattr(win, "args", win)
            wlatent, _clip_audio, _span, trim, next_start, wreport, _new_audio = win
            reports.append(f"[{i + 1}] {wreport}")

            guider = Guider_Basic(model)
            guider.set_conds(conds[cond_keys[i]])
            latent_image = comfy.sample.fix_empty_latent_channels(model, wlatent["samples"])
            noise = comfy.sample.prepare_noise(latent_image, int(seed) + i)
            x0_output = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
            samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=wlatent.get("noise_mask"),
                                    callback=callback, disable_pbar=False, seed=int(seed) + i)
            samples = samples.to(comfy.model_management.intermediate_device())
            prev = {"samples": samples}

            # the video VAE takes the video stream; core's VAEDecode unbinds the pair the same way
            video_stream = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
            images = vae.decode(video_stream)
            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])
            images = images[int(trim):]
            video_path, latent_path = loop_resume.window_paths(work_dir, filename, i + 1)
            # the old latent goes first: a latent on disk must mean its video finished
            with contextlib.suppress(FileNotFoundError):
                os.remove(latent_path)
            written = _write_frames_mp4(video_path, images, crf)
            files.append(video_path)
            if root is not None:
                key = loop_resume.window_key(root, i + 1, texts[i], frames, start, int(seed) + i,
                                             keys[-1] if keys else None)
                keys.append(key)
                stored_latents.append(loop_resume.save_window(work_dir, filename, i + 1, key, samples,
                                                              trim, next_start))
            reports.append(f"[{i + 1}] wrote {written} frames to {os.path.basename(video_path)}")
            del images
            comfy.model_management.soft_empty_cache()
            start = float(next_start)

        # join, and mux the whole track cut to the video
        out_path = os.path.join(full_out, stem + ".mp4")
        join_and_mux(files, waveform, rate, out_path, work_dir, stem, prompt=graph, extra_pnginfo=extra)
        png_path = (write_metadata_png(os.path.join(full_out, stem + ".png"), out_path, graph, extra)
                    if save_metadata_png else None)
        if not keep_windows:
            for p in files + stored_latents:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(p)
            try:
                os.rmdir(work_dir)
            except OSError:
                pass  # windows from a longer earlier run are still there; they are that run's
        total = sum(frames_per_window) - int(context_frames) * (n_windows - 1)
        report = (f"{n_windows} windows {frames_per_window} with {context_frames}-frame context, "
                  f"prompt blocks {[p + 1 for p in pick]} ({prompt_mode}), windows {window_mode}, "
                  f"{total} frames ({total / FPS:.2f}s) over {seconds:.2f}s of track, "
                  f"{first} reused -> {out_path}\n" + "\n".join(reports))
        logger.info("[h3] MiniMaxH3AudioFreezeSong: %s", report.splitlines()[0])
        for line in reports:
            # which value each window took is the one thing a list run cannot be checked without
            if "__ = " in line:
                logger.info("[h3]   %s", line)
        filenames, preview = saved_outputs(out_path, subfolder, png_path)
        return io.NodeOutput(out_path, report, filenames, ui=preview)
