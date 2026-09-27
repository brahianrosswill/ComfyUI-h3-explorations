#!/usr/bin/env python3
"""Does the DiT's prediction move when the encoder is int8_convrot instead of bf16?

## The question this owns

The int8_convrot encoder's conditioning is close to bf16 on text tokens and
carries a heavy tail on image tokens, including a patch int8 inflates well past
its bf16 norm; the conditioning-level record is
`bench/results/2026-09-27_encoder_int8_vs_bf16_conditioning.json`. Whether the
DiT responds to that tail is the part a conditioning comparison cannot say,
and a rendered clip cannot A/B it either (`CLAUDE.md`: the trajectory diverges
at frame zero). So this compares the DiT's denoised prediction at identical
noise, latent and sigma, with only the encoder changed -- the method
`bench/measure_marker_epsilon.py` uses for the marker rows, whose forward,
sigma shift and delta are imported here rather than retyped.

## Two phases, two processes

**capture** drives a running ComfyUI server through the shipped graphs, cut
at `MiniMaxH3Preflight`, and saves the conditioning and latent exactly as a
sampler would receive them (`bench/comfy_capture_nodes/h3_bench_capture`).
The encoder is loaded by `MiniMaxH3EncoderLoader`, the references and
keyframes are VAE-encoded by the graphs' own nodes, and only `encoder_name`
changes between arms. The server must be launched with the extra config this
subcommand writes; it prints the command.

**forward** loads those captures and the scene's own base DiT (the graph's
`UNETLoader` file, no LoRA, no Sol, no sage: ComfyUI's default attention, for
the reason `measure_marker_epsilon.py` gives), and runs one forward per arm
per probe step.

## The arms, and what each comparison is for

Every comparison is against `bf16`, the reference.

    null        bf16 vs bf16, the same capture run twice    MUST be exactly 0.0
    treatment   bf16 vs int8                                the question
    attribution bf16 vs bf16 with int8's IMAGE tokens only  how much is the image tail
    control     bf16 vs bf16 + random noise, per-token norms matched to int8's error
    scale       bf16 vs bf16 of the prompt with " ." appended

**The control is the null distribution for the treatment.** It perturbs every
token by exactly as much as int8 does, in a random direction. If the treatment
reads like the control, int8 costs what any error of that size costs and its
particular directions do not matter; if it reads well above, they do.

**The scale row is not a bound.** It is one semantically empty text edit, to
put the treatment beside something a user would call "no change". It moves
the token count, so it is a scale, not a controlled pair.

## What it does not establish

Whether the difference is visible. A per-step prediction delta at one point on
the trajectory can wash out or compound by the final frame; perceptual claims
need `docs/eval_comparison.md`.

    python bench/measure_encoder_quant_dit.py capture --source-video <render.mp4>
    python bench/measure_encoder_quant_dit.py forward --out bench/results/<date>_encoder_quant_dit.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
CAPTURE_NODES = BENCH / "comfy_capture_nodes"
URL = "http://127.0.0.1:8188"

#: The two encoder files compared. int8 is `h3_config.MODELS["clip"]`; bf16 is
#: the release truncated by `bench/convert_h3_bf16_encoder.py`.
ENCODERS = {"int8": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "bf16": "qwen3vl_32b_minimax_h3_bf16_pruned.safetensors"}

#: Frames of the source render used as images, by frame index. Chosen by eye
#: from a PDD8 render of `t2va_look_anchor` on 2026-09-27: frame 0 is the empty
#: workshop, frame 120 the close shot of the violin maker in the beam.
FRAMES = {"violin_f000.png": 0, "violin_f120.png": 120}

#: Scenes: the shipped graph whose conditioning subgraph is used, the bank
#: prompt substituted into it (None keeps the graph's own), and the images
#: substituted into its LoadImage nodes in node-id order (None keeps them).
#: ref2va_dialogue is the shipped dialogue graph unchanged: two faces, its own
#: prompt, the heaviest image tail measured at the conditioning level.
SCENES = {
    "violin_t2v": ("h3_text_to_video_pdd_api.json", "t2va_look_anchor", []),
    "violin_i2v": ("h3_first_frame_to_video_api.json", "i2va_look_anchor",
                   ["violin_f000.png"]),
    "violin_ref2va": ("h3_image_ref_plus_text_to_video_pdd_api.json", "ref2va_look_anchor",
                      ["violin_f120.png", "violin_f000.png"]),
    "ref2va_dialogue": ("h3_image_ref_plus_text_to_video_dialogue_api.json", None, None),
}


# --------------------------------------------------------------------- capture

def _post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    return json.loads(data) if data else None


def _get(path):
    with urllib.request.urlopen(URL + path) as r:
        return json.loads(r.read())


def _upload_temp(path: Path) -> None:
    """Stage an image in the server's temp folder through its own API."""
    boundary = "h3bench" + hashlib.sha1(path.name.encode()).hexdigest()
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ntemp\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
         f"filename=\"{path.name}\"\r\nContent-Type: image/png\r\n\r\n").encode(),
        path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()])
    req = urllib.request.Request(URL + "/upload/image", body,
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    urllib.request.urlopen(req).read()


def _cut(graph: dict, root: str) -> dict:
    keep, stack = set(), [root]
    while stack:
        k = stack.pop()
        if k in keep:
            continue
        keep.add(k)
        for v in graph[k]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return {k: copy.deepcopy(graph[k]) for k in keep}


def _preflight(graph: dict) -> str:
    ids = [k for k, v in graph.items() if v["class_type"] == "MiniMaxH3Preflight"]
    if len(ids) != 1:
        raise SystemExit(f"expected one MiniMaxH3Preflight, found {ids}")
    return ids[0]


def scene_graph(name: str) -> dict:
    """The scene's shipped graph, cut to its Preflight, with substitutions."""
    gfile, bank_id, images = SCENES[name]
    g = json.loads((REPO / "workflows" / gfile).read_text())
    pre = _preflight(g)
    cond = g[pre]["inputs"]["conditioning"][0]
    sub = _cut(g, pre)
    if bank_id is not None:
        text = (REPO / "prompt_bank" / f"{bank_id}.txt").read_text()
        sub[cond]["inputs"]["prompt"] = text
    loads = sorted((k for k, v in sub.items() if v["class_type"] == "LoadImage"), key=int)
    if images is not None:
        if len(loads) != len(images):
            raise SystemExit(f"{name}: graph has {len(loads)} LoadImage, scene names {len(images)}")
        for k, img in zip(loads, images):
            sub[k]["inputs"]["image"] = f"{img} [temp]"
    return sub


def _run(graph: dict, label: str) -> None:
    posted = _post("/prompt", {"prompt": graph})
    if not posted:
        raise SystemExit(f"{label}: the server returned no prompt id")
    pid = posted["prompt_id"]
    while True:
        h = _get(f"/history/{pid}")
        st = h.get(pid, {}).get("status", {})
        if st.get("completed") or st.get("status_str") == "error":
            break
        time.sleep(0.5)
    if st.get("status_str") != "success":
        raise SystemExit(f"{label}: {json.dumps(st)[:1500]}")


def kitchen_build() -> str:
    """The comfy_kitchen build this process would load, read from the install."""
    from importlib.metadata import version
    return version("comfy_kitchen")


def capture(args) -> int:
    out = Path(args.capture_dir)
    try:
        info = _get("/object_info/H3BenchSaveConditioning")
    except Exception:
        info = {}
    if "H3BenchSaveConditioning" not in info:
        cfg = Path(tempfile.gettempdir()) / "h3_bench_capture_nodes.yaml"
        cfg.write_text(f"h3_bench:\n  custom_nodes: {CAPTURE_NODES}\n")
        print("The server does not have the capture node. Launch it with:\n"
              f"  H3_BENCH_CAPTURE_DIR={out} ./start.sh default "
              f"--extra-model-paths-config {cfg}")
        return 2
    frames = {}
    if any(SCENES[s][2] for s in args.scenes):
        if not args.source_video:
            raise SystemExit("--source-video is required for scenes with images")
        with tempfile.TemporaryDirectory() as td:
            for fname, idx in FRAMES.items():
                p = Path(td) / fname
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.source_video,
                                "-vf", f"select='eq(n\\,{idx})'", "-frames:v", "1", str(p)],
                               check=True)
                frames[fname] = {"frame": idx,
                                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                _upload_temp(p)
    manifest = {"kitchen_build_at_capture": kitchen_build(),
                "source_video": Path(args.source_video).name if args.source_video else None,
                "frames": frames, "encoders": ENCODERS, "captures": []}
    for scene in args.scenes:
        base = scene_graph(scene)
        pre = _preflight(base)
        cond = base[pre]["inputs"]["conditioning"][0]
        enc = [k for k, v in base.items() if v["class_type"] == "MiniMaxH3EncoderLoader"]
        if len(enc) != 1:
            raise SystemExit(f"{scene}: expected one MiniMaxH3EncoderLoader, found {enc}")
        arms = [("int8", ENCODERS["int8"], ""), ("bf16", ENCODERS["bf16"], ""),
                ("bf16_edit", ENCODERS["bf16"], " .")]
        for arm, encoder, suffix in arms:
            _post("/free", {"unload_models": True, "free_memory": True})
            g = copy.deepcopy(base)
            g[enc[0]]["inputs"]["encoder_name"] = encoder
            g[cond]["inputs"]["prompt"] += suffix
            label = f"{scene}__{arm}"
            g["h3_bench_save"] = {"class_type": "H3BenchSaveConditioning",
                                  "inputs": {"conditioning": [pre, 0], "samples": [pre, 1],
                                             "name": label}}
            t0 = time.time()
            _run(g, label)
            print(f"[capture] {label} {time.time() - t0:.1f}s", flush=True)
            manifest["captures"].append({"scene": scene, "arm": arm, "encoder": encoder,
                                         "prompt_suffix": suffix})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


# --------------------------------------------------------------------- forward

def forward(args) -> int:
    sys.path.insert(0, str(BENCH))
    import measure_marker_epsilon as E  # sets sys.path and imports comfy in order
    import torch
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    from h3_config import SAMPLING

    cap = Path(args.capture_dir)
    manifest = json.loads((cap / "manifest.json").read_text())
    probe_steps = [int(s) for s in args.probe_steps.split(",")]

    def load(scene, arm):
        # Captures this tool's own node wrote; the conditioning extras carry
        # reference dicts that a weights-only load is not guaranteed to accept.
        return torch.load(cap / f"{scene}__{arm}.pt", weights_only=False)

    def extras_equal(a, b) -> bool:
        def eq(x, y):
            if isinstance(x, torch.Tensor):
                return isinstance(y, torch.Tensor) and x.shape == y.shape and torch.equal(x, y)
            if isinstance(x, dict):
                return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
            if isinstance(x, (list, tuple)):
                return len(x) == len(y) and all(eq(u, v) for u, v in zip(x, y))
            # comfy.nested_tensor.NestedTensor (the AV latent) and similar
            # holders are not torch.Tensor and have no value equality: `==`
            # would compare identity and fail two identical captures.
            if hasattr(x, "__dict__") and not isinstance(x, type):
                return type(x) is type(y) and eq(vars(x), vars(y))
            return x == y
        return eq(a, b)

    def with_tensor(conditioning, tensor):
        c = [[tensor, dict(conditioning[0][1])]]
        return c

    records = []
    by_unet: dict[str, list[str]] = {}
    for scene in args.scenes:
        g = json.loads((REPO / "workflows" / SCENES[scene][0]).read_text())
        unet = [v["inputs"]["unet_name"] for v in g.values() if v["class_type"] == "UNETLoader"]
        by_unet.setdefault(unet[0], []).append(scene)

    for unet, scenes in by_unet.items():
        print(f"[forward] loading DiT {unet}", flush=True)
        model = E.sigma_shifted(comfy.sd.load_diffusion_model(E._resolve("diffusion_models", unet)))
        for scene in scenes:
            i8, bf, ed = load(scene, "int8"), load(scene, "bf16"), load(scene, "bf16_edit")
            ci8, cbf = i8["conditioning"], bf["conditioning"]
            x8, xb = ci8[0][0], cbf[0][0]
            if x8.shape != xb.shape:
                raise SystemExit(f"{scene}: encoder outputs differ in shape")
            extras = {k: v for k, v in ci8[0][1].items()}
            if not extras_equal(extras, cbf[0][1]):
                raise SystemExit(f"{scene}: non-encoder conditioning differs between arms; "
                                 "the treatment would not be the encoder alone")
            if not extras_equal(i8["samples"], bf["samples"]):
                raise SystemExit(f"{scene}: latents differ between arms")
            tags = torch.as_tensor(cbf[0][1]["minimax_token_tags"]).flatten()
            vision = (tags == 0)
            err = (x8 - xb).float()
            # attribution: bf16 with int8's image-token rows only
            spliced = xb.clone()
            spliced[:, vision] = x8[:, vision]
            # control: per-token norms of int8's error, random directions
            gen = torch.Generator().manual_seed(args.seed)
            noise_dir = torch.randn(err.shape, generator=gen)
            noise_dir = noise_dir / noise_dir.norm(dim=-1, keepdim=True)
            ctrl = (xb.float() + noise_dir * err.norm(dim=-1, keepdim=True)).to(xb.dtype)
            arms = {"bf16": cbf, "int8": ci8,
                    "int8_image_only": with_tensor(cbf, spliced),
                    "control_matched_noise": with_tensor(cbf, ctrl),
                    "bf16_edit": ed["conditioning"]}
            latent = bf["samples"]["samples"]
            noise = comfy.sample.prepare_noise(latent, args.seed)
            sigmas = comfy.samplers.calculate_sigmas(
                model.get_model_object("model_sampling"), SAMPLING["scheduler"], SAMPLING["steps"])
            comparisons = [("null", "bf16", "bf16"), ("treatment", "bf16", "int8"),
                           ("attribution", "bf16", "int8_image_only"),
                           ("control", "bf16", "control_matched_noise"),
                           ("scale", "bf16", "bf16_edit")]
            for step in probe_steps:
                preds = {}
                for arm, cond in arms.items():
                    t0 = time.time()
                    preds[arm] = E.one_forward(model, cond, noise, latent, sigmas, step, args.seed)
                    print(f"[forward] {scene} step {step} {arm}: {time.time() - t0:.1f}s", flush=True)
                t0 = time.time()
                preds["bf16_again"] = E.one_forward(model, cbf, noise, latent, sigmas, step, args.seed)
                print(f"[forward] {scene} step {step} bf16_again: {time.time() - t0:.1f}s", flush=True)
                for kind, left, right in comparisons:
                    right_key = "bf16_again" if kind == "null" else right
                    records.append({"scene": scene, "step": step, "sigma": float(sigmas[step]),
                                    "kind": kind, "left": left, "right": right,
                                    "delta": E.delta(preds[left], preds[right_key])})
                    d = records[-1]["delta"]
                    print(f"  {kind:<11} {right:<22} video {d['video']['relative_l2']:.6f} "
                          f"audio {d['audio']['relative_l2']:.6f}", flush=True)
            records.append({"scene": scene, "kind": "conditioning",
                            "vision_tokens": int(vision.sum()), "text_tokens": int((~vision).sum()),
                            "int8_rel_l2": float(err.norm() / xb.float().norm()),
                            "int8_image_rel_l2": float(err[:, vision].norm() / xb[:, vision].float().norm())
                            if vision.any() else None})
        del model
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

    record = {
        "question": "does the DiT's denoised prediction move when the encoder is int8_convrot "
                    "instead of bf16, and is that more than an equal-sized random error",
        "measurement": "DiT denoised-prediction relative L2 per latent block at fixed noise, "
                       "latent and sigma; ComfyUI default attention, base DiT, no LoRA",
        "producer": "bench/measure_encoder_quant_dit.py",
        "capture_manifest": manifest,
        "schedule": {"scheduler": SAMPLING["scheduler"], "steps": SAMPLING["steps"],
                     "probe_steps": probe_steps},
        "seed": args.seed,
        "kitchen_build_at_forward": kitchen_build(),
        "does_not_establish": [
            "visibility: a per-step prediction delta can wash out or compound by the final frame",
            "the scale row is one text edit, not a bound; do not quote the treatment as a fraction of it",
        ],
        "results": records,
    }
    out = Path(args.out)
    out.write_text(json.dumps(record, indent=1) + "\n")
    bad = [r for r in records if r.get("kind") == "null"
           and r["delta"]["video"]["relative_l2"] != 0.0]
    if bad:
        print("NULL IS NOT ZERO: the forward is not deterministic; nothing above is attributable.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "forward"):
        p = sub.add_parser(name)
        p.add_argument("--capture-dir", required=True,
                       help="where captures go; outside the repo (they are GBs of tensors)")
        p.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    sub.choices["capture"].add_argument("--source-video", default=None,
                                        help="the look_anchor render the violin frames are cut from")
    f = sub.choices["forward"]
    f.add_argument("--probe-steps", default="0,8,15")
    f.add_argument("--seed", type=int, default=730451892,
                   help="default is the shipped graphs' own noise seed")
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    return capture(args) if args.cmd == "capture" else forward(args)


if __name__ == "__main__":
    raise SystemExit(main())
