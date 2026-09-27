#!/usr/bin/env python3
"""FlashGen's LoRA as weights: its effective rank, and where in t it moves modulation.

Written 2026-09-26 for the owner's request, relayed by the VAE session, to
look at FlashGen the way `bench/checkpoint_delta_map.py` and
`bench/compare_adaln_modulation.py` look at FastH3's full weights. The
predictions came first: `bench/results/2026-09-26_flashgen_weights_predictions.md`.
CPU only.

- **Rank** (FP1): each module's delta B @ A, from the publisher's source file,
  through the singular values of R_B R_A^T (A^T = Q_A R_A, B = Q_B R_B), which
  equal those of B @ A. The report gives the singular values holding 95% and 99%
  of the Frobenius energy, by module kind and block band, and whether the
  rank-64 conversion's non-adaln modules equal the source's.
- **Modulation in t** (FP3): each block's modulation on the pruned base,
  `table @ W.T + b` as `compare_adaln_modulation.modulation` evaluates it, and
  FlashGen's change to it, `scale * table @ A.T @ B.T + diff_b` from the
  rank-64 conversion. The change is relative to the base, per table row, and
  is reported at FlashGen's four training sigmas through each stream's shift,
  over the band it trained in, and below it.

    CUDA_VISIBLE_DEVICES= python bench/analyze_flashgen_weights.py SOURCE.safetensors \\
        CONVERTED.safetensors BASE_PRUNED.safetensors [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "workflows"))
import h3_config  # noqa: E402
from compare_adaln_modulation import modulation, shifted  # noqa: E402

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear")
#: FlashGen's trained positions, shifted per stream. **Inherited**: the
#: publisher's `base_schedule`, as the converted file's metadata records it.
POSITIONS = (1.0, 0.7, 0.4, 0.15)


def sv(a, b):
    """Singular values of b @ a without forming it: a [r, in], b [out, r]."""
    _, ra = torch.linalg.qr(a.T.to(torch.float64))
    _, rb = torch.linalg.qr(b.to(torch.float64))
    return torch.linalg.svdvals(rb @ ra.T)


def k_for(s, frac):
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((e < frac).sum().item()) + 1


def find(f, suffix):
    hits = [k for k in f.keys() if k.endswith(suffix)]
    if len(hits) != 1:
        raise SystemExit(f"{suffix}: {len(hits)} matches")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("converted")
    ap.add_argument("base")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {"source": Path(args.source).name, "converted": Path(args.converted).name,
           "base": Path(args.base).name}

    # --- FP1: rank ---------------------------------------------------------
    rank_rows = []
    with safe_open(args.source, "pt") as src, safe_open(args.converted, "pt") as cv:
        mods = sorted({k.rsplit(".lora_", 1)[0] for k in src.keys() if ".lora_A." in k})
        for m in mods:
            a = src.get_tensor(f"{m}.lora_A.default.weight")
            b = src.get_tensor(f"{m}.lora_B.default.weight")
            s = sv(a, b)
            row = {"module": m, "k95": k_for(s, 0.95), "k99": k_for(s, 0.99),
                   "rank": int(a.shape[0])}
            ck = f"diffusion_model.{m}"
            if "adaln" not in m and f"{ck}.lora_A.weight" in cv.keys():
                ca = cv.get_tensor(f"{ck}.lora_A.weight").to(torch.float64)
                cb = cv.get_tensor(f"{ck}.lora_B.weight").to(torch.float64)
                al = cv.get_tensor(f"{ck}.alpha").item() if f"{ck}.alpha" in cv.keys() else ca.shape[0]
                # Singular values, not elements: the conversion permutes qkv's
                # output rows from the release's per-head interleave to ComfyUI's
                # q/k/v bands (`convert_flashgen_lora.py`), which leaves them equal.
                cs = sv(ca, cb * (al / ca.shape[0]))
                row["conv_rel_err"] = float((cs - s).norm() / s.norm())
            rank_rows.append(row)

    def summarize(rows, key):
        vals = [r[key] for r in rows]
        return {"mean": round(sum(vals) / len(vals), 1), "min": min(vals), "max": max(vals)} if vals else None

    by_kind = {}
    for kind in KINDS:
        rows = [r for r in rank_rows if r["module"].startswith("blocks.") and r["module"].endswith(kind)]
        by_kind[kind] = {"k95": summarize(rows, "k95"), "k99": summarize(rows, "k99")}
    bands = {}
    for name, lo, hi in (("blocks 0-16", 0, 16), ("blocks 17-33", 17, 33), ("blocks 34-49", 34, 49)):
        rows = [r for r in rank_rows if r["module"].startswith("blocks.")
                and lo <= int(r["module"].split(".")[1]) <= hi]
        bands[name] = {"k95": summarize(rows, "k95"), "k99": summarize(rows, "k99")}
    conv = [r["conv_rel_err"] for r in rank_rows if "conv_rel_err" in r]
    out["rank"] = {"modules": len(rank_rows), "all": {"k95": summarize(rank_rows, "k95"),
                                                       "k99": summarize(rank_rows, "k99")},
                   "by_kind": by_kind, "by_band": bands,
                   "conversion_non_adaln": {"modules": len(conv), "max_rel_err": max(conv) if conv else None}}

    # --- FP3: modulation in t -----------------------------------------------
    rungs = {"video": [shifted(u, 12.0) for u in POSITIONS], "audio": [shifted(u, 3.0) for u in POSITIONS]}
    per_block = []
    with safe_open(args.base, "pt") as bf, safe_open(args.converted, "pt") as cv:
        table = bf.get_tensor(find(bf, "adaln_t_table")).to(torch.float64)      # [1025, 8]
        n = table.shape[0]
        tgrid = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        prefixes = [f"blocks.{i}.adaln_proj.linear" for i in range(50)] + ["final_layer.adaln_proj.linear"]
        for p in prefixes:
            bp = find(bf, p + ".weight")[: -len(".weight")]
            base = modulation(bf, bp, table)                                    # [T, out]
            ck = f"diffusion_model.{p}"
            a = cv.get_tensor(f"{ck}.lora_A.weight").to(torch.float64)
            b = cv.get_tensor(f"{ck}.lora_B.weight").to(torch.float64)
            al = cv.get_tensor(f"{ck}.alpha").item() if f"{ck}.alpha" in cv.keys() else a.shape[0]
            db = cv.get_tensor(f"{ck}.diff_b").to(torch.float64) if f"{ck}.diff_b" in cv.keys() else 0.0
            delta = (al / a.shape[0]) * ((table @ a.T) @ b.T) + db
            rel = (delta.norm(dim=1) / base.norm(dim=1))                        # [T]

            def at(ts):
                return [round(float(rel[int(round(t * (n - 1)))]), 4) for t in ts]

            def band(lo, hi):
                m = (tgrid >= lo) & (tgrid <= hi)
                return round(float(rel[m].mean()), 4)
            per_block.append({"module": p, "at_video_rungs": at(rungs["video"]),
                              "at_audio_rungs": at(rungs["audio"]),
                              "mean_trained_band": band(min(rungs["video"]), 1.0),
                              "mean_below_0_6": band(0.0, 0.6),
                              "max_over_t": round(float(rel.max()), 4),
                              "argmax_t": round(float(tgrid[int(rel.argmax())]), 4)})
    out["modulation"] = {"rungs": rungs, "per_block": per_block}

    blocks = [r for r in per_block if r["module"].startswith("blocks.")]
    print(f"rank: {len(rank_rows)} modules, k95 {out['rank']['all']['k95']}, k99 {out['rank']['all']['k99']}")
    for kind, v in by_kind.items():
        print(f"  {kind:18s} k95 {v['k95']}  k99 {v['k99']}")
    for name, v in bands.items():
        print(f"  {name:12s} k95 {v['k95']}")
    print(f"  conversion, non-adaln: {out['rank']['conversion_non_adaln']}")
    print("modulation change relative to the base, blocks 0, 24, 49, final:")
    for r in [blocks[0], blocks[24], blocks[49], per_block[-1]]:
        print(f"  {r['module']:28s} video rungs {r['at_video_rungs']} band {r['mean_trained_band']} "
              f"below0.6 {r['mean_below_0_6']} max {r['max_over_t']} at t={r['argmax_t']}")
    mb = sum(r["mean_trained_band"] for r in blocks) / len(blocks)
    ml = sum(r["mean_below_0_6"] for r in blocks) / len(blocks)
    print(f"  all 50 blocks: mean in trained band {mb:.4f}, below 0.6 {ml:.4f}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
