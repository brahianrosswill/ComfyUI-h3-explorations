#!/usr/bin/env python3
"""Which measured properties separate the distills consistently across scenes?

Written 2026-09-26 for the owner's "video and audio are both equally
interesting if theres a lead to follow and a so what". Reads per-clip records
from `measure_clip_tone.py`, `measure_clip_resolution.py`,
`measure_clip_temporal.py` and `compare_audio_pairs.py` over the same scenes,
maps each clip to (scene, arm) by `run_graph_arms.py`'s `_<scene>__<arm>_`
naming, and for every numeric metric reports each arm's median across scenes
and on how many scenes it is the highest and the lowest of the arms.

One clip per arm per scene: a count like "highest on 12 of 13" is how many
paired draws share a sign, not an effect size. A metric is flagged when one
arm is extreme on at least `--flag` of the scenes that have every arm.

    python bench/distill_signatures.py --arm pdd8 --arm flashgen --arm fasth3 --scene S ... \\
        --tone T.json --resolution R.json [--temporal X.json] [--audio A.json] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LABEL = re.compile(r"_([a-z0-9_]+)__([a-z0-9_]+?)_0\d+")


def clip_rows(path: Path, arms: set, scenes: list) -> dict:
    """{(scene, arm): {metric: value}} from a per-clip record."""
    out = {}
    rows = json.loads(path.read_text())
    rows = rows.get("rows") or rows.get("clips") or rows
    if isinstance(rows, dict):                      # measure_clip_temporal: {clip: {"median": {...}}}
        rows = [{"clip": c, **v.get("median", {})} for c, v in rows.items()]
    for r in rows:
        m = LABEL.search(r.get("clip", ""))
        if not m or m.group(2) not in arms or r.get("part", "all") != "all":
            continue
        scene = next((sc for sc in scenes if m.group(1).endswith("_" + sc) or m.group(1) == sc), None)
        if scene is None:
            continue
        out[(scene, m.group(2))] = {k: v for k, v in r.items()
                                         if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return out


def audio_rows(path: Path, arms: set, scenes: list) -> dict:
    out = {}
    for scene, per in json.loads(path.read_text())["scenes"].items():
        for arm, m in per.items():
            if arm in arms:
                out[(scene, arm)] = {f"audio_{k}": v for k, v in m.items() if isinstance(v, (int, float))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--scene", action="append", required=True)
    ap.add_argument("--tone", type=Path)
    ap.add_argument("--resolution", type=Path)
    ap.add_argument("--temporal", type=Path)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--flag", type=float, default=0.8, help="share of scenes for a flag")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    arms = set(args.arm)

    data = defaultdict(dict)
    for src, reader in ((args.tone, clip_rows), (args.resolution, clip_rows),
                        (args.temporal, clip_rows), (args.audio, audio_rows)):
        if src and src.exists():
            for key, m in reader(src, arms, args.scene).items():
                prefix = "" if reader is audio_rows else src.stem.rsplit("_", 1)[-1] + "_"
                data[key].update({prefix + k: v for k, v in m.items()})

    scenes = sorted({s for s, _ in data if all((s, a) in data for a in args.arm)})
    metrics = sorted(set().union(*[set(data[(s, a)]) for s in scenes for a in args.arm])) if scenes else []
    report = {}
    flagged = []
    for met in metrics:
        hi = defaultdict(int)
        lo = defaultdict(int)
        have = [s for s in scenes if all(met in data[(s, a)] for a in args.arm)]
        if not have:
            continue
        for s in have:
            vals = {a: data[(s, a)][met] for a in args.arm}
            if len(set(vals.values())) == 1:
                continue
            hi[max(vals, key=lambda a: vals[a])] += 1
            lo[min(vals, key=lambda a: vals[a])] += 1
        med = {a: statistics.median(data[(s, a)][met] for s in have) for a in args.arm}
        report[met] = {"median": med, "highest": dict(hi), "lowest": dict(lo), "scenes": len(have)}
        for a in args.arm:
            for side, cnt in (("highest", hi[a]), ("lowest", lo[a])):
                if cnt >= args.flag * len(have):
                    flagged.append((met, a, side, cnt, len(have)))
    print(f"{len(scenes)} scenes with every arm; {len(metrics)} metrics")
    for met, a, side, cnt, n in flagged:
        m = report[met]["median"]
        print(f"  {met:<28} {a:<9} {side:<8} on {cnt}/{n}   medians " +
              "  ".join(f"{k} {v:.4g}" for k, v in m.items()))
    if args.json:
        args.json.write_text(json.dumps({"measured_by": "bench/distill_signatures.py", "arms": args.arm,
                                         "scenes": scenes, "flag_share": args.flag,
                                         "flagged": [dict(zip(("metric", "arm", "side", "count", "of"), f)) for f in flagged],
                                         "metrics": report}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
