#!/usr/bin/env python3
"""Run the measurement reads on the 2026-09-26 follow-up batch as its arms land.

Reads `bench/results/2026-09-26_followup.jsonl` (`bench/followup_arms.json`'s
rows), finds each arm's clip and saved video latent on the output share by its
label (`run_graph_arms.py` appends `_<label>` to every filename prefix), and
runs the tool each prediction names
(`bench/results/2026-09-26_distill_run_predictions.md`):

- `looks`: `measure_clip_tone.py` and `measure_clip_temporal.py` on every
  `look_*` arm (O1's dark blocking, F7's pull toward a typical look).
- `ladder`: `latent_path_distance.py` per scene, with PDD8 (exact) as the
  reference and PDD4, PDD6 and merged PDD8 against it (P1, P5).
- `vsa`: `measure_clip_temporal.py`, `measure_block_period.py` and
  `measure_clip_tone.py` on FastH3 with VSA on and off (P7).

Scenes whose prompts ask for frame-to-frame brightness change are left out of
temporal reads, per the manifest's `analysis_notes`. An arm that has not
landed yet is skipped and listed. Each group writes
`bench/results/2026-09-26_followup_<group>_<tool>.json`.

    python bench/analyze_followup.py [--group looks|ladder|vsa ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from _paths import comfy_output  # noqa: E402

ROWS = REPO / "bench" / "results" / "2026-09-26_followup.jsonl"
OUT = REPO / "bench" / "results"
PY = sys.executable
#: from the manifest's analysis_notes: prompts that ask for brightness change
#: frame to frame, which the temporal measures would read as boiling or flicker.
TEMPORAL_EXCLUDE = ("kpop_dance_studio", "silent_film", "subway_chase_short")


def rows() -> dict:
    out = {}
    for line in ROWS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("warmup") and not r.get("error"):
                out[r["label"]] = r
    return out


def newest(pattern: str, root: Path):
    hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def clip(label: str, out_root: Path):
    return newest(f"*_{label}_0*[0-9].mp4", out_root / "Video")


def latent(label: str, out_root: Path):
    return newest(f"*_video_{label}_0*_.latent", out_root / "latents")


def run(tool: str, args: list, json_out: Path):
    cmd = [PY, str(HERE / tool), *map(str, args), "--json", str(json_out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-4000:])
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", choices=("looks", "ladder", "vsa"))
    args = ap.parse_args()
    groups = args.group or ["looks", "ladder", "vsa"]
    out_root = comfy_output()
    if not ROWS.exists():
        print(f"no rows yet: {ROWS.relative_to(REPO)} does not exist")
        return 0
    have = rows()
    missing = []

    if "looks" in groups:
        looks = sorted(l for l in have if l.startswith("look_"))
        clips = [c for c in (clip(l, out_root) for l in looks) if c]
        missing += [l for l in looks if not clip(l, out_root)]
        if clips:
            print("== looks: tone")
            run("measure_clip_tone.py", clips, OUT / "2026-09-26_followup_looks_tone.json")
            print("== looks: temporal")
            run("measure_clip_temporal.py", [*clips, "--stride", "4"],
                OUT / "2026-09-26_followup_looks_temporal.json")

    if "ladder" in groups:
        scenes = sorted({l.split("__")[0] for l in have if l.endswith(("__pdd4", "__pdd6", "__pdd8_merge"))})
        for scene in scenes:
            ref = latent(f"{scene}__pdd8", out_root)
            others = [latent(f"{scene}__{a}", out_root) for a in ("pdd4", "pdd6", "pdd8_merge")]
            others = [o for o in others if o]
            if not ref or not others:
                missing.append(f"{scene} ladder latents")
                continue
            print(f"== ladder: {scene}")
            run("latent_path_distance.py", [ref, *others], OUT / f"2026-09-26_followup_ladder_{scene}.json")

    if "vsa" in groups:
        scenes = sorted({l.split("__")[0] for l in have if l.endswith("__fasth3_novsa")})
        pairs = []
        for scene in scenes:
            on, off = clip(f"{scene}__fasth3", out_root), clip(f"{scene}__fasth3_novsa", out_root)
            if on and off and scene not in TEMPORAL_EXCLUDE:
                pairs += [on, off]
            else:
                missing.append(f"{scene} vsa pair")
        if pairs:
            print("== vsa: temporal")
            run("measure_clip_temporal.py", [*pairs, "--stride", "4"], OUT / "2026-09-26_followup_vsa_temporal.json")
            print("== vsa: block period")
            run("measure_block_period.py", pairs, OUT / "2026-09-26_followup_vsa_block.json")
            print("== vsa: tone")
            run("measure_clip_tone.py", pairs, OUT / "2026-09-26_followup_vsa_tone.json")

    if missing:
        print("not landed or not found:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
