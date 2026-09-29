#!/usr/bin/env python3
"""Write FastH3-contract graphs whose UNETLoader is MiniMaxH3OverlayLoader.

Written 2026-09-29 to check the exact overlay (`checkpoint_overlay.py`) through
a real render. The source is the shipped contract `_savelat` graph; node "1"
becomes the loader, which outputs MODEL at slot 0 as UNETLoader does, so every
link to it stays. These graphs are scratch, written to `--out-dir` and never
into `workflows/` (CLAUDE.md: `workflows/*.json` come from
`build_workflows.py`); the point is an arm that `bench/run_graph_arms.py` can
render and a `torch.equal` on the final latent against the hybrid-file arm of
`bench/fasth3_gates_arms.json`.

Arms (`overlay_fasth3_v2_on_fl2va`, `blocks` as the loader takes it):

- `overlay_gates`: no backbone, gates only. Should equal the
  `fl2va_gates` hybrid file's render.
- `overlay_all`: every piece. Should equal the FastH3 V2 file's render.
- `overlay_no_gates`: every piece but the gates. Should equal the
  `fasth3_nogates` hybrid file's render.
- `overlay_all_g075`, `overlay_all_g050`, `overlay_all_g000` (#36, the gate dial,
  `bench/fasth3_gate_dial_arms.json`): every piece with the gate row scales
  times 0.75, 0.5 and 0.
- `overlay_late_30_49` and `overlay_early_0_29` (#48, `bench/fasth3_late_blocks_arms.json`):
  the gates plus FastH3's backbone diff on blocks 30-49, and on blocks 0-29 as the
  location control; refiner, adaln and io stay fl2va's. Looks, not equality checks.
- `overlay_no_refiner`: every piece but the refiner, so FastH3's backbone and
  gates run on fl2va's own bf16 token refiner. Not an equality check: a look.

    python bench/make_overlay_loader_graphs.py --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "workflows/distill_experiments/h3_probe_t2v_fasth3_8step_contract_savelat_api.json"
OVERLAY = "fasth3_v2_on_fl2va.h3overlay.safetensors"

ARMS = {
    "overlay_gates": dict(blocks="", gates=True, refiner=False, adaln=False, io_layers=False),
    "overlay_all": dict(blocks="all", gates=True, refiner=True, adaln=True, io_layers=True),
    "overlay_no_gates": dict(blocks="all", gates=False, refiner=True, adaln=True, io_layers=True),
    "overlay_no_refiner": dict(blocks="all", gates=True, refiner=False, adaln=True, io_layers=True),
    # #36, the gate dial: every piece, the gate row scales multiplied by alpha.
    "overlay_all_g075": dict(blocks="all", gates=True, refiner=True, adaln=True, io_layers=True, gate_scale=0.75),
    "overlay_all_g050": dict(blocks="all", gates=True, refiner=True, adaln=True, io_layers=True, gate_scale=0.5),
    "overlay_all_g000": dict(blocks="all", gates=True, refiner=True, adaln=True, io_layers=True, gate_scale=0.0),
    # #48, FastH3's backbone diff by depth on top of the gates alone. Gates are their own
    # piece, so `blocks` only selects backbone diffs. Late is where FastH3 and FlashGen
    # agree (2026-09-26_fasth3_weights.md finding 5); early is the control for location.
    "overlay_late_30_49": dict(blocks="30-49", gates=True, refiner=False, adaln=False, io_layers=False),
    "overlay_early_0_29": dict(blocks="0-29", gates=True, refiner=False, adaln=False, io_layers=False),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(SRC.read_text())
    assert base["1"]["class_type"] == "UNETLoader", "the contract graph's node 1 is no longer the UNETLoader"
    for name, sel in ARMS.items():
        g = json.loads(json.dumps(base))
        g["1"] = {"class_type": "MiniMaxH3OverlayLoader",
                  "inputs": {"overlay_name": OVERLAY, "gate_scale": 1.0, **sel}}
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(g))
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
