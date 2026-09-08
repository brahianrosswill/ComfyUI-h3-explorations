#!/usr/bin/env python3
"""Does the `token_aug` nondeterminism belong to upstream's kernel or to ours?

Our fork's delta against the upstream tag is Python only -- `git diff --stat
v0.2.33..HEAD -- '*.cu' '*.cuh' '*.hip' '*.h'` is empty and the selection kernel
`sol_attn_token.cu` is byte-identical -- so the diff already says we did not
write this. That is an argument, not a measurement, and the first thing anyone
upstream will ask is whether it happens without our code at all.

So this runs the standalone repro under each wheel in its own process and
reports both. Point `--stock-path` at a directory holding a plain
`comfy-kitchen==<upstream version>` install (`uv pip install --target <dir>`);
it is put on PYTHONPATH ahead of site-packages so it shadows the installed
build, and torch still comes from the environment running this.

Each arm carries a POSITIVE CONTROL that it is the wheel it claims to be:
`blk_cnt` is our fork's added parameter and appears in no upstream release, so
its absence identifies stock and its presence identifies ours. Without that,
"stock reproduces" could just be our build running twice under two labels,
which is the failure this file exists to rule out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROBE = """
import inspect, importlib.metadata, comfy_kitchen as ck, json
print(json.dumps({
    "version": importlib.metadata.version("comfy-kitchen"),
    # deliberately not ck.__file__: the stock arm loads from a path outside
    # this repo, and the record is tracked. has_blk_cnt is the identifier.
    "has_blk_cnt": "blk_cnt" in inspect.signature(ck.sol_attn).parameters,
    "has_token_aug": "token_aug" in inspect.signature(ck.sol_attn).parameters,
}))
"""


def identify(env):
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True,
                         text=True, env=env)
    if out.returncode:
        return {"error": out.stderr.strip()[-400:]}
    return json.loads(out.stdout)


def run_repro(slice_path, launches, env):
    proc = subprocess.run(
        [sys.executable, str(_HERE / "repro_token_aug_nondeterminism.py"),
         "--slice", slice_path, "--launches", str(launches)],
        capture_output=True, text=True, env=env)
    rows = 0
    for line in proc.stdout.splitlines():
        if "token_aug=" in line and "control" not in line and "rows moved" in line:
            rows = int(line.split("rows moved")[1].split("/")[0])
    return {"reproduces": proc.returncode == 1, "rows_moved": rows,
            "stdout_tail": proc.stdout.strip().splitlines()[-1:] or [""]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--stock-path", required=True,
                    help="directory holding a plain upstream comfy-kitchen install")
    ap.add_argument("--launches", type=int, default=60)
    ap.add_argument("--runs", type=int, default=3, help="cold runs per wheel")
    ap.add_argument("--out")
    args = ap.parse_args()

    base = dict(os.environ)
    stock = dict(base)
    stock["PYTHONPATH"] = args.stock_path + os.pathsep + base.get("PYTHONPATH", "")

    arms = {}
    for label, env in (("installed_build", base), ("stock_upstream_wheel", stock)):
        ident = identify(env)
        runs = [run_repro(args.slice, args.launches, env) for _ in range(args.runs)]
        arms[label] = {
            "identity": ident,
            "is_our_build": ident.get("has_blk_cnt"),
            "runs": runs,
            "reproduced_in": sum(1 for r in runs if r["reproduces"]),
            "of_runs": len(runs),
        }
        print(f"  {label:22s} {ident.get('version', '?'):20s} "
              f"blk_cnt={ident.get('has_blk_cnt')}  "
              f"reproduced {arms[label]['reproduced_in']}/{len(runs)} cold runs")

    ours = arms["installed_build"]
    up = arms["stock_upstream_wheel"]
    controls_ok = ours["is_our_build"] is True and up["is_our_build"] is False
    if not controls_ok:
        verdict = ("INCONCLUSIVE: the two arms did not identify as different "
                   "wheels, so nothing here separates upstream's code from ours")
    elif up["reproduced_in"]:
        verdict = ("upstream: the defect reproduces on a plain upstream wheel "
                   "that contains none of our code")
    else:
        verdict = ("not reproduced on the stock wheel in this many cold runs. "
                   "Given the effect is occupancy-sensitive and intermittent, "
                   "that is weak evidence and not an attribution to our fork")

    record = {
        "what": "whether token_aug's nondeterminism reproduces on a plain "
                "upstream comfy-kitchen wheel containing none of our changes",
        "produced_by": "bench/compare_token_aug_wheels.py",
        "slice": Path(args.slice).name,
        "launches_per_run": args.launches,
        "arms": arms,
        "positive_controls_passed": controls_ok,
        "verdict": verdict,
        "cannot_settle": [
            "which upstream version introduced it. Only the pinned upstream "
            "release was tested, not the history of the token routing code.",
            "that another GPU behaves the same. Reproduction depends on "
            "occupancy, so the arms share this machine's scheduling.",
        ],
    }
    print(f"\n{verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
