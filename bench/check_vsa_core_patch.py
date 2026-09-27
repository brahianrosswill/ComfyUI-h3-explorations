#!/usr/bin/env python3
"""Report whether this ComfyUI builds H3's VSA gate, and whether consistently.

VSA on H3 needs ComfyUI core to build a `to_gate_compress` slot on every DiT
block. **Stock core does since core commit e308cc73** ("Add Sparse Attention
node", #16072). Before that the support existed only as the DRAFT pull request
recorded below, which was applied here as an uncommitted change on 2026-08-30
and lost on 2026-08-31. This file keeps that provenance and notices when the
support half-exists.

**It does not tell stock support from a local patch.** "Present" means the
token is in both files, which is true of stock core past e308cc73 and of a
draft-patched older core alike. Whether this checkout carries a local change
is core's git state, which this check does not read.

## Why absence is reported and not failed

A core without the support (older than e308cc73) is a legitimate state, and
failing on it would train a reader to ignore red. What is NOT legitimate is
the support present in one file and not the other, because the two halves fail
in opposite directions and the combination is silent:

  model.py only          every H3 model gets a `gate_compress` PARAMETER, but
                         detection never sets it, so it stays False and no
                         gate is ever built. Identical to not patching at all,
                         while `grep gate_compress comfy/` says it is there.
  model_detection only   detection sets `gate_compress=True` from the state
                         dict and the model constructor does not accept it.
                         That one raises, so it is the loud half.

So the graded case is CONSISTENCY, not presence.

## What this cannot tell you

That the support is CORRECT, or where it came from. The check greps both
files for one token; it compares no content hash and reads no git state, so
stock support, the old draft and a local edit all read as "present".

**And it cannot tell you the gate is USED.** Core's comment on the slot says
the weight is "unused by the dense forward; consumed by sparse attention
patches". Core loading it is necessary and not sufficient -- an attention
patch computes it and passes it to the kernel: core's `BlockSparseAttention`
(selection "vsa"), or `MiniMaxH3VSAAttention`, which is parked.

    python bench/check_vsa_core_patch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent

# ---------------------------------------------------------------------------
# Provenance of the draft patch applied on this box on 2026-08-30, kept as
# history. Stock core carries gate support since e308cc73 (#16072), so the
# constants below describe what this box once ran, not what it runs now.
# ---------------------------------------------------------------------------
UPSTREAM_REPO = "github.com/comfyanonymous/ComfyUI"
PR_NUMBER = 15958
PR_TITLE = "Minimax-H3: support FastVideo VSA"
PR_AUTHOR = "kijai"
PR_STATE = "DRAFT, still open as of 2026-08-31"
#: **The decision, 2026-08-31: do not apply it. Wait for the merge.** Superseded
#: 2026-09-27: core built the gates itself in e308cc73 (#16072).
#: It was applied to this box's working tree on 2026-08-30 and is GONE --
#: a `git reset` followed by two pulls took master to 95d755cd and carried
#: the uncommitted change away with it. Rather than re-apply a draft, this
#: box now tracks stock ComfyUI and waits. So ABSENT is the expected state
#: here, and every case below grades that state as correct rather than as
#: a shortfall -- see the checkpoint case for the one that used to not.
PR_APPLY_POLICY = "wait for merge; do not apply the draft"
#: The PR head this box's working tree was patched from, applied 2026-08-30 as
#: an UNCOMMITTED working-tree change on master rather than a merge. That is
#: deliberate: `git checkout -- <the two files>` reverts it, and a later
#: `git pull` REFUSES rather than quietly merging a draft into master.
PR_HEAD = "10febb01"
PR_BASE = "0a33ed6c"

#: The artifact the patch exists for. Machine-specific, so its case skips
#: when absent rather than failing.
VSA_CHECKPOINT = ("minimax_h3_fastvideo_vsa_datafree_1300step"
                  "_4step_int8_convrot.safetensors")

TOUCHED = {
    "comfy/ldm/minimax/model.py": "gate_compress",
    "comfy/model_detection.py": "gate_compress",
}

failures = []
skipped = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    print("VSA core support: stock core since e308cc73 (#16072). Before that, "
          f"{UPSTREAM_REPO} PR #{PR_NUMBER} ({PR_STATE})")
    print(f"  \"{PR_TITLE}\" by {PR_AUTHOR}, head {PR_HEAD} on {PR_BASE}\n")

    present = {}
    for rel, token in TOUCHED.items():
        path = COMFY / rel
        if not path.is_file():
            print(f"  SKIP  {rel} not found; is this a ComfyUI checkout?")
            return 2
        present[rel] = token in path.read_text()

    hits = [rel for rel, ok in present.items() if ok]
    misses = [rel for rel, ok in present.items() if not ok]

    patched = bool(hits) and not misses

    if not hits:
        print("  none  gate support is ABSENT: this core predates e308cc73 "
              "(#16072). Not a failure.")
        print("        Update core rather than apply the old draft PR.")
        print("        H3 VSA checkpoints load with their gate keys DROPPED and "
              "render as\n        the dense base. MiniMaxH3VSAAttention refuses "
              "rather than let that pass.")
        check("consistent", True, "absent from both files, which is coherent")
    elif misses:
        check("consistent", False,
              f"present in {hits} but not {misses}. Half-present support is "
              f"worse than none: with only the model change every H3 model "
              f"takes a gate_compress parameter that detection never sets, so "
              f"it silently stays False and looks exactly like stock.")
    else:
        check("consistent", True,
              f"present in both files ({', '.join(TOUCHED)})")
        print("        Stock core carries this since e308cc73 (#16072). This "
              "check cannot tell\n        stock support from a local edit; "
              "core's git state can.")

    # Is the SERVER running the core with gate support, or code from before it?
    # ("patch" in the case names below dates from the draft; on stock core it
    # means the two gate-support files.)
    #
    # **This case was vacuous when first written**, and the way it was vacuous
    # is worth keeping. It imported `comfy.ldm.minimax.model` in this process
    # and asked whether `Attention.__init__` takes `gate_compress` -- a fresh
    # import, reading the same files this check had just read, so it agreed
    # with them by construction and could not fail. The question that matters
    # is not what the files say, it is whether the process serving requests has
    # LOADED them, and a fresh import cannot see that.
    #
    # So: compare the port owner's start time against the patched files'
    # mtimes, which is the same thing `bench/restart_comfy.sh --newer-than`
    # asserts and for the same reason -- three measurements on 2026-08-29 were
    # taken against a server that had not reloaded, and every one produced a
    # plausible number rather than an error.
    import subprocess

    def port_owner(port=8188):
        try:
            out = subprocess.run(["ss", "-lptnH", f"sport = :{port}"],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        # `ss` emits the owner as one whitespace-delimited token,
        # `users:(("python3",pid=289502,fd=44))`, so the pid is INSIDE it
        # rather than at the start of a token. Splitting and testing
        # `startswith("pid=")` finds nothing and skips forever -- which is what
        # this did when first written, and a permanent skip is indistinguishable
        # from "no server" in the output. Same expression restart_comfy.sh uses.
        import re
        match = re.search(r"pid=(\d+)", out)
        return int(match.group(1)) if match else None

    pid = port_owner()
    if pid is None:
        skipped.append("the server postdates the patch")
        print("  SKIP  the server postdates the patch   "
              "nothing is listening on 8188")
    else:
        try:
            started = (Path("/proc") / str(pid)).stat().st_mtime
        except OSError:
            started = None
        if started is None:
            skipped.append("the server postdates the patch")
            print(f"  SKIP  the server postdates the patch   "
                  f"cannot read the start time of pid {pid}")
        else:
            newest = max((COMFY / rel).stat().st_mtime for rel in TOUCHED)
            check("the server postdates the patch", started > newest,
                  f"pid {pid} started after the gate-support files were written"
                  if started > newest else
                  f"pid {pid} started {newest - started:.0f}s BEFORE the "
                  f"gate-support files were last written, so it is serving "
                  f"older code. Restart before believing any VSA result.")

    # Does the patch actually do its job on the artifact it exists for?
    #
    # The cases above check that the patch is THERE. This checks that it WORKS,
    # against the published checkpoint, and it is the only case here that
    # touches a real artifact. Skipped rather than failed when the checkpoint
    # is not on the box, which is the normal state for anyone who has not
    # downloaded 30-odd GB.
    #
    # **And skipped when the patch is absent, which is the fix for a defect
    # this file argued against and then committed.** The docstring above says
    # absence is legitimate and that failing on it "would train a reader to
    # ignore red" -- then this case failed on exactly that, because it asked
    # whether the gate keys find a slot without first asking whether anything
    # was supposed to build one. Support absent plus checkpoint present is a
    # coherent state (core older than e308cc73 with a file downloaded), and it
    # was the decided state from 2026-08-31 until core gained the support. Red
    # there is noise. The case still has teeth where they belong: with the
    # support present, as on stock core now, a checkpoint whose gate keys find
    # no slot is a real failure and still fails.
    #
    # Nothing is allocated: the state dict is meta tensors carrying only the
    # real shapes, because detection reads `.shape` on a handful of entries and
    # the model is constructed under `torch.device("meta")`.
    ckpt = COMFY / "models" / "diffusion_models" / VSA_CHECKPOINT
    if not patched:
        skipped.append("the checkpoint's gate keys find a slot")
        print(f"  SKIP  the checkpoint's gate keys find a slot   "
              f"gate support is absent, so NO gate slot is built and all "
              f"gate weights\n        would be dropped on load. That is what "
              f"core before e308cc73 does;\n        it is not a shortfall to "
              f"grade. MiniMaxH3VSAAttention\n        refuses on such a model.")
    elif not ckpt.exists():
        skipped.append("the checkpoint's gate keys find a slot")
        print(f"  SKIP  the checkpoint's gate keys find a slot   "
              f"{VSA_CHECKPOINT} is not in models/diffusion_models")
    else:
        try:
            # ComfyUI is not installed; it is a checkout beside this pack.
            if str(COMFY) not in sys.path:
                sys.path.insert(0, str(COMFY))
            import torch
            from safetensors import safe_open

            import comfy.model_detection as detection
            import comfy.ops
            from comfy.ldm.minimax.model import MiniMaxH3Model

            with safe_open(str(ckpt), framework="pt") as handle:
                names = list(handle.keys())
                state = {k: torch.empty(handle.get_slice(k).get_shape(),
                                        device="meta") for k in names}
            config = detection.detect_unet_config(state, "")
            config.pop("image_model", None)
            with torch.device("meta"):
                model = MiniMaxH3Model(**config, operations=comfy.ops.manual_cast)
            slots = set(model.state_dict().keys())
            gates = {k for k in names
                     if "to_gate_compress" in k and k.endswith(".weight")}
            orphans = {k for k in names if k.endswith(".weight")} - slots
            placed = len(gates & slots)
            check("the checkpoint's gate keys find a slot",
                  gates and placed == len(gates) and not orphans,
                  f"{placed} of {len(gates)} gate weights placed, "
                  f"{len(orphans)} weight key(s) with no slot. Without gate "
                  f"support all {len(gates)} are dropped on load and the render "
                  f"succeeds as the dense base."
                  if not (gates and placed == len(gates) and not orphans) else
                  f"all {placed} gate weights placed, no orphan weight keys. "
                  f"Without gate support all {placed} would be dropped and the "
                  f"render would succeed as the dense base.")
        except Exception as exc:
            skipped.append("the checkpoint's gate keys find a slot")
            print(f"  SKIP  the checkpoint's gate keys find a slot   {exc}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    if skipped:
        # Exit 2. A check that did not run must not read as one that passed --
        # and this one skips exactly when nobody is serving, which is when a
        # reader is most likely to be about to start something.
        print(f"INCOMPLETE: {len(skipped)} case(s) skipped: {', '.join(skipped)}")
        return 2
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
