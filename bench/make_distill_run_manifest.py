#!/usr/bin/env python3
"""Write the manifest for the 2026-09-26 fresh distill run: many bank scenes, four models, one seed.

The owner, 2026-09-26: FlashGen's prompt adherence "needs badly to be tested. I'd
use a diverse set of scenes from prompt bank. Varying lengths and goals based on
the json metadata in prompt bank". Seed-matched clips are different scenes per
distill (`docs/h3_distills.md`), so the comparison is across many scenes,
judged against each scene's beat checklist (`bench/adherence_checklists.json`),
not across seeds of one scene.

Arms are grouped by model, so the card loads each model state once. Every arm
patches the scene's prompt by bank id and its `MiniMaxH3Resolution.length` to
the scene's declared frames, so no arm renders a prompt off its length.

    python bench/make_distill_run_manifest.py [--graphs graphs.json] --out bench/distill_run_arms.json

`--graphs` maps each role to a graph path: the saved-latent twins, once built.
Without it, the shipped graphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "prompt_bank" / "bank.json"

#: The scenes, chosen by bank metadata for spread in length, shot count,
#: speaker count and what each prompt tests (see the manifest's `why`).
SCENES = ["t2va_noodle_bar", "t2va_post_office", "t2va_rooftop_pov", "t2va_box_office",
          "t2va_radio_drama", "t2va_desert_crew", "t2va_slapstick_moving_piano",
          "t2va_kpop_dance_studio", "t2va_courtroom_verdict", "t2va_samurai_bamboo_duel",
          "t2va_silent_film"]
#: Model roles, in run order.
MODELS = ["base", "pdd8", "flashgen", "fasth3"]
DEFAULT_GRAPHS = {
    "base": "workflows/h3_text_to_video_api.json",
    "pdd8": "workflows/h3_text_to_video_pdd_api.json",
    "flashgen": "workflows/h3_text_to_video_flashgen_api.json",
    "fasth3": "workflows/h3_probe_t2v_fasth3_8step_contract_api.json",
    "flashgen_dense": "workflows/h3_probe_t2v_flashgen_r64_4step_branch_dense_api.json",
    "pdd8_refine": "workflows/h3_probe_t2v_pdd8_audio_refine_api.json",
    "flashgen_refine": "workflows/h3_probe_t2v_flashgen_4step_audio_refine_api.json",
    "route3": None,          # the VAE session's step-switch arm, once built
    "flashgen_s08": "workflows/h3_text_to_video_flashgen_api.json",
    "flashgen_s12": "workflows/h3_text_to_video_flashgen_api.json",
    "flashgen_noadaln": "workflows/h3_text_to_video_flashgen_api.json",
}
#: Extras, each on a subset. **Reasoned**:
#: - FlashGen dense on the motion scenes, where Sol's sparsity is most likely
#:   to cost adherence;
#: - refine on the two dialogue-and-music scenes, the pairs route 5 needs;
#: - route 3 where motion and detail both matter.
EXTRAS = {
    "flashgen_dense": ["t2va_slapstick_moving_piano", "t2va_kpop_dance_studio",
                       "t2va_samurai_bamboo_duel", "t2va_rooftop_pov"],
    "pdd8_refine": ["t2va_courtroom_verdict", "t2va_radio_drama"],
    "flashgen_refine": ["t2va_courtroom_verdict", "t2va_radio_drama"],
    "route3": ["t2va_slapstick_moving_piano", "t2va_kpop_dance_studio",
               "t2va_courtroom_verdict", "t2va_samurai_bamboo_duel"],
    "flashgen_s08": ["t2va_courtroom_verdict", "t2va_kpop_dance_studio", "t2va_samurai_bamboo_duel"],
    "flashgen_s12": ["t2va_courtroom_verdict", "t2va_kpop_dance_studio", "t2va_samurai_bamboo_duel"],
    "flashgen_noadaln": ["t2va_courtroom_verdict", "t2va_kpop_dance_studio", "t2va_samurai_bamboo_duel"],
}
#: FlashGen variants through `MiniMaxH3LoRABranch`'s granular inputs (0.152.0):
#: does its grade and adherence move with strength, and is the timestep
#: modulation (adaln) what carries the grade? **Reasoned** choices: +-20%
#: strength brackets the publisher's 1.0 without leaving the regime; "no adaln"
#: isolates the one module kind that shapes the per-step dynamics.
VARIANT_PATCHES = {
    "flashgen_s08": ["MiniMaxH3LoRABranch.strength=0.8"],
    "flashgen_s12": ["MiniMaxH3LoRABranch.strength=1.2"],
    "flashgen_noadaln": ['MiniMaxH3LoRABranch.modules="no adaln"'],
}
SEED = 730451892


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    graphs = dict(DEFAULT_GRAPHS)
    if args.graphs:
        graphs.update(json.loads(args.graphs.read_text()))
    frames = {p["id"]: p["frames"] for p in json.loads(BANK.read_text())["prompts"]}
    missing = [s for s in SCENES if s not in frames]
    if missing:
        raise SystemExit(f"not in the bank: {missing}")

    arms, patches, skipped = {}, [], []
    def add(role, scene):
        if graphs.get(role) is None:
            skipped.append(f"{role}:{scene}")
            return
        label = f"{scene.removeprefix('t2va_')}__{role}"
        arms[label] = graphs[role]
        patches.append(f"{label}:MiniMaxH3Conditioning.prompt=@bank:{scene}")
        patches.append(f"{label}:MiniMaxH3Resolution.length={frames[scene]}")
        for extra_patch in VARIANT_PATCHES.get(role, []):
            patches.append(f"{label}:{extra_patch}")
    for role in MODELS:
        for scene in SCENES:
            add(role, scene)
        for extra, scenes in EXTRAS.items():
            if extra.startswith(role + "_"):
                for scene in scenes:
                    add(extra, scene)
    for scene in EXTRAS["route3"]:
        add("route3", scene)
    # FlashGen at its trained length, against the base, on the five-second subway twin.
    for role in ("base", "flashgen"):
        add(role, "t2va_subway_chase_short")

    manifest = {
        "what": ("The fresh distill run, 2026-09-26: base, PDD8, FlashGen (rank 64 at the "
                 "call) and FastH3 (contract) on eleven bank scenes at one seed, each at its "
                 "declared length, plus FlashGen dense, refine pairs, the step-switch arm, and "
                 "the five-second subway twin. Built by bench/make_distill_run_manifest.py."),
        "why": ("The owner, 2026-09-26: test FlashGen's prompt adherence across diverse bank "
                "scenes, and what each distill is good and bad at (docs/h3_distills.md), fresh, "
                "with everything measured. Seed-matched clips are different scenes per "
                "distill, so the unit is the scene, judged blind against its beat checklist."),
        "scenes": {s: frames[s] for s in SCENES + ["t2va_subway_chase_short"]},
        "arms": arms,
        "patches": patches,
        "seeds": {"first": SEED, "note": "held across every arm"},
        "skipped_until_built": skipped,
        "run": (f"run_graph_arms.py --manifest {args.out} --runs 1 --seed {SEED} --hold-seed "
                f"--no-alternate --out bench/results/2026-09-26_distill_run.jsonl"),
        "predictions": "none registered",
    }
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(arms)} arms; skipped until built: {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
