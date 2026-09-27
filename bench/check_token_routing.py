#!/usr/bin/env python3
"""`token_routing` on MiniMaxH3Sol becomes a {block: budget} map; assert which.

The choice exists so nobody has to remember `blocks=budget` syntax, which
makes its failure mode a quiet one: a choice that resolves to the wrong blocks
still renders, and the render still looks like a render. Each case below is
something that would be wrong without anything else noticing.

1. **Off means off, and custom reads its list.** `sol_routing_blocks` with
   "off" is empty whatever text is passed; "custom" is exactly
   `parse_token_aug_profile` on its `blocks` string, and an empty list is
   refused rather than run as off.
2. **A preset reaches the last blocks unguarded.** On the block-49 capture
   token routing raised the error unless the balance factor was on
   (`bench/results/2026-09-15_sol_token_aug_x_options_b49_s15.json`), so
   "early and middle" must stop short of the last `TOKEN_ROUTING_TAIL`, and
   "all blocks" must be refused unless the quantizer turns `qk_balance` on.
   **Balance alone is the requirement, not balance and rotation**: the old
   node also demanded `rotate`, which the record does not support (redesign
   bug #7). So "all blocks" is refused for "plain" and "rotated" and accepted
   for "balanced" and "balanced+rotated", read through `SOL_QUANTIZERS`; the
   "rotated" refusal is the red control that rotation alone does not pass.
3. **The node offers what the function takes.** The `token_routing`
   DynamicCombo's options are exactly `SOL_ROUTING_CHOICES`, in order, "off"
   comes first (a DynamicCombo's first option is its default), and only
   "custom" carries an input, its `blocks` string, whose default parses.

No GPU, no model, no server.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
H3_BLOCKS = 50


def _load():
    # Same path order as check_exact_blocks.py, for the reason given there.
    sys.path.insert(0, str(REPO.parent))
    sys.path.insert(0, str(REPO.parents[1]))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    return __import__(f"{REPO.name}.sol_attn_h3", fromlist=["sol_routing_blocks"])


def _raises(problems, label, fn):
    try:
        got = fn()
    except ValueError:
        return
    problems.append(f"{label}: expected a refusal, got {got!r}")


def main() -> int:
    problems: list[str] = []
    print("token_routing resolves to the blocks its label names:")
    try:
        m = _load()
    except Exception as exc:                       # pragma: no cover
        print(f"  FAIL  cannot import sol_attn_h3: {exc!r}")
        return 1
    n = H3_BLOCKS

    def r(choice, spec="", count=n, quantizer=m.SOL_QUANTIZER_DEFAULT):
        qk_balance, _rotate = m.SOL_QUANTIZERS[quantizer]
        return m.sol_routing_blocks(choice, spec, count, qk_balance=qk_balance)

    # 1. off is off whatever text is passed; custom reads the list
    for spec in ("", "0,24,32=64"):
        if r(m.SOL_ROUTING_OFF, spec) != {}:
            problems.append(f"'off' did not resolve to off with list {spec!r}")
    for spec in ("0,24,32=64", "0-10=128; 40=64"):
        want = m.parse_token_aug_profile(spec, n)
        if not want or r(m.SOL_ROUTING_CUSTOM, spec) != want:
            problems.append(f"'custom' does not read the list {spec!r}")
    _raises(problems, "custom with an empty list", lambda: r(m.SOL_ROUTING_CUSTOM, ""))
    _raises(problems, "custom with a blank list", lambda: r(m.SOL_ROUTING_CUSTOM, "  "))
    _raises(problems, "custom with a budget the kernel refuses",
            lambda: r(m.SOL_ROUTING_CUSTOM, "0=100"))
    _raises(problems, "an unknown choice", lambda: r("everything"))
    _raises(problems, "the retired 'text field' value", lambda: r("text field"))

    # 2. where each preset reaches
    got = r(m.SOL_ROUTING_MEASURED)
    if got != {b: m.TOKEN_ROUTING_BUDGET for b in m.TOKEN_ROUTING_MEASURED_BLOCKS} \
            or tuple(m.TOKEN_ROUTING_MEASURED_BLOCKS) != (0, 24, 32, 40):
        problems.append(f"measured preset resolved to {sorted(got)}")
    got = r(m.SOL_ROUTING_EARLY_MIDDLE)
    if sorted(got) != list(range(n - m.TOKEN_ROUTING_TAIL)) \
            or set(got.values()) != {m.TOKEN_ROUTING_BUDGET}:
        problems.append(f"early-and-middle preset resolved to {min(got)}..{max(got)}, "
                        f"expected 0..{n - m.TOKEN_ROUTING_TAIL - 1}")
    if n - 1 in got:
        problems.append("early-and-middle preset reaches the last block")
    if set(m.SOL_QUANTIZERS) != {"plain", "balanced", "rotated", "balanced+rotated"}:
        problems.append(f"SOL_QUANTIZERS is {sorted(m.SOL_QUANTIZERS)}; this case names four")
    for quantizer in ("plain", "rotated"):
        _raises(problems, f"all blocks under quantizer {quantizer!r}",
                lambda quantizer=quantizer: r(m.SOL_ROUTING_ALL, quantizer=quantizer))
    for quantizer in ("balanced", "balanced+rotated"):
        try:
            got = r(m.SOL_ROUTING_ALL, quantizer=quantizer)
        except ValueError as exc:
            problems.append(f"all blocks refused under {quantizer!r}, which turns the "
                            f"balance on: {exc}")
            continue
        if got != {b: m.TOKEN_ROUTING_BUDGET for b in range(n)}:
            problems.append(f"all-blocks preset under {quantizer!r} resolved to "
                            f"{len(got)} blocks of {n}")
    _raises(problems, "measured preset on a 10-block model",
            lambda: r(m.SOL_ROUTING_MEASURED, count=10))

    # 3. the node's DynamicCombo offers exactly what the function takes
    schema = m.MiniMaxH3Sol.define_schema()
    combo = next((i for i in schema.inputs if i.id == "token_routing"), None)
    if combo is None:
        problems.append("MiniMaxH3Sol declares no token_routing input")
    else:
        keys = [o.key for o in combo.options]
        if keys != list(m.SOL_ROUTING_CHOICES):
            problems.append(f"token_routing offers {keys}, the function takes "
                            f"{list(m.SOL_ROUTING_CHOICES)}")
        if not keys or keys[0] != m.SOL_ROUTING_OFF:
            problems.append(f"token_routing's first option (its default) is "
                            f"{keys[:1]}, not 'off'")
        carriers = {o.key: [i.id for i in o.inputs] for o in combo.options if o.inputs}
        if carriers != {m.SOL_ROUTING_CUSTOM: ["blocks"]}:
            problems.append(f"only 'custom' should carry an input ('blocks'); got {carriers}")
        else:
            blocks = next(o for o in combo.options if o.key == m.SOL_ROUTING_CUSTOM).inputs[0]
            try:
                if not r(m.SOL_ROUTING_CUSTOM, blocks.default):
                    problems.append(f"custom's default list {blocks.default!r} routes nothing")
            except ValueError as exc:
                problems.append(f"custom's default list {blocks.default!r} is refused: {exc}")

    if problems:
        print(f"\n  FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\noff is off, custom reads its list, no preset reaches the last blocks "
          "without the balance, and the node offers what the function takes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
