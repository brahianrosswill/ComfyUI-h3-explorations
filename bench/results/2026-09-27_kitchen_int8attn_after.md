# Kitchen upgrade 0.158.0: same arithmetic, about 1% faster (2026-09-27)

lookingdude installed comfy-kitchen `0.2.35+sol.863e953.up.c8c7825` (0.158.0,
68ef7734), which merges upstream main's int8 attention work (#207, #208) into
the `h3-build` fork. The kitchen int8 attention is what our Sol graphs send
their dense tail (blocks 45, 48, 49) to, through `ModelAttentionBackend`.
FastH3's contract graph never reaches it (`2026-09-27_fasth3_bf16rows.md`).

**Runs:** the same labels, prompts, seed (730451892) and order as their
"before" rows on `0.2.35+sol.8176242`, each on a fresh server with one warmup
per model:
- `2026-09-27_int8attn_after.jsonl` against the `2026-09-26_followup.jsonl`
  rerun rows: PDD8 (exact branch) and FlashGen on slapstick, kpop, courtroom
  and samurai at 345 frames;
- `2026-09-27_pdd_strength_after.jsonl` against the VAE session's
  `2026-09-26_pdd_strength.jsonl`: PDD8 at strength 0.85 and 0.7 on five
  scenes.

The substrate records are `2026-09-27_int8attn_after_substrate.json` and
`2026-09-27_pdd_strength_after_substrate.json`.

## Output: unchanged

**All 20 saved final latents are bit-identical** (`torch.equal`) to their
before rows: 8 in the first set and 12 in the second.
- The arithmetic is unchanged across the upgrade, for these graphs.
- This agrees with lookingdude's kernel reading: #208 changes scheduling (Q
  kept in registers on sm_89, `CTA_Q=64` for mid-size shapes);
  `quant_qk_int8.cu` and `quant_v_int8.cu` are identical across the two
  builds; and the per-row reduction order over K tiles does not change.
- It also settles determinism: PDD8 on the exact branch and FlashGen
  reproduce bit for bit across server processes and across the upgrade.

## Speed: slightly faster

Per-label `sampler_s` against the before rows, so the VAE is excluded:
- PDD8 and FlashGen: 0.7 to 1.1% faster.
- PDD strength: 0.3 to 0.8% faster.

Every label got faster, and decode is flat.
- **Attributed:** to #208's scheduling on the dense tail, which is 3 of 50
  blocks. That is why the gain is small.
- **Not established:** this is one render per label. Run-to-run timing
  noise at this length has not been measured separately, so the size is
  indicative, not exact.

## Not done

A capture re-grade of `int8_attention` was cancelled before it ran.
- It calls the kernel directly, so it cannot show whether renders reach the
  new path.
- Its 2026-09-15 baseline was on kitchen 0.2.34.

Bit-identical latents with a consistent speedup are the evidence the renders
take the new path with the same math (lookingdude's point).
