# #36: a strength dial on FastH3's gates, read (2026-09-29)

`bench/fasth3_gate_dial_arms.json` (predictions D1-D3 committed before any
render). FastH3 V2 through `MiniMaxH3OverlayLoader` with every piece and
`gate_scale` 0.75 and 0.5 on three scenes, and 0 on look_anchor; FastH3's
contract harness, seed 730451892, each scene at its written length, one clip per
arm. Alpha 1 is the loader's own bit-exact FastH3 (`overlay_all`; look_anchor's
row is in `2026-09-29_overlay_loader_render.jsonl`, the other two scenes'
here). Rows: `2026-09-29_fasth3_gate_dial.jsonl`. Measures:
`bench/analyze_followup.py --group dial`, in
`2026-09-29_dial_<scene>_{tone,temporal,resolution,divergence}.json`.

| scene | gate scale | hf | detail | chroma | contrast | moved |
|---|---|---|---|---|---|---|
| look_anchor | 1 | .0113 | .0488 | .129 | .243 | .097 |
| | .75 | .0111 | .0427 | .111 | .244 | .077 |
| | .5 | .0096 | .0375 | .096 | .225 | .058 |
| | 0 | .0076 | .0238 | .068 | .185 | .027 |
| slapstick | 1 | .0257 | .1039 | .203 | .264 | .358 |
| | .75 | .0257 | .0980 | .194 | .254 | .286 |
| | .5 | .0222 | .0862 | .176 | .245 | .228 |
| radio_drama | 1 | .0264 | .0806 | .158 | .240 | .151 |
| | .75 | .0254 | .0759 | .137 | .236 | .114 |
| | .5 | .0184 | .0622 | .123 | .221 | .094 |

## Read

1. **D1 held: detail and hf fall monotonically with the scale** on every
   scene.
2. **D2 held: motion falls with them, and faster.** Moved share drops to about
   three quarters at 0.75 and about 0.6 at 0.5, where detail drops to about
   nine tenths and about four fifths. The falsifier (moved share within 10% at
   0.5 while detail is below 0.8 of alpha 1) is not met on any scene. So the
   dial is a look-and-motion dial, not a detail-only control: turning the
   polish down takes motion with it, and also chroma.
3. **D3 held, exactly: scale 0 is the no-gates checkpoint.** On look_anchor
   the scale-0 render's final latent is `torch.equal` to the `fasth3_nogates`
   hybrid-file arm of `2026-09-29_fasth3_gates.md`, video and audio. So a zero
   gate removes the coarse branch's contribution completely, and the earlier
   no-gates result (worse by eye: darker, quiet audio) is the far end of this
   dial. This does not separate gate values from the branch's presence: a
   scaled gate is the same branch at lower weight.
4. **What this cannot say.** One seed, three scenes, tone and resolution
   measures; the owner's eye is a blind batch on look_anchor
   (`fasth3_gate_dial_look_anchor`), not yet scored. Whether a scale below 1
   reads as "less over-polished" rather than "worse" is exactly what that batch
   asks; the measures say it also costs motion and colour.
