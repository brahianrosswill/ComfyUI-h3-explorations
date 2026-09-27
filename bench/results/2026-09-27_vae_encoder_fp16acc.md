# #33: the H3 video encode under fp16 accumulation, across the kitchen upgrade (2026-09-27)

`bench/grade_vae_encoder_precision.py --fp16-accumulation` (GPU, the switch
`start.sh`'s `--fast fp16_accumulation` sets), before and after lookingdude's
install:
- before: `0.2.35+sol.8176242`;
- after: `0.2.35+sol.863e953.up.c8c7825`, upstream main c8c7825 with #192.

Records: `2026-09-27_vae_encoder_fp16acc_{before,after}_672.json` and the
128x128 pair. The input is one random frame, seed 0.

| build | fp16-accumulate vs plain fp16 (mean) | fp16-accumulate from fp32 | plain fp16 from fp32 |
|---|---|---|---|
| 8176242 | 3.837e-4 | 7.341e-4 | 7.185e-4 |
| 863e953.up.c8c7825 | 3.837e-4 | 7.341e-4 | 7.185e-4 |

**No change, to every printed digit, at 672x384 and at 128x128.**

## Why: the encoder tiles

`comfy/ldm/minimax/vae.py` builds the H3 video VAE with `tiling=True` and
256-pixel tiles, and encodes through `tiled_encode` at any input size.
`space_down=(2, 2, 2, 2, 1, 1)` and `ch_mult=(1, 2, 2, 4, 4, 8)` put the
512-channel stage at 1/16 scale, so its 3x3x3 convs (depth 13824) always see
16x16 tiles. A direct call of kitchen's `fp16_conv3d` on the new build
(this session, not saved) shows:
- 512 channels at 24x42 and 48x84: kitchen's error from fp32 equals cuDNN
  fp16's (3.3e-4). #192's small-launch rule (`kDeepK`) keeps these on fp32
  accumulation.
- 512 channels at 96x168: kitchen's error is 1.0e-2 against cuDNN's 3.3e-4,
  so fp16 accumulation is live at that launch size. The tiled encoder never
  reaches it.
- 256 channels at 96x168 (depth 6912, under the old gate too): 5.0e-3 against
  2.3e-4. The encoder's 256-channel convs meet this regime only at 32x32
  tiles. The 2% gap between the two builds' fp16 and accumulate arms (row
  above) is this shallower fp16 accumulation, present on both builds.

## Verdict

#33 closes: the kitchen upgrade does not change the H3 encode as ComfyUI runs
it. The encode's existing fp16 accumulation, in the convs under the old gate,
costs about 2% more error from fp32 than plain fp16, on both builds. That
predates the upgrade and is not a decision this experiment was for. It would
reopen if core ever encodes H3 untiled, or with tiles larger than about 1024
pixels.
