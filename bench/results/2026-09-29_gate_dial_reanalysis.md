# #36's gate dial, read again from the saved latents and measure records (2026-09-29)

Second pass on `2026-09-29_fasth3_gate_dial.md` (h3dude). Script:
`bench/analyze_gate_dial_reanalysis.py`; output: `2026-09-29_gate_dial_reanalysis.json`.
CPU only, nothing rendered, one seed and one clip per arm as in the original.

## What holds

- **The write-up's table matches the measure records.** hf, moved share, detail
  and chroma for all three scenes agree with the
  `2026-09-29_dial_<scene>_{resolution,tone}.json` rows (`table` in the JSON).
  The contrast column was not checked.
- **D3 holds independently.** On the saved final latents, scale 0 is
  `torch.equal` to the no-gates arm for video and audio, and alpha 1 through the
  loader is `torch.equal` to the FastH3 checkpoint arm (`equalities`). Both
  controls come out unequal (scale 0 against the FastH3 arm; scale 0.75 against
  alpha 1), so the comparison can fail and does not here. My first attempt at
  this read a zero-size version key and reported everything equal; the script
  reads `latent_tensor` and prints the shape.
- D2's ratios follow from the table: moved share falls faster than detail on
  all three scenes, and the falsifier is not met.

## What the write-up leaves loose

- **D1 says the fall is monotonic, and one cell is flat.** On slapstick, hf at
  0.75 equals hf at alpha 1 to the four digits recorded; detail, chroma and
  moved share do fall there. "Falls with the scale" is true of every scene's
  detail and of hf as non-increasing, not strictly of every hf cell.
- **Each notch is a different trajectory, not the same picture turned down.**
  The latent path's distance from alpha 1's (`mean_rel_l2_to_alpha1_look_anchor`)
  is already large at 0.75, grows with the drop, and is largest at 0. So a
  clip at 0.75 is not alpha 1's clip with less polish; the dial changes the
  sample as well as the look. That matches `CLAUDE.md`'s rule that a rendered
  clip cannot A/B a numerical change, and it is why the blind batch and the
  measures should be read as looks at three or four different clips. This record
  has no baseline for how far two unrelated seeds sit from each other, so it does
  not say how much of the distance is the gate and how much is chaos.
- **The scenes are one seed each.** The ordering is consistent across three
  scenes and three scales, which is more than one clip, but the size of any step
  is not a measured effect.
