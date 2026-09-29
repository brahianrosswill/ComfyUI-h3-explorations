# How much video attention goes to the reference images, by depth (2026-09-29)

In the captured ref2va cell (`2026-09-27_sol_test2_pdd8_ref2va`: a man as Picture 1,
a mountain landscape as Picture 2, one scene, one seed), exact float32 attention
of sampled video query rows over every key, at steps 2 and 6, blocks 0, 8, 24,
40 and 44 to 49. Records: `2026-09-29_ref2va_reference_attention_s2.json` and
`_s6.json` (`bench/analyze_attention_mass.py`). The reading below divides each
segment's mean mass by its share of the keys, so 1.0 means "as if keys were
equally likely".

## What it shows

1. **Reference use is a mid-depth activity.** Block 0's video queries put about a
   tenth of the uniform share on either reference image and read almost only
   video; the share rises through block 8 and peaks near block 24; it falls again
   toward the tail, where the text rows take several times their uniform share.
2. **The two references are not read alike.** At block 24, step 2, the landscape
   (Picture 2, the environment) gets more than its uniform share and the man
   (Picture 1, the identity) less. One scene, so this is a description, not a
   finding about how identity is carried.
3. **Later steps read the references less** at every block sampled (step 6
   against step 2), consistent with the reference rows being pinned near clean
   and the video coming to agree with them as it denoises. That is a reading.
4. **Peakiness grows with depth in the same way in both cells**
   (`eff_keys_median_head`, `heads_eff_under_20`), which is what
   `2026-09-29_ref2va_block49_hardest_cell.md` used.

## Not shown

- Anything about a defect. Nothing here has a control: no fl2va capture with the
  same references exists on disk (`workflows/h3_probe_capture_ref3_fl2va_api.json`
  makes one, and needs the card), so "reads the references less than it should"
  cannot be asked yet.
- Anything about how the reference is used other than through attention from
  video rows; nothing about audio queries or the audio reference.

## Correction notes, 2026-09-29, after verification

- **"Text" here is the whole text span.** In the ref2va cell the first 8424 rows
  include vision-embed rows tagged with the video modality (see the correction in
  `2026-09-29_ref2va_block49_hardest_cell.md`), so the text-mass figures are for the
  span, not for the prompt's text alone. Their number was not counted.
- **Only `heads_eff_under_20` is a stable peakiness measure.** `eff_keys_median_head`
  moves with the query sample (this run used 128 queries).
