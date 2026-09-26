# INT8 against fp16 video VAE on the ref2va and i2va FlashGen probes, 2026-09-26

Rows: `2026-09-26_int8_vae_tasks_s1.jsonl`. The graphs are
`h3_probe_r2v_flashgen_4step` and `h3_probe_i2v_flashgen_4step` at 0.151.0,
which ship the INT8 VAE; the fp16 arm patches node 3's `vae_name` back to
`h3_config.VIDEO_VAE_FP16`. i2va is patched to `canvas="explicit"`, which
cover-crops the probe's square keyframe to 1344x768, so both tasks decode
1344x768 x 345 frames (checked on the clips). Seed 730451892, one warmup on
the fp16 arm, then fp16 and INT8. The server was unarmed and warm.

**Read the per-node times, not the totals.** Changing node 3 changes the input
signature of the conditioning node that encodes the references or keyframe,
so the INT8 row re-ran conditioning while the fp16 row reused the warmup's
cached result. That makes INT8's raw total look worse on ref2va (220.1 s
against 214.2 s). The encode itself is bit-identical between the two files
(`2026-09-26_vae_encoder_int8_file.json`), so conditioning is excluded below.

| task | arm | sampler s | decode s | combine s | total excl. conditioning |
|---|---|---|---|---|---|
| ref2va | fp16 | 180.0 | 29.9 | 3.8 | 213.7 |
| ref2va | INT8 | 180.7 | 17.6 | 3.8 | 202.1 |
| i2va | fp16 | 134.4 | 30.0 | 3.6 | 168.0 |
| i2va | INT8 | 135.4 | 17.6 | 3.7 | 156.7 |

The saving is the decode at this canvas, about 12 s, on both tasks as on
t2v (`2026-09-26_vae_decoders_345f.md`, "End to end after the switch"). The
sampler differences are run-to-run noise. The saving is a smaller share of
the ref2va render because its sampler is longer.
