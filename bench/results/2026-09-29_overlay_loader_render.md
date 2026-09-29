# The overlay loader through a real render (2026-09-29)

`MiniMaxH3OverlayLoader` (`overlay_loader.py`) had only run on the CPU
(`2026-09-29_fasth3_overlay_exact.md`). This renders it on the server. Graphs:
`bench/make_overlay_loader_graphs.py`, the shipped FastH3 contract `_savelat`
graph with node 1 swapped for the loader (scratch graphs, not `workflows/`).
Look_anchor, seed 730451892, 345 frames, the harness of
`2026-09-29_fasth3_gates.md`. Rows: `2026-09-29_overlay_loader_render.jsonl`.

| arm | selection | compared with | final latent |
|---|---|---|---|
| `overlay_gates` | gates only, no backbone, refiner, adaln or io layers | the `fl2va_gates` hybrid-file arm | `torch.equal`, video and audio |
| `overlay_all` | every piece | the `fasth3_rerun` arm (FastH3's own file, same day) | `torch.equal`, video and audio |
| `overlay_no_refiner` | every piece but the refiner | `overlay_all` | not equal, as any change to the network re-rolls the trajectory |

## Read

1. **The selector renders exactly.** Gates only through the loader is the
   hybrid file's render bit for bit, and everything on is FastH3's file's
   render bit for bit, on video and audio. So the overlay is a working way to
   run these two checkpoints without their 20 GB files, on fl2va.
2. **Two loads of the same checkpoint on this build agree exactly.** That is
   the same-build floor the gates record lacked: `fasth3_rerun` differing from
   the 2026-09-27 FastH3 clip is therefore a change in the stack between the
   two days, not run-to-run noise (which the stack change is is still untested;
   the kitchen build is one candidate).
3. **Refiner off is a look, not a check.** It moves the trajectory as any
   change does; whether the picture is better, worse or the same is for the
   owner's eye (`internal/blind_keys/overlay_refiner_look_anchor.json`, batch
   `overlay_refiner_look_anchor`). One seed, one clip.

The loader's own load time is in the rows' per-node timings (node `1`), and it
ran while a CPU measurement was reading the same disk, so read it as an upper
bound.
