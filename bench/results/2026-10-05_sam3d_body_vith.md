# SAM 3D Body's ViT-H release behind core's nodes: one frame through both releases (2026-10-05)

lane: sam3d
verdict: the port runs on the card and puts a body where the DINOv3 model does; not an accuracy comparison

**Result: both releases put a mesh on the boxed person in the same pose, on
the CPU in float32 and on the card in half precision, and the two meshes
differ by a few centimetres.** The ViT-H release loads through this pack's
`MiniMaxH3SAM3DBodyViTHLoader` (`sam3d_body_vith.py`) and everything after
the loader is core's own node code. The numbers are in
`2026-10-05_sam3d_body_vith.json`; the script is
`bench/compare_sam3d_body_releases.py`.

## What ran

One frame of the band clip of `2026-10-04_masked_v2v_band.md`, at 104 s,
1920 by 1012, with the lead singer boxed by hand (x 660, y 0, 520 wide, the
full height; his legs leave the frame). For each release: load, core's
`SAM3DBody_Predict` with that one box and hand refinement on, core's
`SAM3DBody_Render` as a mesh over the frame and as a silhouette. The DINOv3
file is `sam_3d_body_dinov3.safetensors`
(`2026-10-05_sam3d_body_conversion.md`), through core's loader. The ViT-H
file is `sam_3d_body_vith.safetensors`, written the same day by
`bench/convert_sam3d_body_vith_checkpoint.py` from Meta's `model.ckpt` and
rig, no tensor renamed.

Two runs, each in its own process and not through a server: masked, on the
CPU in float32; and on the card, alone on it, with the server stopped.

## What the two releases produce

| | CPU, float32 | card, half |
|---|---|---|
| person found, both releases | yes | yes |
| silhouette inside the box, both releases | all of it | all of it |
| silhouette overlap between the releases (IoU) | 0.97 | 0.97 |
| 2D keypoints apart, mean over 70 (px, on a 1920-wide frame) | 13.9 | 13.5 |
| 2D keypoints apart, largest (px) | 38.5 | 39.2 |
| mesh vertices apart, mean (m) | 0.033 | 0.033 |
| mesh vertices apart, largest (m) | 0.077 | 0.077 |
| camera depth, DINOv3 then ViT-H (m) | 2.63, 2.59 | 2.63, 2.59 |

Read from the tile: the same standing pose from both, arms down and in
front of the hips; the ViT-H mesh sits a little higher and narrower at the
shoulders. The tiles are the owner's media and stay under `internal/`.

**This is not an accuracy comparison.** There is no ground truth on this
frame, and the two are different models; a few centimetres between them is
the size of difference one would expect and says nothing about which is
closer. What the table does establish is that the port is not broken: a
wrong crop, a shifted position encoding or mis-sampled keypoint features
would move the mesh off the person or deform it, and the silhouette stays
inside the box and overlaps the DINOv3 one almost entirely.

## Three things the run showed that the code did not

1. **Core runs both releases in half precision on this card, not bf16.**
   Both files are mostly bf16 (`comfy.utils.weight_dtype`), and core's
   loader passes that to `comfy.model_management.unet_dtype` with no
   `supported_dtypes`, which answers float16 here. Meta trained under bf16
   autocast. Each release's camera translation moves by half a millimetre
   between the float32 CPU run and the half-precision card run, so on this
   frame the precision does not matter. The ViT-H loader does what core's
   does; nothing was changed to force bf16.
2. **A script that calls core's nodes in process on the card must set up
   the server's dynamic VRAM layer first.** Without it core's OWN DINOv3
   path stops in `run_keypoint_prompt`, a float input meeting a half
   weight, because it is that layer's ops that cast a weight to its input
   at use. `bench/compare_sam3d_body_releases.py::server_memory_mode`
   transcribes the two blocks of ComfyUI's `main.py` that do it. The first card run failed this
   way before the ViT-H model was reached. On the CPU everything is float32
   and nothing shows.
3. **The ViT-H release takes the same square crop as the DINOv3 one.** Its
   config names `vit_hmr_512_384` and keeps `IMAGE_SIZE` at 512 by 512;
   Meta's estimator crops that square for either backbone and the model
   cuts the centre 384 columns out in six places
   (`sam3d_body_vith.py`'s docstring lists them). So core's predictor feeds
   it unchanged. A reading of a third-party C++ port had suggested the
   ViT-H crop would be 3:4; Meta's code says otherwise.

## Timing, with its conditions

Predict took about a second for DINOv3 and half a second for ViT-H on the
card, and about twelve and six on the CPU. Each is one frame, one person,
the first call after loading, with the weights' move to the device inside
it, on a card with nothing else on it. They show that ViT-H is the lighter
backbone (768 patch tokens against 1024 and its prefix tokens) and nothing
about throughput on a clip. `SAM3DBody.memory_used_forward`, which the
subclass inherits, is core's calibration for the DINOv3 backbone and
over-reserves for this one; not measured.

## Not done

- No clip, only one frame; no smoothing node; no second person.
- The node has not been run from a graph through a server. It registers:
  the first server started after it landed lists it in `/object_info`
  (mrorange, the same afternoon), `bench/check_node_ids.py` holds its id,
  and the loader function is the one this script calls.
- Which release is better for the mannequin route is the owner's eye on a
  clip, not this record.
