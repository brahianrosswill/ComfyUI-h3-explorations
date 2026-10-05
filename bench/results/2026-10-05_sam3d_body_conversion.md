# SAM 3D Body: Meta's original files repacked for core, and what the repack holds that Meta did not publish (2026-10-05)

Session mryellow. Owner: "I'm going to download the original weights for
sam3d instead of comfy so i know how theyre converted exactly." Script:
`bench/convert_sam3d_body_checkpoint.py`; mapping check:
`bench/check_sam3d_body_conversion.py`. CPU only; nothing rendered.

## Sources

| file | from | size | sha256 |
|---|---|---|---|
| `model.ckpt` | `facebook/sam-3d-body-dinov3` (gated, sam-license) | 2,109,129,346 B | `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf` |
| `assets/mhr_model.pt` | same repo | 696,110,248 B | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` |
| `detection/sam_3d_body_dinov3_bf16.safetensors` | `Comfy-Org/sam-3d-body`, read by HTTP range request only | 2,831 MB (header) | not downloaded whole |

Output: `sam_3d_body_dinov3.safetensors` beside the originals in the Storage
folder (2,830,738,300 B), linked into `ComfyUI/models/detection/`.

## What was measured (all on 2026-10-05, CPU)

- `model.ckpt` is a flat pickled state dict: 1127 tensors; 551 backbone
  tensors in bf16, 568 fp32, 8 int64. No `state_dict` nesting.
- `mhr_model.pt` is a TorchScript module with 55 state-dict tensors in three
  submodules (`face_expressions_model`, `pose_correctives_model`,
  `character_torch`) and two attributes core needs as tensors:
  `character_torch.skeleton._pmi_buffer_sizes = [65, 56, 62, 83]` and
  `pose_correctives_model.pose_dirs_predictor.0.sparse_shape = (3000, 750)`.
  Both equal the repack's `mhr.skel_pmi_buffer_sizes` and
  `mhr.pose_corr_sparse_shape`.
- Core's model (`comfy/ldm/sam3d_body/model/model.py::SAM3DBody`) has 1627
  tensors; the repack has 1629 (plus `hand_cls_embed.*`, which the loader
  pops). 576 keys are identical by name between Meta's file and the repack;
  551 backbone keys are renamed; 1053 repack keys are not in Meta's file: 614
  backbone (the renamed ones), 421 `face_landmarker.*`, 17 `mhr.*`, and
  `head_pose.face_region_rgb`.
- **The backbone rename, verified bit for bit** on layer 0 and the
  embeddings before the script was written (q, k, v and o projections and
  biases, gate, up and down projections, both layer scales, norms, the class
  and register tokens, the patch embedding, the final norm): DINOv3's
  `blocks.N.attn.qkv` splits into thirds; `mlp.w1/w2/w3` are gate, up, down;
  `ls1/ls2.gamma` are `layer_scale1/2.lambda1`; `storage_tokens` are
  `register_tokens`. Then 24 randomly chosen tensors of the written file
  (seed 0) compared bit-identical to the repack.
- **Every q, k and v bias in Meta's backbone is exactly zero** (max abs 0.0
  over 32 layers), and the `qkv.bias_mask` buffers are all zero in bf16.
  DINOv3's `LinearKMaskedBias.forward` multiplies the bias by that mask
  (`coderef/dinov3/dinov3/layers/attention.py`, checkout 6876159), so Meta's
  forward applies no attention bias; core's k projection having no bias and
  its q and v biases being zero gives the same forward. The script refuses a
  non-zero k bias.
- `backbone.encoder.rope_embed.periods` (16 values, bf16) is dropped: core
  computes its rotary periods and has no buffer for them.
- Meta's checkpoint shares eight head buffers between `head_pose` and
  `head_pose_hand` (same storage); safetensors refuses shared memory, so each
  key is written as its own copy, as the repack does.

## What Meta did not publish, and where the file gets it

- `face_landmarker.*`: 421 tensors, 5.35 MB, a pure-PyTorch port of Google's
  MediaPipe `face_landmarker_v2_with_blendshapes.task`
  (`comfy_extras/mediapipe/face_landmarker.py`), used only by the
  face-expression node. Not Meta's and not in either original file.
- `head_pose.face_region_rgb`: 18439 x 3 fp32, values in 1/255 steps, 3975
  non-zero rows: a per-vertex colour map painted for the mesh renderer's
  face tint (`comfy_extras/sam3d_body/utils.py::compute_canonical_colors`).
  Not derivable from the rig.

By default the script takes both from the published repack by range request
and records the URL and the sha256 of the bytes taken in the file's
metadata. `--comfy-extras zeros` writes zeros instead: the file loads, the
mesh, silhouette and skeleton renders work, the face-expression node and the
face tint do not.

## What the written file passed

- Key set and shapes equal core's model after the loader's rename and pop;
  dtypes equal the published file's for every key.
- Readback `torch.equal` for every tensor.
- 24 random tensors bit-identical to the repack.
- Core's own `SAM3DBody_Loader.execute` on the CPU, through the link in
  `models/detection/`: strict load passed; the rig's level sizes and sparse
  shape read back as the values above.

## Not done

- No render, no pose prediction: this is a container change and its proof is
  the key check and the bit comparison. The first use is the mannequin route
  on the masking board (`route-mannequin`).
- `facebook/sam-3d-body-vith` was downloaded too; its backbone is not the
  DINOv3 layout and core's model is, so the script refuses it by name and no
  file was made from it.
