bump: minor

### Added

- **`meta_sam3/`: Meta's SAM 3.1 inference code, copied into the pack.** The
  multiplex video predictor from facebookresearch/sam3 at the commit named
  in `bench/build_meta_sam3.py::COMMIT`, under the SAM License in
  `meta_sam3/LICENSE` and not under this pack's own licence, edited only
  where it must be to run inside ComfyUI. Nothing imports it yet: no node,
  graph or other check runs this code. It is the first step of using the
  segmenter the way its authors built it in place of ComfyUI core's port
  (owner's decision, 2026-10-06); `meta_sam3/README.md` says what the
  directory is, why each edit is there, and what the copy leaves to its
  caller.
- **`bench/build_meta_sam3.py` makes the copy** from a checkout of upstream:
  the paths in `meta_sam3/FILES.txt`, one rule on every Python file (a
  header naming the commit and the licence, and the package's absolute
  imports made relative), then the hunks of `meta_sam3/EDITS.diff`.
  `--declare` writes those two files from the tree, so an edit is made in
  the tree and declared, never typed into the diff.
- **`bench/check_meta_sam3_copy.py` holds the copy to its declaration.** It
  regenerates the diff against upstream and compares it with the tracked
  one byte for byte, holds `LICENSE` to upstream's bytes and every file to
  its header, and fails on an import that leaves the copied package, an
  absolute import of the copy, or an import of `huggingface_hub` (upstream's
  builder downloads the weights when it is given no path; the copy cannot).
  It returns 2 without `coderef/sam3` at the commit.

### Changed

- `README.md` gains a licence section: the pack is MIT except `meta_sam3/`.
  `vendor/README.md` points at `meta_sam3/` as the edited copy that is not
  kept there.
