bump: patch

### Changed

- **The masked graphs write the Subject Track's `corrections` input**, at
  its empty default (`h3_config.SUBJECT_TRACK`). It is the input a clip
  with many cuts needs most, and a graph that did not carry it could not be
  patched by a runner: the first render of such a clip went through a
  scratch copy of the graph for that reason. The eight masked graphs are
  rebuilt. The tracker's inputs are part of a kept mask's key, so a mask
  kept before this is tracked once more.
