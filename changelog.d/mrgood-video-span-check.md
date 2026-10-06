bump: patch

### Added

- **The split reference pass is held to core on a span shaped like a video
  reference.** `bench/check_reference_encode.py` compared
  `reference_encode.py`'s copied loop with core's `Llama2_.forward` on spans
  that hold one vision block at most. Core tokenizes a video reference as
  one span of several vision blocks with a timestamp's text between them
  (`MiniMaxH3Tokenizer.tokenize_with_weights`), which no case had. The
  small-model comparison now has it: a still's span, then a span of three
  vision blocks with text between, then a prompt. It passes inside
  `SMALL_MODEL_TOL` like the still cases; the check prints the figure.
  Asked for by the masked lane's speed plan before `keep_references` is
  considered for a graph with a motion reference. Not established by this:
  the values on the shipped encoder at a real video span's length, which
  needs the encoder (`bench/measure_keep_references.py`).
