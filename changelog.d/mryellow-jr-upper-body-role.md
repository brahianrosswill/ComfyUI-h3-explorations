bump: minor

### Added

- **`MiniMaxH3MaskedPrompt` can say the still gives `the head and upper
  body`** (`picture_gives`). The text keeps the still's face, hair, headwear
  and the clothing above the waist, and says the legs and what is worn below
  the waist are the scene's own. It is chosen on the node and never read off
  the Masked Source, whose `replace` has no value that means it: wire
  Sapiens2's hair, face and neck, upper clothing and hands into `parts` and
  set `replace` to `the wired parts`. It is for a still that shows the
  person from the chest up. With no motion reference the text ties the upper
  body's movement to the legs below it.
  - Where it came from: a text typed for one still, rendered twice on the
    ref2va motion graph with those parts wired, at two window lengths, one
    clip and one seed. The role is that text with the still's own words moved
    into `subject`.
  - `bench/check_masked_prompt.py` gains a case for it.

Not rendered: the node's wording for this role, with or without a motion
reference. No shipped graph wires the matching parts yet.
