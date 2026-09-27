# The subway_chase "1 s clone": base control (2026-09-27)

`bench/clone_base_x0_arms.json` (predictions written before rendering,
99f05455). The base ran on Euler over PDD's 32-point grid, with the x0
observer on, from the same seeded noise, prompt and length as the PDD8 x0 row.
Records:
- `2026-09-27_x0_steps_subway_base_euler32.json`, from
  `bench/x0_step_frames.py` over its 32 step x0s;
- `2026-09-26_clone_base_x0.jsonl`, the rows.
One render.

- **The base places two figures at the same moment.** By base step 4, the
  grid point PDD8's step 1 lands on, there are two dark shapes at latents
  7-9 (about 0.9-1.4 s).
- **The full-resolution frames at 1.1 s and 1.3 s show the same two people
  in both renders.** One wears a black leather jacket, the other a grey
  hoodie: the suspect and the agent the prompt names ("only two people in the
  whole station"). In the base they run through side by side. In PDD8 (exact)
  they overlap mid-vault at 1.1 s and separate by 1.3 s. The merged PDD8
  previews overlap the same way.

## Verdicts

- **B1 (the base also places two figures; the difference is identity):**
  holds on the count. On identity, both renders show two differently dressed
  people, so at full resolution PDD8's pair is not one person twice either. The
  moment that can read as a clone is PDD8's overlap at about 1.1 s, where one
  figure passes in front of the other.
- **falsifier_a (the base shows one figure): does not hold.**
- **So the two-figure composition belongs to the seed and prompt, not to
  PDD.** Together with the ladder (the same event at every PDD step count) and
  the reverse switch (it survives a FlashGen finish), the clone lane closes
  here, on one scene and one seed as the owner asked. What remains is the
  owner's eye on the 1.1 s overlap: a PDD motion artefact, or two people
  passing.
