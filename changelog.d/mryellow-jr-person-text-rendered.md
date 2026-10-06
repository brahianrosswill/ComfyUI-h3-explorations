bump: minor

### Changed

- **`MiniMaxH3MaskedPrompt`'s inputs are redesigned to be said in the
  user's own words** (owner, 2026-10-06: each field clear about what it
  does, combined where that makes sense, working out what it can and
  showing it). The node shipped earlier the same day and nothing but this
  pack's graphs used it.
  - `subject` is free text where it was a choice of three: who the still
    shows, completing "<Subject 1> is the ... shown in <Picture 1>". `woman`,
    `blonde haired woman` and `man wearing a red cap` all fit. The pronouns
    are read off it (the first man or woman word decides; none means
    "their"). The three texts that have rendered are `person`, `man` and
    `woman`, and the node writes them as before.
  - `extra` is renamed `add_to_shot`, which is what it does.
  - The node shows one line per input above the text: the sentence the
    subject makes and the pronouns with the word that decided them, where
    `picture_gives` came from, and whether the movement comes from
    `<Video 1>` or the prompt.
  - A brace or a `__list__` placeholder in either text field passes through
    as typed.
- The docs and comments that said the node's "person" motion text had not
  rendered now say what its one render showed, and point at the record:
  `masked_prompt_text.py`, the comment on `h3_config.MASKED_PROMPT`,
  `docs/wiki/masked_v2v.md` (which now says to set `subject` to match the
  still), `docs/wiki/decisions.md`, `docs/prompt_audit.md` and the bank
  entry's `tests` line. No default text moved.
- The eight masked graphs are rebuilt on the new inputs.

### Not done

- Nothing beyond the three single-word subjects has rendered. Whether more
  words about the still hold a look better is the arm set proposed to the
  owner.
