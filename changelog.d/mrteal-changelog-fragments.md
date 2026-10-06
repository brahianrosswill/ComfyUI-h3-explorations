bump: patch

### Changed

- **A changelog entry is now a file of its own.** Write
  `changelog.d/<session>-<a-few-words>.md` with a `bump: patch` or
  `bump: minor` line and the entry under it, then run
  `bench/build_changelog.py --with <that file>`: it writes the entry into
  `CHANGELOG.md` and assigns the version from the order fragments were
  committed in. `changelog.d/README.md` is the three steps. Nobody edits
  the top of `CHANGELOG.md` by hand any more.
- **Why.** Sessions sharing this tree each added an entry to the top of
  one file and read the next number off it. Twice in one day two sessions
  took the same number, and once a commit by pathspec carried a peer's
  uncommitted entry. A file per entry cannot be swept into another
  commit, and a number assigned at build time cannot collide.
- **What is kept.** Every entry below the marker line in `CHANGELOG.md` is
  the changelog as it stood, byte for byte. The build refuses to write
  when it finds an entry above the marker that no fragment makes, so an
  entry added by hand out of habit is reported and never deleted.

### Added

- `bench/check_changelog.py`, in the sweep: `CHANGELOG.md`'s top is what
  the committed fragments make, with any uncommitted fragment on top, and
  the builder's promises hold on made-up text (numbering, a hand edit
  refused, a stale file red, a bad fragment refused).
- `bench/build_index.py` neither lists nor counts the files in
  `changelog.d/`, so adding an entry does not make a commit regenerate the
  root index; and `bench/check_doc_links.py` leaves the entry files
  unscanned, as it leaves `CHANGELOG.md`.
