#!/usr/bin/env python3
"""`meta_sam3/` differs from Meta's code by the rule and the declared edits, and by nothing else.

`meta_sam3/sam3` is a copy of Meta's SAM 3.1 inference code under the SAM
License, edited where it must be to run inside ComfyUI
(`bench/build_meta_sam3.py` says how it is made). A copy that drifts from its
declaration is a fork nobody can read, so this compares the tree with upstream
itself.

## What is graded

  the declaration     the diff between "upstream with the rule applied" and
                      the tree is regenerated and compared with the tracked
                      `meta_sam3/EDITS.diff` byte for byte. An edit made in the
                      tree and not declared is red, and so is a declared edit
                      that is no longer in the tree.
  the file list       `meta_sam3/FILES.txt` is the upstream paths the tree
                      holds, no more and no fewer.
  the licence         `meta_sam3/LICENSE` is upstream's file, byte for byte.
  the headers         every Python file under `meta_sam3/` starts with the
                      header that says where it is from and what licence it is
                      under, or with the line that says it is ours.
  one direction       nothing in the copy imports from outside the copied
                      package, by a relative import that climbs out of it or
                      by an absolute `sam3` or `meta_sam3`. Our wrapper imports
                      Meta's code; Meta's code never imports ours.
  no download         nothing in the copy imports `huggingface_hub`. Upstream's
                      builder fetches the weights when given no path; the copy
                      must not be able to.

## What this cannot tell you

That the copy RUNS, or gives Meta's output: it reads text and imports nothing.
That the edits are right: it shows they are the declared ones. And it cannot
tell a `coderef/sam3` that someone edited and committed from upstream's own
commit of the same name; it reads `HEAD` and the working tree's state.

Returns 2 when `coderef/sam3` is absent or not at the commit the copy is made
from (`build_meta_sam3.COMMIT`): `coderef/` is gitignored, so a clone without
it has nothing to compare against.

    python bench/check_meta_sam3_copy.py
"""

from __future__ import annotations

import ast
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_meta_sam3 as build  # noqa: E402  the rule and the declaration have one copy, there

REPO = build.REPO
COPY = build.COPY
TREE = build.TREE
#: Top-level names the copy may not import: itself by an absolute name, and the download library.
FORBIDDEN_IMPORTS = (build.PKG, "meta_sam3", "huggingface_hub")

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    why = build.upstream_state()
    if why:
        print(f"  SKIP  {why}; nothing to compare the copy against")
        return 2
    if not TREE.is_dir():
        print(f"  FAIL  {TREE.relative_to(REPO)} is absent")
        return 1

    tracked = (COPY / "EDITS.diff").read_text()
    found = build.declaration()
    drift = list(difflib.unified_diff(tracked.splitlines(True), found.splitlines(True), "declared", "found"))
    check("the tree differs from upstream by the rule and the declared edits only", tracked == found,
          "" if tracked == found else "first lines of what the declaration does not record:\n" + "".join(drift[:30]))

    listed = (COPY / "FILES.txt").read_text()
    check("FILES.txt is the upstream paths the tree holds", listed == build.file_list(),
          "" if listed == build.file_list() else "run bench/build_meta_sam3.py --declare, and read what changed")

    check("LICENSE is upstream's, byte for byte",
          (COPY / "LICENSE").is_file() and (COPY / "LICENSE").read_bytes() == (build.UPSTREAM / "LICENSE").read_bytes())

    no_header, leaves, forbidden = [], [], []
    for path in sorted(COPY.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        rel = path.relative_to(COPY)
        if not (text.startswith(build.HEADER) or text.startswith(build.OURS)):
            no_header.append(rel.as_posix())
        # depth inside the copied package; a file beside it (meta_sam3/__init__.py) may not climb at all
        depth = len(path.relative_to(TREE).parts) if TREE in path.parents else 0
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.level > depth:
                leaves.append(f"{rel.as_posix()}:{node.lineno}")
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else (
                [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    forbidden.append(f"{rel.as_posix()}:{node.lineno} imports {name}")
    check("every Python file carries its header", not no_header, ", ".join(no_header[:8]))
    check("no relative import leaves the copied package", not leaves, ", ".join(leaves[:8]))
    check("no absolute import of the copy itself, and no download library", not forbidden, "; ".join(forbidden[:8]))

    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print("meta_sam3 is upstream at the named commit, the rule, and the declared edits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
