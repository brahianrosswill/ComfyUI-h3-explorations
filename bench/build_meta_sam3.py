#!/usr/bin/env python3
"""Make `meta_sam3/sam3` from `coderef/sam3`: Meta's files, one rule, the declared edits.

`meta_sam3/` is Meta's SAM 3.1 inference code, copied into this pack under the
SAM License and edited only where it must be. This script is how the copy is
made and how an edit to it is declared, so that what differs from upstream is
a file a person can read (`meta_sam3/EDITS.diff`) and not something to trust.

    python bench/build_meta_sam3.py            # write meta_sam3/sam3 from coderef and EDITS.diff
    python bench/build_meta_sam3.py --declare  # write EDITS.diff and FILES.txt from the tree as it stands

## The three parts

  the file list   `meta_sam3/FILES.txt`: the upstream paths the copy holds.
  the rule        applied to every Python file of upstream's: the header
                  (`HEADER`), and each `from sam3.a.b import c` rewritten to
                  the relative import for the file's depth. The pack's own
                  directory name is not importable and another pack may ship
                  a top-level `sam3`, so the copy never names itself.
  the edits       `meta_sam3/EDITS.diff`: the unified diff between "upstream
                  with the rule applied" and the tree, every file, sorted.
                  Generated here, never typed.

A build is: every listed file with the rule applied, then the hunks of
`EDITS.diff`. A hunk whose context no longer matches stops the build and
names the file; that is what moving to a newer upstream commit looks like, and
the hunk is then re-made by hand and declared again.

Files that are ours and not upstream's (`OURS` is their first line) are left
alone by a build and listed by `--declare`.

## Two directions, and which one is the source

An edit is made in the tree and then declared (`--declare`). A move to another
upstream commit changes `COMMIT`, rebuilds, and reads what no longer applies.
`bench/check_meta_sam3_copy.py` holds the two together: it regenerates the
declaration and compares it with the tracked one byte for byte.

`coderef/` is gitignored and nothing shipped imports from it. This script
reads its files as text; it runs none of them. Without `coderef/sam3` at
`COMMIT` it returns 2.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COPY = REPO / "meta_sam3"
TREE = COPY / "sam3"
UPSTREAM = REPO / "coderef" / "sam3"

PKG = "sam3"
#: The upstream commit the copy is made from (facebookresearch/sam3). Inherited:
#: it is the commit `coderef/sam3` stood at when the copy was first made.
COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
#: The first lines of every Python file in the copy that is upstream's.
HEADER = (
    f"# Copied from facebookresearch/sam3 at commit {COMMIT} into ComfyUI-h3-explorations.\n"
    "# Under the SAM License (meta_sam3/LICENSE), not this pack's licence. Differences from upstream:\n"
    "# this header, relative imports, and the edits recorded in meta_sam3/EDITS.diff.\n"
)
#: The first line of a file in the copy that is not upstream's.
OURS = "# Not Meta's file. Written for ComfyUI-h3-explorations; distributed with the copy under the SAM License.\n"
NEW_FILE = "new file, not upstream's: "
DATA_DIFFERS = "binary or data file differs: "
FROM = re.compile(rf"^(\s*)from {PKG}((?:\.\w+)*) import\b", re.M)


def upstream_state() -> str | None:
    """None when `coderef/sam3` is at `COMMIT` with a clean tree; otherwise why not."""
    if not (UPSTREAM / PKG).is_dir():
        return f"{UPSTREAM.relative_to(REPO)} is absent"
    try:
        head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        dirty = subprocess.run(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--", PKG, "LICENSE"],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git could not read {UPSTREAM.relative_to(REPO)}: {exc}"
    if head.stdout.strip() != COMMIT:
        return f"{UPSTREAM.relative_to(REPO)} is at {head.stdout.strip()[:12] or '?'}, the copy is made from {COMMIT[:12]}"
    if dirty.stdout.strip():
        return f"{UPSTREAM.relative_to(REPO)} has local changes under {PKG}/"
    return None


def relativise(text: str, rel: Path) -> str:
    """The rule on one file; `rel` is its path under the package, e.g. model/vitdet.py."""
    dots = "." * len(rel.parts)  # a file directly in the package is depth 1: one dot

    def sub(m):
        return f"{m.group(1)}from {dots}{m.group(2).lstrip('.')} import"

    return HEADER + FROM.sub(sub, text)


def tree_files(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def declaration(tree: Path = TREE) -> str:
    """The unified diff between upstream with the rule applied and the tree, every file, sorted."""
    out = []
    for rel in tree_files(tree):
        mine = (tree / rel).read_bytes()
        up = UPSTREAM / PKG / rel
        if not up.exists():
            out.append(f"{NEW_FILE}{rel.as_posix()}\n")
            continue
        if rel.suffix != ".py":
            if up.read_bytes() != mine:
                out.append(f"{DATA_DIFFERS}{rel.as_posix()}\n")
            continue
        want = relativise(up.read_text(), rel)
        got = mine.decode()
        if want != got:
            out.extend(difflib.unified_diff(want.splitlines(True), got.splitlines(True),
                                            f"upstream+rule/{rel.as_posix()}", f"copy/{rel.as_posix()}"))
    return "".join(out)


def file_list(tree: Path = TREE) -> str:
    """The upstream paths the tree holds, one a line."""
    return "".join(f"{rel.as_posix()}\n" for rel in tree_files(tree) if (UPSTREAM / PKG / rel).exists())


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_declaration(text: str) -> tuple[dict[str, list], list[str]]:
    """{path: [(first line, old count, [hunk lines])]} and the paths declared as ours."""
    edits: dict[str, list] = {}
    ours = []
    lines = text.splitlines(True)
    i, current = 0, None
    while i < len(lines):
        line = lines[i]
        if line.startswith(NEW_FILE):
            ours.append(line[len(NEW_FILE):].strip())
            i += 1
        elif line.startswith(DATA_DIFFERS):
            raise SystemExit(f"FAIL  the declaration holds a data file that differs from upstream: {line.strip()}")
        elif line.startswith("--- upstream+rule/"):
            current = line[len("--- upstream+rule/"):].strip()
            edits[current] = []
            i += 2  # and its `+++ copy/` line
        else:
            m = HUNK.match(line)
            if not m or current is None:
                raise SystemExit(f"FAIL  cannot read line {i + 1} of the declaration: {line!r}")
            old = int(m.group(2)) if m.group(2) is not None else 1
            new = int(m.group(4)) if m.group(4) is not None else 1
            body, seen_old, seen_new = [], 0, 0
            i += 1
            while seen_old < old or seen_new < new:
                body.append(lines[i])
                seen_old += lines[i][0] in " -"
                seen_new += lines[i][0] in " +"
                i += 1
            edits[current].append((int(m.group(1)), old, body))
    return edits, ours


def apply_hunks(base: list[str], hunks: list, rel: str) -> list[str]:
    out, at = [], 0
    for first, old, body in hunks:
        start = first - 1 if old else first  # a pure insertion names the line it follows
        out += base[at:start]
        at = start
        for line in body:
            tag, text = line[0], line[1:]
            if tag in " -":
                if at >= len(base) or base[at] != text:
                    raise SystemExit(f"FAIL  {rel}: a declared edit no longer applies near upstream line {at + 1 - 3}. "
                                     f"Upstream moved; re-make the edit in the tree and run --declare.")
                at += 1
            if tag in " +":
                out.append(text)
    return out + base[at:]


def build() -> int:
    edits, ours = parse_declaration((COPY / "EDITS.diff").read_text())
    listed = [Path(p) for p in (COPY / "FILES.txt").read_text().split()]
    for rel in listed:
        up = UPSTREAM / PKG / rel
        dst = TREE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.suffix != ".py":
            dst.write_bytes(up.read_bytes())
            continue
        text = relativise(up.read_text(), rel).splitlines(True)
        dst.write_text("".join(apply_hunks(text, edits.pop(rel.as_posix(), []), rel.as_posix())))
    if edits:
        raise SystemExit(f"FAIL  the declaration edits files FILES.txt does not list: {sorted(edits)}")
    missing = [p for p in ours if not (TREE / p).is_file()]
    if missing:
        raise SystemExit(f"FAIL  files of ours the declaration names are not in the tree: {missing}")
    (COPY / "LICENSE").write_bytes((UPSTREAM / "LICENSE").read_bytes())
    print(f"  ok   wrote {len(listed)} files of upstream's under {TREE.relative_to(REPO)} and its LICENSE; "
          f"{len(ours)} of ours left as they are")
    return 0


def declare() -> int:
    (COPY / "EDITS.diff").write_text(declaration())
    (COPY / "FILES.txt").write_text(file_list())
    print(f"  ok   wrote {(COPY / 'EDITS.diff').relative_to(REPO)} and {(COPY / 'FILES.txt').relative_to(REPO)} from the tree")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--declare", action="store_true", help="write EDITS.diff and FILES.txt from the tree as it stands")
    args = ap.parse_args()
    why = upstream_state()
    if why:
        print(f"  SKIP  {why}")
        return 2
    return declare() if args.declare else build()


if __name__ == "__main__":
    sys.exit(main())
