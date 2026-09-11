#!/usr/bin/env bash
# Rebuild and install comfy_kitchen's Sol-Attn kernel from a source checkout.
#
# **This used to say "from kijai's branch", and that the checkout it built from
# was SOMEONE ELSE'S REPO.** Sol-Attn is upstream now and the default source
# moved to our own fork clone (see SRC below); the leave-it-as-you-found-it
# design did not move, because every checkout under coderef/ is shared with
# other agents and a tree left dirty blocks the next pull. That is the design:
#
#   2026-08-14: the version-tag edit was left as a working-tree modification
#   on the theory that a future `git pull` would then conflict loudly rather
#   than silently reverting it. It does not conflict -- it BLOCKS:
#
#       error: cannot pull with rebase: You have unstaged changes.
#
#   A guard that stops the owner updating a dependency is worse than the drift
#   it was guarding against. The edit is made by this script for the duration
#   of a build and reverted on every exit path.
#
# Why the edit exists at all: the source declares a plain version ("0.2.31",
# "0.2.32", ...), identical to the PyPI wheel ComfyUI pins, so a fork build
# and the stock wheel are indistinguishable to `pip list` -- and a stock wheel
# silently has no sol_attn, which makes every Sol call fall back to dense with
# no error. PEP 440 still matches `X.Y.Z+sol.<sha>` against `==X.Y.Z`, so a
# plain requirements install stays satisfied and will not clobber it -- but
# only while X.Y.Z is exactly ComfyUI's pin, which is why the base has to be
# the pinned tag (see "Carry blk_cnt, track everything else" below).
#
#   2026-09-03: this used to be `git apply vendor/patches/001-local-version-tag.patch`,
#   a diff hardcoded against `version = "0.2.31"`. Upstream released 0.2.32 on
#   2026-09-02 and the patch stopped applying on any checkout based on it; the
#   first build of our blk_cnt branch rebased onto v0.2.32 found that. The
#   version line is now rewritten by sed, whatever it says, so the script no
#   longer carries a copy of upstream's version number that has to be kept
#   in step with upstream.
#
# Usage:  vendor/rebuild_kernel.sh [CUDA_ARCH]     (default 89, this box's 4090)
#         vendor/rebuild_kernel.sh --check         (is the source current? builds nothing)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# --- Carry blk_cnt, track everything else (owner, 2026-09-11) ----------------
# We carry the `blk_cnt` commits indefinitely, whether or not upstream ever
# merges them (Comfy-Org/comfy-kitchen #168), and take everything else from
# upstream. "Current" means those commits rebased onto the tag ComfyUI's
# requirements pin: not upstream's newest tag, and not main. The reason is
# PEP 440. Our build declares `X.Y.Z+sol.<sha>`, which satisfies
# `comfy-kitchen==X.Y.Z` only when X.Y.Z IS the pin; built on a newer tag it
# stops matching, and the next requirements install silently swaps in the
# stock wheel, which has no `blk_cnt`. So a newer upstream tag is news, not an
# instruction: when ComfyUI moves its pin, rebase onto the new tag. The gate
# below refuses any source that is not `v<pin>` plus our commits, and
# `--check` runs only the gate.
#
# Derived from this checkout, not typed: the repo sits at
# <comfy>/custom_nodes/<pack>, so ComfyUI and its venv are two levels up.
# Override with COMFY=... / PY=... for any other layout.
COMFY="${COMFY:-$REPO/../..}"
PY="${PY:-$COMFY/.venv/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- set PY=/path/to/python"; exit 1; }
PIN="$(sed -n 's/^comfy-kitchen==\([0-9][0-9.]*\).*/\1/p' "$COMFY/requirements.txt" 2>/dev/null | head -1)"
[ -n "$PIN" ] || { echo "ERROR: no comfy-kitchen==X.Y.Z pin in $COMFY/requirements.txt"; exit 1; }
CLONE="$REPO/coderef/comfy-kitchen"

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then CHECK_ONLY=1; shift; fi
ARCH="${1:-89}"

# How to become current, printed wherever the gate refuses. The worktree goes
# somewhere durable, not a session scratchpad: the build record points at it
# and start.sh warns when a running build's source is gone.
pin_worktree() {   # the worktree holding sol-blk-cnt-<pin>, or nothing
    git -C "$CLONE" worktree list --porcelain 2>/dev/null |
        awk -v b="branch refs/heads/sol-blk-cnt-$PIN" '/^worktree /{w=substr($0,10)} $0==b{print w; exit}'
}
recipe() {
    echo
    if git -C "$CLONE" rev-parse -q --verify "refs/heads/sol-blk-cnt-$PIN" >/dev/null; then
        local wt; wt="$(pin_worktree)"
        if [ -n "$wt" ]; then
            echo "sol-blk-cnt-$PIN already carries our commits on v$PIN, in $wt;"
            echo "run without SRC to build from it."
        else
            echo "sol-blk-cnt-$PIN exists but no worktree has it:"
            echo "  git -C $CLONE worktree add <durable-dir>/ck_$PIN sol-blk-cnt-$PIN"
            echo "  git -C <durable-dir>/ck_$PIN submodule update --init --recursive"
        fi
        return
    fi
    local old
    old="$(git -C "$CLONE" for-each-ref --sort=-version:refname --format='%(refname:short)' \
           'refs/heads/sol-blk-cnt-[0-9]*' | head -1)"
    old="${old:-<last-carried-branch>}"
    echo "To carry the blk_cnt commits onto v$PIN (ComfyUI's pin):"
    echo "  git -C $CLONE fetch upstream --tags"
    echo "  git -C $CLONE cherry -v v$PIN $old    # '-' = already in v$PIN: skip it"
    echo "  git -C $CLONE worktree add <durable-dir>/ck_$PIN -b sol-blk-cnt-$PIN v$PIN"
    echo "  git -C <durable-dir>/ck_$PIN cherry-pick v${old#sol-blk-cnt-}..$old"
    echo "  git -C <durable-dir>/ck_$PIN submodule update --init --recursive"
    echo "  vendor/rebuild_kernel.sh --check && vendor/rebuild_kernel.sh"
}

# Default source: the worktree holding `sol-blk-cnt-<pin>`. It used to be the
# clone itself, and before that `coderef/comfy-kitchen-sol` (kijai's checkout,
# since renamed `comfy-kitchen-kijai`: a place to read, never to build from).
# The clone's own HEAD is whatever branch was last left there -- on 2026-09-11
# an old `sol-blk-cnt` based on no current tag -- while the build lives in a
# worktree, so the default follows the pin, not the HEAD. Overridable, e.g. to
# build one specific commit:
#
#   SRC=/path/to/worktree vendor/rebuild_kernel.sh 89
#
# Every checkout under coderef/ is shared, and this script's whole design is
# to leave the one it builds from exactly as it found it.
if [ -z "${SRC:-}" ]; then
    WANT="sol-blk-cnt-$PIN"
    SRC="$(git -C "$CLONE" worktree list --porcelain 2>/dev/null |
           awk -v b="branch refs/heads/$WANT" '/^worktree /{w=substr($0,10)} $0==b{print w; exit}')"
    if [ -z "$SRC" ]; then
        echo "REFUSED: no worktree of $CLONE has $WANT checked out, so nothing"
        echo "carries our commits on ComfyUI's pinned tag."
        recipe; exit 1
    fi
fi
[ -d "$SRC" ] || { echo "no checkout at $SRC"; exit 1; }
cd "$SRC"
echo "== source: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)  (ComfyUI pins comfy-kitchen==$PIN)"

# --- The currency gate -------------------------------------------------------
# Fetching tags moves refs only, never this tree. Offline, it judges against
# the tags already here and says so.
git fetch --quiet upstream --tags 2>/dev/null ||
    echo "== WARNING: could not fetch upstream; judging against the tags already here"
CURRENT=1
DECLARED="$(sed -n 's/^version = "\([^"+]*\).*/\1/p' pyproject.toml | head -1)"
if ! git rev-parse -q --verify "refs/tags/v$PIN" >/dev/null; then
    echo "REFUSED: tag v$PIN, ComfyUI's pin, is not in this checkout"; CURRENT=0
elif ! git merge-base --is-ancestor "v$PIN" HEAD; then
    echo "REFUSED: this source is not based on v$PIN, the comfy-kitchen ComfyUI pins"; CURRENT=0
fi
if [ "$DECLARED" != "$PIN" ]; then
    echo "REFUSED: the source declares $DECLARED but ComfyUI pins $PIN; $DECLARED+sol.<sha>"
    echo "would not satisfy the pin, and a requirements install would put the stock wheel back"
    CURRENT=0
fi
if [ "$CURRENT" = 1 ]; then
    echo "== carried on top of v$PIN (+ ours; - already in v$PIN, drop it on the next rebase):"
    while read -r mark sha subject; do
        if git merge-base --is-ancestor "$sha" upstream/main 2>/dev/null; then
            echo "     $mark ${sha:0:7} $subject   <-- upstream main, in no tag"
            CURRENT=0
        else
            echo "     $mark ${sha:0:7} $subject"
        fi
    done < <(git cherry -v "v$PIN" HEAD)
    [ "$CURRENT" = 1 ] ||
        echo "REFUSED: the source carries untagged upstream work; the base is a tag, never main"
fi
NEWEST="$(git tag -l 'v[0-9]*' --sort=-version:refname | head -1)"
[ "$NEWEST" = "v$PIN" ] ||
    echo "== upstream has $NEWEST, newer than the pin: news, not an instruction (see the header)"
echo "== upstream main past $NEWEST, untagged, not built by policy: $(git rev-list --count "$NEWEST"..upstream/main 2>/dev/null || echo unknown) commit(s)"
[ "$CURRENT" = 1 ] || { recipe; exit 1; }

if [ "$CHECK_ONLY" = 1 ]; then
    WOULD="$PIN+sol.$(git rev-parse --short=7 HEAD)"
    INSTALLED="$("$PY" -c 'import importlib.metadata as m; print(m.version("comfy_kitchen"))' 2>/dev/null || echo none)"
    if [ "$INSTALLED" = "$WOULD" ]; then
        echo "== --check: current, and the venv already holds this build ($INSTALLED)"
    else
        echo "== --check: current source; the venv holds $INSTALLED, a rebuild would install $WOULD"
    fi
    exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: $SRC has local changes. This script needs a clean tree so it"
    echo "can guarantee it leaves one. Resolve them first:"
    git status --short
    exit 1
fi

# Revert the edit no matter how we leave, including on a failed build. Without
# this a compile error strands the tree dirty and blocks the next pull, which
# is the exact failure this script exists to prevent.
cleanup() { git -C "$SRC" checkout -- pyproject.toml 2>/dev/null || true; }
trap cleanup EXIT

# Append the local segment to whatever version the checkout declares; the
# tag is the built commit's short sha, derived rather than typed, so it
# cannot go stale on an update.
SHA="$(git rev-parse --short=7 HEAD)"
sed -i "s/^version = \"\([0-9][^\"+]*\)\"/version = \"\1+sol.$SHA\"/" pyproject.toml
if ! grep -q "^version = \".*+sol.$SHA\"" pyproject.toml; then
    echo "ERROR: could not tag the version line in pyproject.toml:"
    grep -n '^version' pyproject.toml; exit 1
fi
VER="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
echo "== version: $VER"

# `--no-build-isolation` means the build uses an EXISTING environment rather
# than a fresh one, and uv picks that up from VIRTUAL_ENV. Neither checkout has
# a .venv of its own, so without this uv finds nothing, and the failure is
# reported as a MISSING BUILD DEPENDENCY ("No module named 'setuptools'")
# rather than as a missing environment -- which sends you off installing
# setuptools somewhere it already is. Derived from $PY so it follows the
# interpreter override rather than being a second place to configure the venv.
#
# `--python "$PY"` as well, since 2026-09-10: VIRTUAL_ENV alone loses to a
# `.python-version` in the source checkout. The 0.2.33 worktree pins 3.12, so
# once the ComfyUI venv moved to 3.14 `uv build` picked a managed 3.12 with no
# setuptools and failed with the same misleading "No module named
# 'setuptools'" described above. An explicit interpreter beats both.
export VIRTUAL_ENV="$(cd "$(dirname "$PY")/.." && pwd)"
echo "== building against $VIRTUAL_ENV"
COMFY_CUDA_ARCHS="$ARCH" uv build --wheel --no-build-isolation --python "$PY" .
# By exact version, not a glob: dist/ keeps every wheel ever built here, so
# `comfy_kitchen-*.whl` grew to match more than one the first time this script
# ran twice, and uv would have been handed both.
WHL=(dist/comfy_kitchen-"$VER"-*.whl)
[ -f "${WHL[0]}" ] || { echo "ERROR: no wheel built for $VER"; exit 1; }
uv pip install --python "$PY" --force-reinstall --no-deps "${WHL[0]}"

# The build record: ONE file beside the venv saying which build is installed
# and where its source is, written at the only moment both are known. start.sh
# prints it on every launch and cross-checks it against the installed wheel,
# so "which comfy-kitchen is running" has one answer and one path to it
# (owner's ask, 2026-09-03). A stock reinstall replaces the wheel but not this
# file, which is exactly the mismatch start.sh is there to shout about.
RECORD="$VIRTUAL_ENV/comfy_kitchen_build.json"
"$PY" - "$RECORD" "$VER" "$SRC" "${WHL[0]}" "$ARCH" <<'PYEOF'
import json, subprocess, sys, time
record, ver, src, whl, arch = sys.argv[1:6]
def git(*a):
    return subprocess.run(["git", "-C", src, *a], capture_output=True, text=True).stdout.strip()
json.dump({
    "version": ver,
    "sha": git("rev-parse", "HEAD"),
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "source": src,
    "origin": git("remote", "get-url", "origin"),
    "wheel": whl,
    "cuda_arch": arch,
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "written_by": "vendor/rebuild_kernel.sh",
}, open(record, "w"), indent=2)
print(f"== build record: {record}")
PYEOF

cleanup; trap - EXIT
echo "== checkout restored: $(git status --porcelain | wc -l) modified files (want 0)"

echo "== verifying the kernel is actually present"
"$PY" "$REPO/bench/check_sol_kernel.py" --require || {
    echo "check_sol_kernel FAILED -- the build installed but is not usable"; exit 1; }

echo
echo "Restart ComfyUI (this is node code), then confirm the reload by reading a"
echo "changed default back out of /object_info before trusting any measurement."
