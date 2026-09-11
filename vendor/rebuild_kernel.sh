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
CLONE="${CLONE:-$REPO/coderef/comfy-kitchen}"

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then CHECK_ONLY=1; shift; fi
ARCH="${1:-89}"

# --- The build branch: one name, always (owner, 2026-09-11) ------------------
# The owner's fork (the clone's `origin`) holds exactly:
#
#   main            a mirror of upstream main; never built
#   h3-build        ComfyUI's pinned tag plus our blk_cnt commits: what we build
#   sol-blk-cnt-pr  PR 168's head, our commits on upstream main; for the PR
#                   only, and deleted when the PR closes
#   archive/* tags  retired builds that records here cite by sha, kept
#                   reachable so those shas still resolve
#
# Until 2026-09-11 every pin got its own branch (sol-blk-cnt-0.2.32,
# sol-blk-cnt-0.2.33, ...) and the old ones piled up on the fork. Now a pin
# move rebases h3-build in place, after tagging its old tip
# archive/h3-build-<old version>.
#
# A build from h3-build installs as `<pin>+sol.<sha>`: the version names the
# upstream release and the sha names our commits, so
# `git -C coderef/comfy-kitchen log v<pin>..<sha>` is everything we add. That
# is why the base is a tag and never main: main declares the last release's
# version while holding later code, so the version would stop saying what is
# in the build.
#
# The rebase stays manual on purpose. The gate below refuses a build whose
# base is not the pin and prints the exact steps (recipe), so a forgotten
# rebase cannot produce a wrong build. An automatic one would rewrite a branch
# in a clone other agents share, and would make what is in the build depend
# on when it was last built.
BRANCH="h3-build"
branch_worktree() {   # the worktree that has $BRANCH checked out, or nothing
    git -C "$CLONE" worktree list --porcelain 2>/dev/null |
        awk -v b="branch refs/heads/$BRANCH" '/^worktree /{w=substr($0,10)} $0==b{print w; exit}'
}
# How to become current, printed wherever the script refuses. The clone is
# durable, not a session scratchpad: the build record points at it, and a
# record naming a source that has been deleted answers nothing.
recipe() {
    echo
    if ! git -C "$CLONE" rev-parse -q --verify "refs/heads/$BRANCH" >/dev/null; then
        echo "$CLONE has no $BRANCH; take it from the fork:"
        echo "  git -C $CLONE fetch origin && git -C $CLONE switch $BRANCH"
        return 0
    fi
    if [ -z "$(branch_worktree)" ]; then
        echo "$CLONE is not on $BRANCH; put it there:"
        echo "  git -C $CLONE switch $BRANCH"
        return 0
    fi
    local old
    old="$(git -C "$CLONE" show "$BRANCH:pyproject.toml" 2>/dev/null |
           sed -n 's/^version = "\([^"+]*\).*/\1/p' | head -1 || true)"
    # On the pin already: the refusal above says what else is wrong.
    if [ -z "$old" ] || [ "$old" = "$PIN" ]; then return 0; fi
    echo "ComfyUI now pins v$PIN and $BRANCH is built on v$old. Keep the old build"
    echo "reachable, carry our commits onto the new tag, build, and back it up:"
    echo "  git -C $CLONE fetch upstream --tags"
    echo "  git -C $CLONE tag -a archive/$BRANCH-$old $BRANCH -m \"$BRANCH on v$old, retired when ComfyUI pinned v$PIN\""
    echo "  git -C $CLONE rebase --onto v$PIN v$old $BRANCH    # commits upstream already has drop out"
    echo "  vendor/rebuild_kernel.sh --check && vendor/rebuild_kernel.sh"
    echo "  git -C $CLONE push --force-with-lease origin $BRANCH refs/tags/archive/$BRANCH-$old"
}

# Default source: the checkout that has $BRANCH, which is the fork clone
# itself -- one folder. The lookup follows the branch, not a path, so a clone
# left on some other branch (sol-blk-cnt-pr, for PR work) is refused rather
# than built. (Before 2026-09-11 the branch was sol-blk-cnt-<pin>; before
# 2026-09-08 the default was kijai's checkout, now coderef/comfy-kitchen-kijai:
# a place to read, never to build from.) Overridable, e.g. to build one
# specific commit:
#
#   SRC=/path/to/worktree vendor/rebuild_kernel.sh 89
#
# Every checkout under coderef/ is shared, and this script's whole design is
# to leave the one it builds from as it found it; the submodule step below is
# the one exception, and says why.
if [ -z "${SRC:-}" ]; then
    SRC="$(branch_worktree)"
    if [ -z "$SRC" ]; then
        echo "REFUSED: $CLONE is not on $BRANCH, the branch that carries our"
        echo "commits on ComfyUI's pinned tag."
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

# --- Submodules: the commits this source pins, on every build (owner, 2026-09-11)
# The CUDA build compiles against third_party/flash-attention and
# third_party/cutlass (comfy_kitchen/backends/cuda/CMakeLists.txt). A fresh
# clone or a new worktree -- which every pin move makes -- leaves them empty,
# and the build then dies deep in the compile with `flash.h: No such file or
# directory`, naming neither the cause nor the fix. That happened on
# 2026-09-01 and three postmortems carried the missing check. So the build
# brings every top-level submodule to the commit the checked-out source
# records, every time: when a rebase onto a new tag moves a pin, the next
# build follows it and nothing has to be remembered.
#
# "Latest" here means the source's own pins, never the submodules' upstream
# heads (`git submodule update --remote`). The tag was built and tested
# against these commits; a moved head would appear nowhere in the
# `+sol.<sha>` version, so two builds with one version could differ; and it
# would modify the gitlinks the clean-tree check below reads.
#
# Top-level only, as docs/SOLATTN.md's manual recipe does: the build reads
# nothing under flash-attention's own submodules (its csrc/cutlass, and
# csrc/composable_kernel, which is ROCm). This is the one change to the source
# checkout the script keeps, because a submodule off its pin is not a state
# anyone builds from.
SUBS="$(git submodule status)"
SUBS_AT_PIN=1
if grep -q '^[-+U]' <<<"$SUBS"; then SUBS_AT_PIN=0; fi

if [ "$CHECK_ONLY" = 1 ]; then
    if [ "$SUBS_AT_PIN" = 1 ]; then
        echo "== --check: submodules at this source's pins:"
    else
        echo "== --check: submodules NOT at this source's pins ('-' empty, '+' other commit); a build checks them out:"
    fi
    sed 's/^/     /' <<<"$SUBS"
    WOULD="$PIN+sol.$(git rev-parse --short=7 HEAD)"
    # -I (isolated): without it `-c` puts the current directory -- this source
    # checkout -- first on sys.path, and its build-left comfy_kitchen.egg-info
    # answered instead of the venv (2026-09-11: an empty venv read as the
    # clone's stale 0.2.31 build).
    INSTALLED="$("$PY" -I -c 'import importlib.metadata as m; print(m.version("comfy_kitchen"))' 2>/dev/null || echo none)"
    if [ "$INSTALLED" = "$WOULD" ]; then
        echo "== --check: current, and the venv already holds this build ($INSTALLED)"
    else
        echo "== --check: current source; the venv holds $INSTALLED, a rebuild would install $WOULD"
    fi
    exit 0
fi

# Before the clean-tree check: a submodule on another commit shows there as a
# modified gitlink, and would refuse the build as "local changes" when the fix
# is this checkout.
if [ "$SUBS_AT_PIN" = 0 ]; then
    echo "== submodules: checking out the commits $(git rev-parse --short HEAD) pins"
    git submodule update --init || {
        echo "REFUSED: could not check out the submodules this source pins (offline, or"
        echo "a submodule has local changes); the CUDA build cannot compile without them:"
        git submodule status; exit 1; }
    SUBS="$(git submodule status)"
    if grep -q '^[-+U]' <<<"$SUBS"; then
        echo "REFUSED: a submodule is still not at the commit this source pins:"
        echo "$SUBS"; exit 1
    fi
fi
echo "== submodules at this source's pins:"
sed 's/^/     /' <<<"$SUBS"

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
# and where its source is, written at the only moment both are known (owner's
# ask, 2026-09-03). It is read by a person and cited by docs/SOLATTN.md and
# docs/sol_upstream.md. A stock reinstall replaces the wheel but not this
# file, so a record whose version differs from the installed one means
# something reinstalled comfy-kitchen after this script ran. Until 2026-09-11
# start.sh printed and cross-checked it on every launch; it now prints the
# installed version and says to run this script when that is the stock wheel,
# because the node refuses an armed route observer on a wheel without blk_cnt
# (sol_attn_h3.py::_require_kernel) and --check covers the pin.
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
