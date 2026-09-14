#!/usr/bin/env python3
"""The prompt list contract, and the controls that show each rule can fail.

`prompt_lists.py` is the module; the owner's rules (2026-09-14) are in its
docstring. What this holds it to, each with a scripted input that would pass
a weaker implementation:

1. **Names.** `__dance_moves__`, `__dance/verbs__` and `__v2-beat__` are
   placeholders; `snake_case`, `_x_` and `__ spaced __` are not.
2. **Shuffled never repeats before the list is used up.** Every pass is a
   permutation; no pass repeats the one before; with three or more values the
   first of a pass is never the last of the previous. The two rejection rules
   are driven by a scripted shuffle that offers the forbidden pass first, so
   an implementation that accepts its first draw goes red here.
3. **In order cycles, random draws, and both are a function of `shuffle`.**
4. **Filling advances a list only when a text uses it**, and the same name
   twice in one text takes one value. A name with no list is refused by name,
   and so is a list a loop connects that no text uses.
5. **Wildcard files.** A `.txt` skips blank and `#` lines; a `.json` holds one
   list of strings, and a JSON object is refused.
6. **The node.** A typed list chains; a duplicate name and a bad name raise;
   the file source reads a `.txt` and a `.json`, refuses a missing file, and
   its fingerprint moves when the file is edited.
7. **Every loop node fills the same way.** A pack module importing
   `loop_output` or `loop_resume` defines loop nodes, and each must declare a
   `lists` input and call `fill_windows`. The song node must be detected as
   one, since a detector that finds nothing passes everything, and a copy of
   its source with either removed must fail.
8. **Fill Prompt Lists agrees with a loop**: index N gives what window N of
   a loop gets when every window uses the name.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_prompt_lists.py

Imports `comfy_api` for the node, so the ComfyUI checkout two directories up
must be importable; the script adds it to `sys.path` itself. No model, no
CUDA, no server.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import prompt_lists as pl  # noqa: E402


def _fail(problems, msg):
    problems.append(msg)


class ScriptedRng:
    """A `shuffle` that writes each scripted permutation in turn."""

    def __init__(self, perms):
        self.perms = list(perms)

    def shuffle(self, seq):
        seq[:] = self.perms.pop(0)


def _refused(problems, label, call, needle=None):
    try:
        call()
        _fail(problems, f"{label} was accepted")
    except ValueError as exc:
        if needle is not None and needle not in str(exc):
            _fail(problems, f"{label}: the refusal did not name {needle!r} ({exc})")


def check_names(problems):
    found = pl.placeholders("a __dance_moves__, __dance/verbs__ and __v2-beat__ then __dance_moves__ again")
    if found != ["dance_moves", "dance/verbs", "v2-beat"]:
        _fail(problems, f"names: found {found}")
    for text in ("snake_case", "_x_", "__ spaced __", "no placeholders here"):
        if pl.placeholders(text):
            _fail(problems, f"names: {text!r} read as a placeholder")
    if pl.clean_name("__verbs__") != "verbs" or pl.clean_name(" verbs ") != "verbs":
        _fail(problems, "names: clean_name did not strip the underscores or spaces")
    for bad in ("two words", "__", "a__b", ""):
        _refused(problems, f"names: {bad!r} as a list name", lambda bad=bad: pl.clean_name(bad))


def check_shuffled(problems):
    values = tuple("abcdefg")
    k = len(values)
    seq = pl.ListSequence(pl.PromptList("x", values, "shuffled", 7, "typed"))
    drawn = [seq.index(n) for n in range(k * 6)]
    passes = [drawn[i:i + k] for i in range(0, len(drawn), k)]
    for i, p in enumerate(passes):
        if sorted(p) != list(range(k)):
            _fail(problems, f"shuffled: pass {i} is not a permutation: {p}")
        if i and p == passes[i - 1]:
            _fail(problems, f"shuffled: pass {i} repeats the pass before")
        if i and values[p[0]] == values[passes[i - 1][-1]]:
            _fail(problems, f"shuffled: pass {i} starts with the value the pass before ended on")
    again = pl.ListSequence(pl.PromptList("x", values, "shuffled", 7, "typed"))
    if [again.index(n) for n in range(k * 6)] != drawn:
        _fail(problems, "shuffled: the same shuffle number gave a different sequence")
    other = pl.ListSequence(pl.PromptList("x", values, "shuffled", 8, "typed"))
    if [other.index(n) for n in range(k * 6)] == drawn:
        _fail(problems, "shuffled: a different shuffle number gave the same sequence")
    few = pl.ListSequence(pl.PromptList("x", values, "shuffled", 7, "typed"))
    if len({few.index(n) for n in range(5)}) != 5:
        _fail(problems, "shuffled: five uses of seven values repeated one")

    # Controls: the scripted shuffle offers the forbidden pass first.
    boundary = pl.next_pass(ScriptedRng([[2, 1, 0], [1, 2, 0]]), ("a", "b", "c"), [0, 1, 2])
    if boundary != [1, 2, 0]:
        _fail(problems, f"shuffled: a pass starting on the previous pass's last value was accepted ({boundary})")
    same = pl.next_pass(ScriptedRng([[0, 1, 2], [1, 0, 2]]), ("a", "b", "c"), [0, 1, 2])
    if same != [1, 0, 2]:
        _fail(problems, f"shuffled: a pass identical to the previous was accepted ({same})")
    stuck = pl.next_pass(ScriptedRng([[0, 1, 2]] * pl.PASS_ATTEMPTS), ("a", "b", "c"), [0, 1, 2])
    if stuck != [1, 2, 0]:
        _fail(problems, f"shuffled: the fallback after rejected draws was {stuck}, not the rotation")
    two = pl.next_pass(ScriptedRng([[0, 1], [1, 0]]), ("a", "b"), [0, 1])
    if two != [1, 0]:
        _fail(problems, f"shuffled: two values did not reverse on the next pass ({two})")
    one = pl.ListSequence(pl.PromptList("x", ("only",), "shuffled", 0, "typed"))
    if {one.value(n) for n in range(4)} != {"only"}:
        _fail(problems, "shuffled: a one-value list gave something else")


def check_orders(problems):
    ordered = pl.ListSequence(pl.PromptList("x", ("p", "q", "r"), "in_order", 3, "typed"))
    if [ordered.value(n) for n in range(7)] != ["p", "q", "r", "p", "q", "r", "p"]:
        _fail(problems, "in_order: did not cycle the list in order")
    a = pl.ListSequence(pl.PromptList("x", tuple("abc"), "random", 11, "typed"))
    b = pl.ListSequence(pl.PromptList("x", tuple("abc"), "random", 11, "typed"))
    draws = [a.index(n) for n in range(40)]
    if draws != [b.index(n) for n in range(40)]:
        _fail(problems, "random: the same shuffle number gave different draws")
    if not any(draws[i] == draws[i + 1] for i in range(len(draws) - 1)):
        _fail(problems, "random: forty draws of three values never repeated back to back; is it shuffling?")
    _refused(problems, "orders: an unknown order",
             lambda: pl.ListSequence(pl.PromptList("x", ("a",), "sideways", 0, "typed")))


def check_resolve(problems):
    lists = {"x": pl.PromptList("x", ("p", "q"), "in_order", 0, "typed")}
    filled, picks = pl.resolve_texts(["a __x__", "b", "c __x__ and __x__"], lists)
    if filled != ["a p", "b", "c q and q"]:
        _fail(problems, f"resolve: filled {filled}")
    if picks != [{"x": "p"}, {}, {"x": "q"}]:
        _fail(problems, f"resolve: picks {picks}")
    _refused(problems, "resolve: a placeholder with no list", lambda: pl.resolve_texts(["a __x__ __y__"], lists),
             "__y__")
    if pl.resolve_texts(["plain text"], {})[0] != ["plain text"]:
        _fail(problems, "resolve: a text without placeholders changed")
    tricky = {"x": pl.PromptList("x", (r"a\1 b",), "in_order", 0, "typed")}
    if pl.resolve_texts(["__x__"], tricky)[0] != [r"a\1 b"]:
        _fail(problems, "resolve: a value with a backslash was rewritten")

    # the loop route: a placeholder with no list, and a connected list no text uses
    chain = (lists["x"], pl.PromptList("place", ("studio",), "in_order", 0, "typed"))
    _refused(problems, "fill_windows: a placeholder with no list", lambda: pl.fill_windows(["__nope__"], ()),
             "__nope__")
    _refused(problems, "fill_windows: a connected list no text uses",
             lambda: pl.fill_windows(["a __x__", "b __x__"], chain), "'place'")
    texts, lines = pl.fill_windows(["a __x__ in the __place__", "b __x__ in the __place__"], chain)
    if texts != ["a p in the studio", "b q in the studio"] or len(lines) != 2:
        _fail(problems, f"fill_windows: filled {texts} with lines {lines}")


def check_files(problems):
    with tempfile.TemporaryDirectory() as d:
        txt = Path(d, "verbs.txt")
        txt.write_text("spinning\n\n# a comment\n  leaping  \n", encoding="utf-8")
        listed = Path(d, "moves.json")
        listed.write_text(json.dumps(["pop", "lock"]), encoding="utf-8")
        named = Path(d, "scene.json")
        named.write_text(json.dumps({"outfits": ["grey vest"]}), encoding="utf-8")
        empty = Path(d, "empty.txt")
        empty.write_text("# nothing\n", encoding="utf-8")
        if pl.values_from_file(str(txt)) != ("spinning", "leaping"):
            _fail(problems, f"files: verbs.txt read as {pl.values_from_file(str(txt))}")
        if pl.values_from_file(str(listed)) != ("pop", "lock"):
            _fail(problems, f"files: moves.json read as {pl.values_from_file(str(listed))}")
        _refused(problems, "files: a JSON object of named lists", lambda: pl.values_from_file(str(named)))
        _refused(problems, "files: a file with no values", lambda: pl.values_from_file(str(empty)))


def check_node(problems):
    first = pl.MiniMaxH3PromptList.execute("__subject__", {"source": "typed", "values": "detective\nman"},
                                           "shuffled", 3)
    chain = getattr(first, "args", first)[0]
    second = pl.MiniMaxH3PromptList.execute("verb", {"source": "typed", "values": "spinning"}, "in_order", 0,
                                            lists=chain)
    chain = getattr(second, "args", second)[0]
    if [(p.name, p.values, p.order, p.shuffle) for p in chain] != [
            ("subject", ("detective", "man"), "shuffled", 3), ("verb", ("spinning",), "in_order", 0)]:
        _fail(problems, f"node: the chain is {chain}")
    for label, call in (
            ("a duplicate name", lambda: pl.MiniMaxH3PromptList.execute(
                "verb", {"source": "typed", "values": "x"}, "shuffled", 0, lists=chain)),
            ("a bad name", lambda: pl.MiniMaxH3PromptList.execute(
                "two words", {"source": "typed", "values": "x"}, "shuffled", 0)),
            ("an empty list", lambda: pl.MiniMaxH3PromptList.execute(
                "empty", {"source": "typed", "values": "\n# nothing\n"}, "shuffled", 0))):
        _refused(problems, f"node: {label}", call)

    # The file source, with the folder lookup pointed at a temp dir: the real
    # `folder_paths` lookup is observed live, not depended on here.
    real_find = pl.find_wildcard
    with tempfile.TemporaryDirectory() as d:
        Path(d, "outfits.txt").write_text("grey vest\n# no\n\nred jacket\n", encoding="utf-8")
        Path(d, "rooms.json").write_text(json.dumps(["studio", "rooftop"]), encoding="utf-8")
        pl.find_wildcard = lambda rel: str(Path(d, rel)) if Path(d, rel).is_file() else None
        try:
            txt = getattr(pl.MiniMaxH3PromptList.execute(
                "outfit", {"source": "file", "wildcard": "outfits.txt"}, "in_order", 0), "args", None)
            txt = txt[0][0] if txt else None
            if txt is None or (txt.values, txt.source) != (("grey vest", "red jacket"), "outfits.txt"):
                _fail(problems, f"node: the .txt file source gave {txt}")
            js = pl.MiniMaxH3PromptList.execute("room", {"source": "file", "wildcard": "rooms.json"},
                                                "in_order", 0)
            js = getattr(js, "args", js)[0][0]
            if js.values != ("studio", "rooftop"):
                _fail(problems, f"node: the .json file source gave {js}")
            for label, source in (("a missing file", {"source": "file", "wildcard": "absent.txt"}),
                                  ("no file chosen", {"source": "file"})):
                _refused(problems, f"node: {label}",
                         lambda source=source: pl.MiniMaxH3PromptList.execute("outfit", source, "in_order", 0))
            source = {"source": "file", "wildcard": "outfits.txt"}
            before = pl.MiniMaxH3PromptList.fingerprint_inputs(source=source)
            Path(d, "outfits.txt").write_text("grey vest\nred jacket\ntrack top\n", encoding="utf-8")
            if not before or pl.MiniMaxH3PromptList.fingerprint_inputs(source=source) == before:
                _fail(problems, "node: editing a list node's file left its fingerprint unchanged")
        finally:
            pl.find_wildcard = real_find


LOOP_MODULES = ("loop_output", "loop_resume")


def loop_node_problems(source: str) -> tuple[list[str], list[str]]:
    """(loop node class names, problems) in one pack module's source."""
    tree = ast.parse(source)
    is_loop_module = any(
        isinstance(n, ast.ImportFrom) and n.level and (
            (n.module or "") in LOOP_MODULES
            or (n.module is None and any(a.name in LOOP_MODULES for a in n.names)))
        for n in ast.walk(tree))
    if not is_loop_module:
        return [], []
    found, problems = [], []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        if not any("ComfyNode" in ast.unparse(b) for b in cls.bases):
            continue
        found.append(cls.name)
        calls = [c for c in ast.walk(cls) if isinstance(c, ast.Call)]
        if not any(isinstance(c.func, ast.Attribute) and c.func.attr == "Input"
                   and ast.unparse(c.func.value) == "H3PromptLists"
                   and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "lists"
                   for c in calls):
            problems.append(f"{cls.name}: no `lists` input (H3PromptLists)")
        if not any(ast.unparse(c.func).split(".")[-1] == "fill_windows" for c in calls):
            problems.append(f"{cls.name}: never calls fill_windows")
    return found, problems


def check_loop_nodes(problems):
    found = []
    for path in sorted(REPO.glob("*.py")):
        if path.stem in LOOP_MODULES:
            continue
        names, bad = loop_node_problems(path.read_text(encoding="utf-8"))
        found += [f"{path.name}::{n}" for n in names]
        problems += [f"loop nodes: {path.name}::{b}" for b in bad]
    if "audio_freeze_song.py::MiniMaxH3AudioFreezeSong" not in found:
        _fail(problems, f"loop nodes: the song node was not detected as one (found {found})")
    song = (REPO / "audio_freeze_song.py").read_text(encoding="utf-8")
    for label, old, new in (("its lists input", 'H3PromptLists.Input("lists"', 'H3PromptLists.Input("listz"'),
                            ("its fill_windows call", "fill_windows(plan.uses", "resolve_texts(plan.uses")):
        if song.count(old) != 1:
            _fail(problems, f"loop nodes: the control for {label} lost its anchor {old!r}")
        elif not loop_node_problems(song.replace(old, new))[1]:
            _fail(problems, f"loop nodes: the song node with {label} removed still passed")
    return found


def check_fill(problems):
    lists = {"x": pl.PromptList("x", tuple("abcde"), "shuffled", 5, "typed"),
             "y": pl.PromptList("y", ("p", "q"), "in_order", 0, "typed")}
    texts = ["__x__ and __y__"] * 12
    looped, _ = pl.resolve_texts(texts, lists)
    single = [pl.fill_at(texts[0], lists, n)[0] for n in range(12)]
    if looped != single:
        _fail(problems, f"fill: index N differs from window N of a loop ({single[:4]} against {looped[:4]})")
    out = pl.MiniMaxH3FillPromptLists.execute("wear __y__", 2, lists=(lists["y"],))
    args = getattr(out, "args", out)
    if args[0] != "wear q" or not args[1].startswith("[2] "):
        _fail(problems, f"fill: index 2 gave {args}")
    plain = getattr(pl.MiniMaxH3FillPromptLists.execute("no names here", 1), "args", None)
    if plain is None or tuple(plain) != ("no names here", "no placeholders"):
        _fail(problems, f"fill: a prompt with no placeholders gave {plain}")
    _refused(problems, "fill: a placeholder with no list", lambda: pl.fill_at("__nope__", lists, 0), "__nope__")


def main() -> int:
    problems: list[str] = []
    check_names(problems)
    check_shuffled(problems)
    check_orders(problems)
    check_resolve(problems)
    check_files(problems)
    check_node(problems)
    loops = check_loop_nodes(problems)
    check_fill(problems)
    if problems:
        print(f"  FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("  ok    names; shuffled uses a list up before repeating and its rejection rules bite; "
          "in_order and random follow shuffle; filling advances per use and refuses a missing or unused "
          f"list; wildcard files; the node; {len(loops)} loop node(s) fill the same way and the controls "
          "bite; Fill Prompt Lists agrees with a loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
