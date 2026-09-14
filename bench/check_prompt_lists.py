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
3. **In order cycles, random draws, and both are a function of the seed.**
4. **Filling advances a list only when a text uses it**, and the same name
   twice in one text takes one value. A name with no list is refused by name.
5. **Wildcard files.** A `.txt` skips blank and `#` lines; a `.json` holds a
   list or named lists; a placeholder looks up `name.txt`, `name.json`, then
   `parent.json`'s list named for the last segment.
6. **The node.** A typed list chains; a duplicate name and a bad name raise.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_prompt_lists.py

Imports `comfy_api` for the node, so the ComfyUI checkout two directories up
must be importable; the script adds it to `sys.path` itself. No model, no
CUDA, no server.
"""

from __future__ import annotations

import json
import os
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
        try:
            pl.clean_name(bad)
            _fail(problems, f"names: {bad!r} was accepted as a list name")
        except ValueError:
            pass


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
        _fail(problems, "shuffled: the same seed gave a different sequence")
    other = pl.ListSequence(pl.PromptList("x", values, "shuffled", 8, "typed"))
    if [other.index(n) for n in range(k * 6)] == drawn:
        _fail(problems, "shuffled: a different seed gave the same sequence")
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
        _fail(problems, "random: the same seed gave different draws")
    if not any(draws[i] == draws[i + 1] for i in range(len(draws) - 1)):
        _fail(problems, "random: forty draws of three values never repeated back to back; is it shuffling?")
    try:
        pl.ListSequence(pl.PromptList("x", ("a",), "sideways", 0, "typed"))
        _fail(problems, "orders: an unknown order was accepted")
    except ValueError:
        pass


def check_resolve(problems):
    lists = {"x": pl.PromptList("x", ("p", "q"), "in_order", 0, "typed")}
    filled, picks = pl.resolve_texts(["a __x__", "b", "c __x__ and __x__"], lists)
    if filled != ["a p", "b", "c q and q"]:
        _fail(problems, f"resolve: filled {filled}")
    if picks != [{"x": "p"}, {}, {"x": "q"}]:
        _fail(problems, f"resolve: picks {picks}")
    try:
        pl.resolve_texts(["a __x__ __y__"], lists)
        _fail(problems, "resolve: a placeholder with no list was accepted")
    except ValueError as exc:
        if "__y__" not in str(exc):
            _fail(problems, f"resolve: the refusal did not name the placeholder ({exc})")
    if pl.resolve_texts(["plain text"], {})[0] != ["plain text"]:
        _fail(problems, "resolve: a text without placeholders changed")
    tricky = {"x": pl.PromptList("x", (r"a\1 b",), "in_order", 0, "typed")}
    if pl.resolve_texts(["__x__"], tricky)[0] != [r"a\1 b"]:
        _fail(problems, "resolve: a value with a backslash was rewritten")


def check_files(problems):
    with tempfile.TemporaryDirectory() as d:
        def put(rel, text):
            path = os.path.join(d, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")

        put("verbs.txt", "spinning\n\n# a comment\n  leaping  \n")
        put("dance/moves.json", json.dumps(["pop", "lock"]))
        put("scene.json", json.dumps({"outfits": ["grey vest", "red jacket"], "rooms": ["studio"]}))

        def find(rel):
            path = os.path.join(d, rel)
            return path if os.path.isfile(path) else None

        got = {n: pl.wildcard_list(n, find) for n in ("verbs", "dance/moves", "scene/outfits")}
        if got["verbs"].values != ("spinning", "leaping"):
            _fail(problems, f"files: verbs.txt read as {got['verbs'].values}")
        if got["dance/moves"].values != ("pop", "lock"):
            _fail(problems, f"files: dance/moves.json read as {got['dance/moves'].values}")
        if got["scene/outfits"].values != ("grey vest", "red jacket") or got["scene/outfits"].source != "scene.json":
            _fail(problems, f"files: scene/outfits read as {got['scene/outfits']}")
        try:
            pl.wildcard_list("scene/shoes", find)
            _fail(problems, "files: a missing named list was accepted")
        except ValueError:
            pass
        try:
            pl.wildcard_list("nothing", find)
            _fail(problems, "files: a placeholder with no file was accepted")
        except ValueError as exc:
            if "nothing.txt" not in str(exc):
                _fail(problems, f"files: the refusal did not say where it looked ({exc})")
        lists = pl.lists_for(["__verbs__ and __x__"], (pl.PromptList("x", ("q",), "in_order", 0, "typed"),), find)
        if set(lists) != {"verbs", "x"} or lists["x"].source != "typed":
            _fail(problems, f"files: lists_for gave {lists}")


def check_node(problems):
    first = pl.MiniMaxH3PromptList.execute("__subject__", {"source": "typed", "values": "detective\nman"},
                                           "shuffled", 3)
    chain = getattr(first, "args", first)[0]
    second = pl.MiniMaxH3PromptList.execute("verb", {"source": "typed", "values": "spinning"}, "in_order", 0,
                                            lists=chain)
    chain = getattr(second, "args", second)[0]
    if [(p.name, p.values, p.order) for p in chain] != [("subject", ("detective", "man"), "shuffled"),
                                                        ("verb", ("spinning",), "in_order")]:
        _fail(problems, f"node: the chain is {chain}")
    for label, call in (
            ("a duplicate name", lambda: pl.MiniMaxH3PromptList.execute(
                "verb", {"source": "typed", "values": "x"}, "shuffled", 0, lists=chain)),
            ("a bad name", lambda: pl.MiniMaxH3PromptList.execute(
                "two words", {"source": "typed", "values": "x"}, "shuffled", 0)),
            ("an empty list", lambda: pl.MiniMaxH3PromptList.execute(
                "empty", {"source": "typed", "values": "\n# nothing\n"}, "shuffled", 0))):
        try:
            call()
            _fail(problems, f"node: {label} was accepted")
        except ValueError:
            pass

    # The file source, with the folder lookup pointed at a temp dir: the real
    # `folder_paths` lookup is observed live, not depended on here.
    real_find = pl.find_wildcard
    with tempfile.TemporaryDirectory() as d:
        Path(d, "outfits.txt").write_text("grey vest\n# no\n\nred jacket\n", encoding="utf-8")
        Path(d, "scene.json").write_text(json.dumps({"outfit": ["track top"], "room": ["studio"]}),
                                         encoding="utf-8")
        pl.find_wildcard = lambda rel: str(Path(d, rel)) if Path(d, rel).is_file() else None
        try:
            txt = getattr(pl.MiniMaxH3PromptList.execute(
                "outfit", {"source": "file", "wildcard": "outfits.txt"}, "in_order", 0), "args", None)
            txt = txt[0][0] if txt else None
            if txt is None or (txt.values, txt.source) != (("grey vest", "red jacket"), "outfits.txt"):
                _fail(problems, f"node: the .txt file source gave {txt}")
            js = pl.MiniMaxH3PromptList.execute("outfit", {"source": "file", "wildcard": "scene.json"},
                                                "in_order", 0)
            js = getattr(js, "args", js)[0][0]
            if js.values != ("track top",):
                _fail(problems, f"node: the .json file source did not pick the list named for the node ({js})")
            for label, source in (("a missing file", {"source": "file", "wildcard": "absent.txt"}),
                                  ("no file chosen", {"source": "file"}),
                                  ("a JSON with no list of the node's name",
                                   {"source": "file", "wildcard": "scene.json"})):
                try:
                    pl.MiniMaxH3PromptList.execute("shoes" if "JSON" in label else "outfit", source,
                                                   "in_order", 0)
                    _fail(problems, f"node: {label} was accepted")
                except ValueError:
                    pass
        finally:
            pl.find_wildcard = real_find


def main() -> int:
    problems: list[str] = []
    check_names(problems)
    check_shuffled(problems)
    check_orders(problems)
    check_resolve(problems)
    check_files(problems)
    check_node(problems)
    if problems:
        print(f"  FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("  ok    names; shuffled uses a list up before repeating and its rejection rules bite; "
          "in_order and random follow the seed; filling advances per use; wildcard files; the node")
    return 0


if __name__ == "__main__":
    sys.exit(main())
