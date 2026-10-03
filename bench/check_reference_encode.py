#!/usr/bin/env python3
"""Hold the split reference path's pieces that need no encoder.

`reference_encode.py` encodes an H3 reference list apart from the prompt and
continues the prompt on kept keys and values, with a copy of core's language
model loop. The real comparison with core's one pass needs the encoder and is
`bench/measure_reference_split.py`. This file holds what can drift without
one:

  the copied loop's source   `_lm_pass` copies `Llama2_.forward`'s layer loop,
    is pinned                and relies on `Attention.forward`'s list path for
                             kept keys and on `TransformerBlock.forward`
                             writing into its input. Their source is hashed
                             against `PINNED`. A change in core fails here,
                             and the fix is to re-read the three, re-run the
                             server measurement, and move the pin: never to
                             move the pin alone.
  positions                  `_positions` against core's own
                             `qwen2vl_mrope_position_ids` on a layout with a
                             vision block in the first span and one in the
                             second, cut where the spans are cut, and a plain
                             count when no vision block precedes.
  the causal mask            `_causal_rows` is the new tokens' rows of the
                             square mask core builds, for a pass from the
                             start and for continued ones.
  the chained key            the same entries give the same key; a changed
                             pixel, a changed token and a changed parent each
                             give another.
  the span cut               one span per item that carries vision tokens; a
                             label-only item joins the one after it; the cut
                             reads the list alone.
  the store                  least recently used out first once what is kept
                             beyond the chain in use passes the byte budget,
                             the chain in use held whatever it weighs, a host
                             short of memory keeps nothing, and a collected
                             encoder empties it. RED CONTROL: with the budget
                             raised the same sequence evicts nothing.
  the report's labels        `reference_report.price_references` labels each
                             record with its own label when a sounded video
                             (two labels) comes first.
  the two nodes are          both are in the pack's node list, experimental,
    registered               and the prompt node takes the encode node's type.
  the one-node switch        `MiniMaxH3ReferenceConditioning.keep_references`
                             is the last input, optional and off by default;
                             on, the node routes through `encode_references`
                             and `condition_prompt` and still attaches the
                             reference latents; an encoder that cannot be
                             split falls through to the one pass and says so.
                             RED CONTROL: off, neither function is called.

    PYTHONPATH=<ComfyUI root> python bench/check_reference_encode.py
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: sha256 of the source of the core functions `_lm_pass` copies or leans on,
#: read at ComfyUI e9027f2b on 2026-10-03, the day the copy was held to core's
#: output on the shipped encoder (bench/results/2026-10-03_encoder_prefix_reuse.json).
PINNED = {
    "Llama2_.forward": "164b0ad348cb0544",
    "Attention.forward": "e05700e116d6a4ce",
    "TransformerBlock.forward": "71b138c2606ad2a3",
}


def load_pack():
    import comfy.cli_args as cli_args
    import torch
    if not torch.cuda.is_available():
        cli_args.args.cpu = True
    spec = importlib.util.spec_from_file_location(
        "h3x", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["h3x"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("h3x.reference_encode")


def main() -> int:
    import torch
    enc = load_pack()
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("the copied loop's source:")
    import comfy.text_encoders.llama as llama
    for name, pinned in PINNED.items():
        cls, fn = name.split(".")
        got = hashlib.sha256(inspect.getsource(getattr(getattr(llama, cls), fn)).encode()).hexdigest()[:16]
        check(f"{name} is the source the copy was made from", got == pinned,
              "" if got == pinned else f"now {got}, pinned {pinned}: re-read it, re-run "
              "bench/measure_reference_split.py, then move the pin")

    print("\npositions:")
    from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids
    grid_a, grid_b = torch.tensor([[1, 8, 12]]), torch.tensor([[1, 6, 4]])
    images = [{"type": "image", "index": 5, "size": 24, "extra": {"grid": grid_a}},
              {"type": "image", "index": 37, "size": 6, "extra": {"grid": grid_b}}]
    total, cut_a, cut_b = 60, 30, 44                   # span one, span two, then the prompt
    whole = qwen2vl_mrope_position_ids(images, total, "cpu")
    first = enc._positions(images[:1], 0, cut_a, "cpu")
    second = enc._positions(images, cut_a, cut_b - cut_a, "cpu")
    prompt = enc._positions(images, cut_b, total - cut_b, "cpu")
    check("three cuts of the sequence carry core's positions",
          torch.equal(torch.cat([first, second, prompt], dim=1), whole),
          f"{tuple(whole.shape)}")
    plain = enc._positions([], 7, 5, "cpu")
    check("no vision block ahead: a plain count from the start offset",
          torch.equal(plain, torch.arange(7, 12).unsqueeze(0)))

    print("\nthe causal mask:")
    for seq, past in ((7, 0), (5, 11), (1, 9)):
        total = past + seq
        square = torch.empty(total, total).fill_(torch.finfo(torch.float32).min / 4).triu_(1)
        check(f"{seq} new tokens on {past} kept: the rows of core's square mask",
              torch.equal(enc._causal_rows(seq, past, torch.float32, "cpu"), square[past:]))

    print("\nthe chained key:")
    pixels = torch.rand(1, 8, 8, 3)
    entries = [(11, 1.0), (12, 1.0), ({"type": "image", "data": pixels}, 1.0), (13, 1.0)]
    key = enc.span_key("root", entries)
    moved = pixels.clone()
    moved[0, 0, 0, 0] += 0.5
    check("the same entries give the same key",
          key == enc.span_key("root", [(11, 1.0), (12, 1.0), ({"type": "image", "data": pixels.clone()}, 1.0), (13, 1.0)]))
    check("a changed pixel, a changed token and a changed parent each give another",
          len({key,
               enc.span_key("root", entries[:2] + [({"type": "image", "data": moved}, 1.0)] + entries[3:]),
               enc.span_key("root", [(11, 1.0), (99, 1.0)] + entries[2:]),
               enc.span_key("other", entries)}) == 4)

    print("\nthe span cut:")
    kinds = lambda *k: [{"type": t} for t in k]        # noqa: E731
    check("a still, a sounded video, a still: three spans, the soundtrack's label with its video",
          enc.span_groups(kinds("image", "audio", "video", "image")) == [[0], [1, 2], [3]])
    check("trailing audio references form a last span of their own",
          enc.span_groups(kinds("image", "audio", "audio")) == [[0], [1, 2]])

    print("\nthe store:")

    def span(key, megabytes):
        rows = torch.zeros(1, 1, megabytes * 2 ** 20 // 4)
        return enc.Span(key=key, length=1, rows=rows, keys=[], values=[], images=[])

    plenty = lambda: 1 << 62                           # noqa: E731

    class Encoder:                                     # weakly referenceable, as a module is
        pass

    def run(budget_mb, available=plenty):
        store = enc._Store(budget_mb * 2 ** 20)
        store.claim(owner, 0)
        for key in ("a", "b"):
            store.put(span(key, 4), {key}, available=available)
        store.get("a")                                 # a is now the more recent
        store.put(span("c", 4), {"c"}, available=available)
        return store

    owner = Encoder()
    store = run(6)                                     # a and b are 8 MB beyond chain c
    check("past the budget beyond the chain in use, the least recently used entry goes",
          list(store.spans) == ["a", "c"], f"{list(store.spans)}")
    check("RED CONTROL: with room for all three nothing is evicted",
          sorted(run(100).spans) == ["a", "b", "c"])
    tight = enc._Store(0)
    tight.claim(owner, 0)
    tight.put(span("x", 4), {"x", "y"}, available=plenty)
    tight.put(span("y", 4), {"x", "y"}, available=plenty)
    check("the chain in use is held whatever it weighs, even with no budget beyond it",
          sorted(tight.spans) == ["x", "y"])
    import gc
    mortal = Encoder()
    dying = enc._Store(100 * 2 ** 20)
    dying.claim(mortal, 0)
    dying.put(span("m", 1), {"m"}, available=plenty)
    del mortal
    gc.collect()
    check("a collected encoder empties the store", not dying.spans and dying.owner is None)
    check("a host short of memory keeps nothing",
          not run(100, available=lambda: 0).spans)
    other = Encoder()
    store.claim(other, 0)
    check("a different encoder empties the store", not store.spans)
    store.put(span("z", 1), {"z"}, available=plenty)
    store.claim(other, 1)
    check("new patches on the encoder empty it too", not store.spans)

    print("\nthe report's labels:")
    rc_ = importlib.import_module("h3x.reference_conditioning")
    report = importlib.import_module("h3x.reference_report")
    sound = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
    records = (rc_.RuntimeVideoReference(frames=torch.rand(24, 64, 96, 3), loaded_fps=24.0, soundtrack=sound),
               rc_.RuntimeImageReference(image=torch.rand(1, 64, 96, 3), size_policy="match"))
    priced = report.price_references(records, 96, 64, 5)
    got = [getattr(item, "label", None) for item in getattr(priced, "items", [])]
    check("a still after a sounded video is <Picture 1>, not the video's label",
          got == ["<Video 1>", "<Picture 1>"], f"{got}")

    print("\nthe two nodes:")
    nodes = importlib.import_module("h3x.nodes")
    source = inspect.getsource(nodes)
    check("both are in the pack's node list",
          "MiniMaxH3EncodeReferences, MiniMaxH3PromptOnReferences]" in source)
    a, b = enc.MiniMaxH3EncodeReferences.define_schema(), enc.MiniMaxH3PromptOnReferences.define_schema()
    check("both are marked experimental", bool(a.is_experimental and b.is_experimental))
    out_type = a.outputs[0].io_type if hasattr(a.outputs[0], "io_type") else getattr(a.outputs[0], "get_io_type", lambda: None)()
    in_types = [getattr(i, "io_type", None) or getattr(i, "get_io_type", lambda: None)() for i in b.inputs]
    check("the prompt node takes the encode node's output type",
          out_type is not None and out_type in in_types, f"{out_type}")

    print("\nthe one-node switch:")
    rc = importlib.import_module("h3x.reference_conditioning")
    schema = rc.MiniMaxH3ReferenceConditioning.define_schema()
    last = schema.inputs[-1]
    check("keep_references is the last input, optional, off by default",
          last.id == "keep_references" and last.optional and last.default is False,
          f"{last.id}, optional {last.optional}, default {last.default}")

    import types
    calls = []

    class Clip:                                        # what the node touches on a CLIP
        layer_idx = None
        use_clip_schedule = False
        patcher = types.SimpleNamespace(forced_hooks=None)

        def tokenize(self, text, **kw):
            calls.append("tokenize")
            return {"tokens": text}

        def encode_from_tokens_scheduled(self, tokens):
            calls.append("one_pass")
            return [[torch.zeros(1, 2, 4), {"pooled_output": None}]]

    saved = enc.encode_references, enc.condition_prompt
    enc.encode_references = lambda clip, items, labels=None: (
        calls.append("encode_references"),
        types.SimpleNamespace(report=[f"{labels[0]}: encoded"], spans=[]))[1]
    enc.condition_prompt = lambda encoded, text: (
        calls.append("condition_prompt"), [[torch.ones(1, 2, 4), {"pooled_output": None}]])[1]

    class Vae:                                         # a video VAE as the compiler uses it
        def encode(self, pixels):
            return torch.zeros(1, 4, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)

    refs = rc.MiniMaxH3AppendRefImage.execute(torch.rand(1, 64, 96, 3), "match", "shared").result[0]

    def run(clip, **kw):
        # The node's report tokenizes too, to count; only the encode calls are read.
        calls.clear()
        return rc.MiniMaxH3ReferenceConditioning.execute(
            clip, refs, "a prompt", width=96, height=64, length=5, vae=Vae(), **kw)

    try:
        on = run(Clip(), keep_references=True)
        on_calls = list(calls)
        cond, extra = on.result[0][0]
        check("on: the node routes through encode_references and condition_prompt",
              on_calls[:2] == ["encode_references", "condition_prompt"]
              and "one_pass" not in on_calls and bool(cond.all()),
              f"{on_calls}")
        check("on: the reference latents are still attached",
              len(extra.get("minimax_refs", [])) == 1)
        text = on.ui.as_dict()["text"][0] if on.ui is not None else ""
        check("on: the preview says what was kept or encoded", "<Picture 1>: encoded" in text)
        run(Clip())
        split = {"encode_references", "condition_prompt"}
        check("RED CONTROL: off, neither function is called and the one pass runs",
              "one_pass" in calls and not split & set(calls), f"{calls}")
        skipped = Clip()
        skipped.layer_idx = -2
        out = run(skipped, keep_references=True)
        text = out.ui.as_dict()["text"][0] if out.ui is not None else ""
        check("an encoder that cannot be split falls through to the one pass and says so",
              "one_pass" in calls and not split & set(calls)
              and "references not kept" in text, f"{calls}")
    finally:
        enc.encode_references, enc.condition_prompt = saved

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
