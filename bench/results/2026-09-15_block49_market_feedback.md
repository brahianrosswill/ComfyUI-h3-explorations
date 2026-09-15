# Block-49 probe renders, market scene: viewer feedback, verbatim

2026-09-15. Three renders of the shipped text-to-video graph's prompt (the
covered-market scene), seed 730451892, 345 frames at 1344x768, 16 steps, on
the served build (sage fork v0.7.19, kitchen 0.2.34+sol.2aff3c5). Watched as
the original files, in this order:

- clip 1 = `h3_t2v_00019-audio.mp4` (shipped: INT8 attention everywhere)
- clip 2 = `h3_probe_t2v_balanced_00001-audio.mp4` (channel-balance node + sage `fp8++ balanced`)
- clip 3 = `h3_probe_t2v_exact_tail_00001-audio.mp4` (blocks 45, 48, 49 on bf16 attention)

The blind singles made for this comparison were not scored; every note below
is about the originals in the order above.

## Outside viewer, first pass

> general - woman changes identity in all the videos
> vid1. the guy morphs when he turns around
> vid2. is the only one where the man ends up carrying two crates but it kinda morphs in
> vid3. best quality perhaps

## Outside viewer, second pass (watching in the same order, on a phone)

> I don't know which is which order wise on mobile but the one where the
> dude morphs backwards is the worst. The last one seemed the best, no
> morphing in the crate, and the way he shifts the weight and carries the
> crate is more natural, including resting it on the edge of the table
> instead of holding it with one hand when he puts the coins in the can.
> Also maybe less motion blur in that clip too, but it's hard to say

## Owner, who has rendered this scene many times

> actually he moves out of the way in clip 3 of the market scene too. ive
> never seen that in this scene before. in #1 with int8 convrot across the
> board, he always morphs. in clip 2 he goes straight. in 3 with bf16 on
> the sensitive layers, he moves AROUND ppl. and the audio seems better too.
>
> in clip 1 he and the crate morph into something else. ive seen that often
> in this scene in prior renders. the guy and the crate like morph into a
> different direction entirely - it happens often with this scene.
>
> in clip 2 he doesnt... its way better... but clip 3 is even better. and i
> dont see anything WRONG with it - its much better than clip 1. but clip 3
> is way better in subtle ways.
>
> clip 3 is louder audio, sounds crisper... and someone said this and i
> noticed it too for the first time in this prompt/scene: "I've never seen
> him prop the crate against the table before either in that prompt. He
> always used his superhuman strength to one hand it like a beast." - he
> actually sets the crate down on the table and then lifts it up. coupled
> with moving around the people walking toward him its just a better scene.

Interpretation and caveats: `docs/h3_block49_quant_error.md`, section 6.
