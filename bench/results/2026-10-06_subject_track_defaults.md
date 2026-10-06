# The Subject Track and the part node at their defaults on the lane's three windows (2026-10-06)

lane: masked
verdict: at defaults one window was masked on a microphone and one misses a shot; the first is fixed by a rule, the second by a correction; the part was found on every frame the subject is a person

**Result: the defaults did not generalise as they stood. On one of three
windows the automatic pick took a hanging microphone for the subject and
masked it for a whole shot; a rule added the same day fixes that at defaults
and moves nothing on the other two. On another window the subject's first
shot is missed and no threshold can find it; a typed correction does. The
Sapiens2 part node at its own defaults found hair and face on every frame
where the subject is a person and not mid-dissolve.** Nothing was sampled.
Every figure is in `2026-10-06_subject_track_defaults.json`, one entry per
queued prompt.

## What ran

The three windows of `bench/turn_metric_eye_verdicts.json` (`band_112s`,
`solo_0s`, `friday_0s`), loaded as the shipped masked graphs load a source.
Per window one prompt: the loader, the SAM 3 checkpoint
(`h3_config.SEGMENTER`), `MiniMaxH3SubjectTrack` at
`h3_config.SUBJECT_TRACK`, `MiniMaxH3SaveShotTable` on its table and tiles,
and `MiniMaxH3Sapiens2Loader` with `MiniMaxH3SubjectParts` at the part
node's own defaults off the tracker's mask. From the second pass on, the
mask was also saved over the frames as a small video. The graphs were
hand-written for this pass, before the review and parts graphs of the same
day existed, and are not kept; the JSON carries every input of the tracker,
the loader and the part node. The runner was `bench/run_graph_arms.py`.

Three passes on a server started from `start.sh` in default mode, alone on
the card, each after one short prompt that left both models resident; no
DiT, encoder or VAE was loaded:

1. the three windows at defaults;
2. the same with the mask video, plus two remedies at inputs the node
   already has: the one-person window with `pick_on` a named frame, and the
   car window with `corrections`;
3. the three windows at defaults again after the rule below, and the car
   window's correction again.

## What a right answer is

The real cuts and the shots the subject is in are each window's `tracking`
block in `bench/turn_metric_eye_verdicts.json`, read by this session on
consecutive frames. Two things in that reading matter here. The band
window has a cut the tracker does not find, between two angles of the same
room. The car window opens on a dissolve, not a cut, from another shot of
the subject.

Each tracker shot was then read on its tile, the part node's preview and a
sheet of the mask over the frames. These are stills: they say where the
mask is on the frames looked at, and nothing about its edge or how it holds
between them.

## The tracker at defaults, before the rule

| window | cuts | shots read right | what was wrong |
|---|---|---|---|
| `band_112s` | four of five found | four of five; the fifth not known | the missed cut cost nothing: the mask stays on the subject across it |
| `solo_0s` | both found | two of three | shot 1 is masked on a hanging microphone for all its frames; the subject, who walks in partway through, is unmasked there |
| `friday_0s` | all six found | five of seven; one not known | shot 1 is left empty with the subject on screen throughout |

The band window's last shot is a close-up below the shoulders, and one shot
of the car window has a face half hidden at the frame's edge; this reader
could not say whether the subject is in either, and both are left empty.

**The microphone.** With nothing named, each shot's favourite votes with
its shot's length (`subject_track.py::main_subject`). The one-person
window's opening shot is its longest and shows only the microphone on the
frame where it is judged, and the subject's two other shots are framed too
differently to vote together. So the microphone won. The run recorded on
2026-10-04 covered seven shots of the same clip and the vote went the other
way, which is why this was not seen then. A correction cannot repair it:
the tile of that shot shows only the microphone to name. Naming a frame in
another shot does: with `pick_on` a named frame the subject is taken on all
three shots, in shot 1 from where they enter.

**The missed shot.** The subject's first shot on the car window scores
0.76 at best, and so does another person's close-up in the next shot. No
value of `match` takes the one without the other. This is the standing
example of what the signature cannot separate, and the evidence behind the
masking board's identity-model card.

## The rule, and what it moved

A shot's favourite on whom no head is found does not vote while another
shot's has one (`main_subject`, `MASK_VERSION` 7). It is the rule a match
already follows. The report names a favourite left out this way.
`bench/check_subject_track.py` item 8 holds the window's shape and was red
on the code before the rule.

Third pass, at defaults:

- `solo_0s`: right on all three shots. The pick moves to the second shot;
  the opening shot is taken from where the subject enters and is empty
  before that; the microphone is unmasked.
- `band_112s` and `friday_0s`: the mask video's decoded frames are
  identical to the second pass's. The rule fired once on the band window,
  on the headless close-up, and changed nothing.

## The correction

`corrections` = `shot 1: person 4` on the car window, the first run of a
correction on a real clip. The shot is seeded on the tile's frame, which
lies inside the opening dissolve, from the subject as the outgoing image
shows them. On the sheet the mask is on the subject in the incoming shot
from early in the dissolve to the shot's last frame. It gives the same
frames before and after the rule. The window's `tracking.subject_track`
carries the line, so a benchmark render of this window uses it.

## The part node

At its defaults, hair with face and neck, no matting model:

- found on every frame the subject is in, on all three windows, once the
  subject is a person: the band window, the car window at defaults, and the
  one-person window with the right pick;
- with the car window's correction, four frames inside or just after the
  dissolve had no part and took the nearest found frame's;
- on the one-person window before the rule, the frames without a part were
  the microphone's, which is not a part failure.

Seconds, for the node alone with its model resident and no DiT on the card:
0.15 s per frame the subject is in, the same on every prompt of all three
passes. The tracker's seconds are beside it in the JSON. What a part pass
adds to a render, with a DiT resident, is not measured here.

## What this does not say

- Nothing about the part mask's edge, the margins' fit, or the mask between
  the frames looked at.
- Nothing about a render. No prompt sampled.
- One window per clip. The microphone case was invisible on a longer span
  of the same clip, so a window is not its clip.
- The cut the tracker misses on the band window was harmless on this
  window; a missed cut into a shot without the subject would not be.
- The two shots marked not known are the owner's to call.
