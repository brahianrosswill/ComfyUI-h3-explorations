# A picture inside bars lost its cuts: the Subject Track's cut score on a padded source (2026-10-06)

lane: masked
verdict: a source with flat bars at its edges had no cut found at defaults; the score now leaves flat borders out and finds the bare picture's cuts; unbordered sources score exactly as before

**Result: on a 4:3 picture padded with black at each side to the wide
canvas, the tracker found no cut in a stretch that has eleven, and carried
one mask across all of them. The bars' edges are the same in every frame,
so scored with the picture they pull every cut's score down, the clip's
scores lose their gap, and the automatic threshold falls back to its fixed
value, above all of them. With the borders left out of the score the same
file gives the same eleven cuts as the unpadded original, at the same
frames.** Found by mrsun on an offline copy of the score; confirmed here on
the node, then fixed.

## What was measured

Two files of one clip, the same stretch of each: `thrill_2160.mkv`, a 4:3
picture, loaded at 1024x768; and `thrill_2160_pad1344.mkv`, the same
picture scaled to 1024x768 and centred on a 1344x768 frame with black at
each side, loaded at 1344x768. From 56.0 s, 1065 frames at 24 fps.

On the server, through the shipped review graph, nothing sampled, the
tracker at `MASK_VERSION` 7:

| file | cut threshold | steps at or above it | highest step below | cuts found |
|---|---|---|---|---|
| original | automatic, 0.81 | eleven, 1.01 down to 0.87 | 0.75 | eleven |
| padded | automatic, 0.90 | none | 0.86 | none: one shot, the mask on 658 of 1065 frames |
| padded, threshold named at 0.63 | the value named | eleven, 0.86 down to 0.72 | 0.55 | the original's eleven, frame for frame |

The cuts are at frames 501, 544, 611, 626, 705, 723, 757, 780, 842, 940
and 1038 of the stretch. The named value is the middle of the only gap in
the padded file's own steps; it is what the day's render of the padded
file used.

Off the server, on the CPU, the same stretch of each file decoded by
ffmpeg at the loader's width and scored twice, by the formula as it was and
by `subject_track.cut_scores` as it is now:

| file | formula | cut threshold | steps at or above it | highest below | cuts |
|---|---|---|---|---|---|
| padded | as it was | automatic, 0.90 | none | 0.86 | none |
| padded | borders left out | automatic, 0.81 | eleven, 1.01 down to 0.88 | 0.75 | the eleven |
| original | as it was | automatic, 0.81 | eleven, 1.02 down to 0.87 | 0.75 | the eleven |
| original | borders left out | automatic, 0.81 | the same eleven steps | 0.75 | the eleven |

The first row reproduces the node's own printed line for the padded file,
which is what says the offline decode stands in for the loader's. On the
padded file the border step found 22 columns at each side at the score's
size, about 154 pixels of the 160 each bar is wide; the one column beside
each bar, which the reduction mixes with the picture, is dropped as well.
On the original it found none and every step is unchanged.

## The change

`subject_track.py::flat_borders` and `cut_scores`. A row or column at a
frame's edge is border when all of it, in every frame of the loaded
stretch, lies within `BORDER_FLAT` of one value. The gradient maps are
taken inside what is left. A strip that is flat in one shot and not in the
next is picture; so is a still background, which is not one flat value.
The report says what was left out, in the frames' pixels. `MASK_VERSION` 8.
`bench/check_subject_track.py` item 9 holds it, and was red on the code
before: bars at the sides, top and bottom, and all round, each with the
cut found where the bare picture's is; a clip without bars scoring exactly
what the old formula gives; a strip flat in one shot only; bars with
coding noise.

## What it does not say

- Nothing about bars that are not flat: a blurred fill, a logo in the bar,
  a bar that changes value during the stretch. Those are scored as picture,
  as before.
- Nothing about what else the bars do downstream. The detector, the mask
  and the render still see the whole padded frame; only the cut score
  looks inside the bars.
- Not rerun on the server at `MASK_VERSION` 8: the server-side rows are the
  old code with and without a named threshold, and the new code's rows are
  the offline ones. The day's renders ran on the old code with the
  threshold named.
- One clip. The check's stand-ins cover the shapes; no second real
  bordered source has been tried.
