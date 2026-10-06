"""How much of a tracked subject a part mask covers, frame by frame, and which frames to doubt.

The Sapiens2 part node (`sapiens2_parts.py`) calls a frame "found" when one
pixel of a taken class lies on the subject. On a clip where the part model's
labels sat off the tracked subject for a stretch, that was true of every frame
while the parts covered next to nothing of the subject, and the original stayed
in the render there. The node's report could not say so, because it counted
frames and not pixels.

This module is the count. Two masks in, figures out: `coverage` gives the
per-frame shares and `summarise` turns them into the report's lines and the
frames to doubt. It imports torch and nothing else, on purpose: the part
node's report calls it, and so can whatever holds both masks later (the Masked
Source takes the tracker's mask on `mask` and the part mask on `parts`), so
two places cannot print two different numbers for one clip. Nothing here
changes a mask.

Three figures per frame the subject is in:

- `covered`: the share of the subject's own mask that the parts cover.
- `outside`: the share of the parts that lies off the subject's own mask. The
  part node cuts its parts to the subject's mask widened by a margin, so what
  is left of a neighbour's labels sits in that margin: a part that is mostly
  outside is most likely someone else's.
- `labelled`, when the caller has it: the share of the subject the part model
  labelled as anything but background. Only the part node knows it. It is
  what tells a small part chosen on purpose (the head alone covers little of
  a whole person) from a person the model did not see.

And three reasons to doubt a frame, none of which needs the others:

- nothing: the parts cover no pixel of the subject.
- mostly outside: `outside` is above `OUTSIDE_MOST`.
- low: `covered` is under `LOW_OF_MEDIAN` of the clip's own median. Against
  the median and not a fixed share, because what a choice of parts should
  cover depends on the choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

#: A frame is low when the parts cover less of the subject than this share of
#: the clip's own median. **Measured, one clip and one choice of parts**
#: (`bench/results/2026-10-06_part_coverage.md`): on the clip this was written
#: for, the frames this marks are the stretch where the part lay off the
#: subject, and the record has the count at other shares. Not measured on a
#: choice of the head alone.
LOW_OF_MEDIAN = 0.25

#: A frame's parts are mostly outside when more than this share of them lies
#: off the subject's own mask. **Reasoned**: half is what "mostly" means.
#: **Seen** on the same clip, same record: it marks the same stretch, and the
#: frames that hold stay well under it.
OUTSIDE_MOST = 0.5


#: The key under which the Masked Source's record carries `Summary.warning()`
#: for the part mask wired into it: a line of text, or None when no frame is
#: in doubt, when the node was not wired a part mask, and when its mask was
#: kept from an earlier run so no part mask arrived to count. One name for
#: the node that writes it and the two that show it.
RECORD_KEY = "part_warning"


def record_line(source) -> str | None:
    """What a node holding the Masked Source's record shows about the part mask: one line, or None.

    The song node's report and the prompt node's summary both call this, so the two read alike.
    """
    warned = source.get(RECORD_KEY) if isinstance(source, dict) else None
    return f"the Masked Source warns: {warned}" if warned else None


#: Frames counted at a time. **Reasoned**: the count needs three bool masks of
#: the frames it is looking at, and a clip can be thousands of frames long;
#: this many keeps that to megabytes and changes no figure.
CHUNK = 64


def ranges(values: list[int], most: int = 12) -> str:
    """Frame numbers as runs: [3, 4, 5, 9] is `3-5, 9`. Past `most` runs the rest is counted."""
    runs: list[list[int]] = []
    for v in sorted(values):
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    text = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs[:most])
    return text + (f", and {len(runs) - most} more runs" if len(runs) > most else "")


@dataclass
class Coverage:
    present: torch.Tensor                    # [N] bool, the subject is on the frame
    covered: torch.Tensor                    # [N] float, parts on the subject over the subject; 0 where absent
    outside: torch.Tensor                    # [N] float, parts off the subject over the parts; 0 where no part
    labelled: torch.Tensor | None = None     # [N] float, the caller's share of the subject labelled a person


def _as_frames(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.ndim == 4 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"{name} must be [N, H, W]; got {tuple(mask.shape)}")
    return mask


def coverage(subject_mask: torch.Tensor, parts_mask: torch.Tensor, labelled: torch.Tensor | None = None) -> Coverage:
    """The per-frame figures for a subject mask and a part mask of the same frames, both [N, H, W]."""
    subject, parts = _as_frames(subject_mask, "subject_mask"), _as_frames(parts_mask, "parts_mask")
    if tuple(subject.shape) != tuple(parts.shape):
        raise ValueError(f"subject_mask is {tuple(subject.shape)} and parts_mask is {tuple(parts.shape)}: "
                         "coverage is of one mask by another on the same frames")
    n = int(subject.shape[0])
    of_subject, of_parts, both = (torch.zeros(n, dtype=torch.float64) for _ in range(3))
    # a chunk of frames at a time: three bool masks the size of a whole clip is gigabytes on a long one
    for i in range(0, n, CHUNK):
        on, part = subject[i:i + CHUNK] > 0.5, parts[i:i + CHUNK] > 0.5
        of_subject[i:i + CHUNK] = on.flatten(1).sum(dim=1)
        of_parts[i:i + CHUNK] = part.flatten(1).sum(dim=1)
        both[i:i + CHUNK] = (on & part).flatten(1).sum(dim=1)
    present = of_subject > 0
    covered = torch.where(present, both / of_subject.clamp(min=1.0), torch.zeros_like(both))
    outside = torch.where(of_parts > 0, (of_parts - both) / of_parts.clamp(min=1.0), torch.zeros_like(both))
    share = None
    if labelled is not None:
        share = torch.as_tensor(labelled, dtype=torch.float64).flatten()
        if int(share.numel()) != int(present.numel()):
            raise ValueError(f"labelled has {int(share.numel())} frames and the masks have {int(present.numel())}")
    return Coverage(present=present, covered=covered, outside=outside, labelled=share)


def _percent(share: float) -> str:
    """A share as a percentage, with a decimal under ten so that a sliver does not read as nothing."""
    value = 100.0 * float(share)
    return f"{value:.0f}%" if value >= 10.0 or value == 0.0 else f"{value:.1f}%"


@dataclass
class Summary:
    frames: int                              # frames in the clip
    present: int                             # frames the subject is in
    median: float                            # of `covered`, over the frames the subject is in
    lowest: float
    lowest_frame: int | None
    nothing: list[int] = field(default_factory=list)       # no part on the subject
    outside: list[int] = field(default_factory=list)       # most of the part off the subject
    low: list[int] = field(default_factory=list)           # some part, under LOW_OF_MEDIAN of the median
    labelled_median: float | None = None

    @property
    def suspect(self) -> list[int]:
        """Every frame with a reason to doubt it, once."""
        return sorted(set(self.nothing) | set(self.outside) | set(self.low))

    def lines(self) -> list[str]:
        """The report's lines. The first always states the figures; the rest name frames, or say none is named."""
        if not self.present:
            return ["coverage: the subject is on no frame, so there is nothing to cover"]
        out = [f"coverage: the taken parts cover a median of {_percent(self.median)} of the tracked subject's mask, "
               f"the lowest {_percent(self.lowest)} on frame {self.lowest_frame}"]
        if self.labelled_median is not None:
            out.append(f"the part model labelled a median of {_percent(self.labelled_median)} of the subject as "
                       "part of a person")
        if self.nothing:
            out.append(f"no part on the subject at all on {len(self.nothing)} frames: {ranges(self.nothing)}")
        if self.outside:
            out.append(f"most of the part lies off the subject's mask on {len(self.outside)} frames, where it is "
                       f"likely someone else's: {ranges(self.outside)}")
        if self.low:
            out.append(f"under {LOW_OF_MEDIAN:g} of the median on {len(self.low)} frames: {ranges(self.low)}")
        if not self.suspect:
            out.append(f"no frame is empty on the subject, mostly off them, or under {LOW_OF_MEDIAN:g} of the median")
        return out

    def warning(self) -> str | None:
        """One line for a node downstream to repeat, or None when no frame is in doubt."""
        if not self.suspect:
            return None
        return (f"the part mask is in doubt on {len(self.suspect)} of {self.present} frames the subject is in "
                f"({ranges(self.suspect)}): it covers a median of {_percent(self.median)} of the subject and "
                f"as little as {_percent(self.lowest)}; the original stays where the part does not reach")


def summarise(figures: Coverage) -> Summary:
    """The clip's figures and the frames to doubt, from `coverage`'s per-frame ones."""
    where = figures.present.nonzero().flatten()
    n = int(figures.present.numel())
    if not int(where.numel()):
        return Summary(frames=n, present=0, median=0.0, lowest=0.0, lowest_frame=None)
    covered = figures.covered[where]
    median = float(covered.median())
    at = int(covered.argmin())
    nothing = where[covered <= 0.0].tolist()
    low = where[(covered > 0.0) & (covered < LOW_OF_MEDIAN * median)].tolist()
    outside = where[figures.outside[where] > OUTSIDE_MOST].tolist()
    labelled = float(figures.labelled[where].median()) if figures.labelled is not None else None
    return Summary(frames=n, present=int(where.numel()), median=median, lowest=float(covered[at]),
                   lowest_frame=int(where[at]), nothing=nothing, outside=outside, low=low, labelled_median=labelled)
