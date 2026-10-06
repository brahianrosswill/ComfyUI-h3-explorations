"""The masked video-to-video prompt, assembled from a few choices.

H3's reference format is six sections of long prose, and in the masked lane
most of it never changes: the plate holds the setting, the framing and the
cuts, so the text names none of them and one prompt serves every window of
any clip (owner, 2026-10-04: the lane must not need a prompt written for a
shot). What does change is small: who the reference still shows, whether
they are the voice on the track, what of them the still provides, and
whether the original's movement is shown to the model as `<Video 1>`. Until
2026-10-06 each combination was a file somebody typed out, and turning the
motion reference on with the old text left `<Video 1>` unnamed.

`assemble` writes the text from those choices. Every sentence is a constant
below, so a wording change is one edit and every graph that wires the node
(`masked_prompt.py`) picks it up; `bench/check_masked_prompt.py` holds the
output to the bank's copies of the texts that have been rendered.

No torch and no ComfyUI: the node, the generator and `workflows/prompts.py`
(which resolves the node's text for the graders) all import this file.

Where the sentences come from. **Rendered**: the whole-person texts are
mrhf's generic swap prompts of 2026-10-04 (one window, one seed each, on the
band clip) and the `<Video 1>` lines are the ones the motion arms measured
(`bench/results/2026-10-05_masked_v2v_motion_arms.md`). **Not rendered**:
the silent variant, and every head-and-hair text, which generalises
`prompt_bank/ref2va_masked_head_swap.txt` (rendered on the band clip) the
way the whole-person text generalised its own first version: no count of
people, no duration, no camera claim.
"""
from __future__ import annotations

import re

# ---- the choices -----------------------------------------------------------

SUBJECT_PERSON = "a person"
SUBJECT_MAN = "a man"
SUBJECT_WOMAN = "a woman"
#: choice -> (noun, possessive, object pronoun)
SUBJECTS: dict[str, tuple[str, str, str]] = {
    SUBJECT_PERSON: ("person", "their", "them"),
    SUBJECT_MAN: ("man", "his", "him"),
    SUBJECT_WOMAN: ("woman", "her", "her"),
}

VOICE_MAIN = "the main voice on the track"
VOICE_SILENT = "silent"
VOICES = (VOICE_MAIN, VOICE_SILENT)

GIVES_FOLLOW = "what the Masked Source replaces"
GIVES_WHOLE = "the whole person"
GIVES_HEAD = "the head and hair"
GIVES = (GIVES_FOLLOW, GIVES_WHOLE, GIVES_HEAD)

# The Masked Source's own values. COPIES, because `video_mask.py` imports
# torch and ComfyUI and this file must not; `bench/check_masked_prompt.py`
# pins each against the original.
REPLACE_WHOLE = "whole subject"
REPLACE_PART = "head and hair"
REPLACE_PARTS = "the wired parts"
MOTION_NONE = "none"

# ---- the sentences ---------------------------------------------------------
# `{noun}`, `{poss}` and `{obj}` are filled from SUBJECTS; `{motion}` and
# `{voice}` from the clauses below.

MOTION_CLAUSE = (", whose body motion, posture, gestures, head movements and their timing come from the "
                 "person in <Video 1>")
VIDEO_DEFINITION = ("<Video 1> is the source of the movement transferred to <Subject 1>; its scene is not "
                    "reused and its person's appearance is not.")
VIDEO_RETENTION = ("<Video 1> (motion source): attribute_transfer - only the body motion, posture, gestures "
                   "and their timing are taken; the scene and the person's appearance are not.")
MOVES = ("<Subject 1> moves exactly as the person in <Video 1> moves, turning when they turn and by as "
         "much, facing where they face, gesturing when they gesture, at the same moments.")

SUMMARY_VOICE = {VOICE_MAIN: "and performs the main voice heard on the track to its timing",
                 VOICE_SILENT: "without speaking or singing"}

LIP_SYNC = ("<Subject 1> (S1) is the main voice on the track and performs it on screen: the jaw drops and "
            "the lips open on the first syllable of every sung or spoken phrase, the mouth shapes each "
            "vowel and closes on each consonant in time with the voice, and the lips rest together, still, "
            "whenever the voice pauses.")
LIPS_STILL = ("<Subject 1> does not speak or sing at any point: the lips rest together, relaxed and still, "
              "from the first frame to the last, and none of the voices on the track belongs to "
              "<Subject 1>.")
EYES = "The eyes blink naturally and hold a steady line of sight"
CAMERA = "The framing and every movement of the camera are the scene's own and do not change."
SOUNDSCAPE = ("overall_soundscape: The quiet ambience of the scene sits underneath, with the soft brush of "
              "clothing as the performer moves.")
MUSIC = "non_diegetic_music: N/A"

#: What the still provides -> that role's text. `shot` is the paragraph before
#: the performance sentences, `performance` the sentences per voice choice,
#: `close` what ends the shot.
ROLES: dict[str, dict] = {
    GIVES_WHOLE: dict(
        definition=("<Subject 1> is the {noun} shown in <Picture 1>, preserving {poss} facial identity, hair, "
                    "build and the clothing visible in <Picture 1>{motion}. The background, lighting and "
                    "framing of <Picture 1> are not present in the target video."),
        summary=("[reference generation] <Subject 1> takes the place of one person in a scene that is "
                 "already lit, framed and cut, {voice}, while everything else in the scene stays as it is."),
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - retain the same face, hair, build "
                   "and clothing in every frame, at every distance from the camera and from every side; "
                   "only the setting changes."),
        scene=("The target video is photorealistic live-action, and its setting, its lighting, its framing "
               "and every other person and object in it stay exactly as they already are from the first "
               "frame to the last."),
        place=("[Shot 1] <Subject 1> is in the scene for the whole take, in the place the scene holds for "
               "one person, at that place's distance from the lens and at the scale of everything around "
               "it."),
        shot=(
            "The scene's own light falls on <Subject 1> exactly as it falls on what is beside {obj}: the "
            "same direction, the same softness and the same colour on the face, the hair and the clothing, "
            "and whenever the light changes colour or brightness, the light on <Subject 1> changes with it "
            "at the same moment.",
            "The shadow <Subject 1> casts lies where that light sends it and moves when <Subject 1> moves.",
            "Where the frame shows the whole figure, <Subject 1> is whole, in the clothing of <Picture 1>, "
            "and whatever <Picture 1> does not show of {poss} clothing is plain and in keeping with it; "
            "where the frame shows only the head and shoulders, the face of <Subject 1> fills that space at "
            "the same scale, sharp and evenly exposed, with the skin texture, the hairline and the eyes of "
            "<Picture 1>.",
            "Seen from the side or from behind, <Subject 1> keeps the same hair, the same build and the "
            "same clothing.",
        ),
        performance={
            VOICE_MAIN: (
                LIP_SYNC,
                "A breath lifts the chest and shoulders before each new phrase.",
                EYES + ", and the brows and cheeks carry the feeling of the line.",
                "The head tips and turns with the delivery, the shoulders loosen and move with the rhythm, "
                "and the hands move with the phrasing, opening on a long note and settling as it ends.",
            ),
            VOICE_SILENT: (
                LIPS_STILL,
                "The breathing is slow and even, lifting the chest a little.",
                EYES + ", and the face stays attentive to what is happening in the scene.",
                "The head, the shoulders and the hands move a little and often, the small shifts of a "
                "person at ease, in time with the rhythm of the scene.",
            ),
        },
        close=("Everyone and everything else is untouched: wherever <Subject 1> does not cover them, the "
               "other people, the walls, the furniture and the ground are visible exactly as before, steady "
               "and in focus."),
    ),
    GIVES_HEAD: dict(
        definition=("<Subject 1> is the {noun} shown in <Picture 1>, preserving {poss} facial identity, "
                    "{poss} hair and anything worn on the head in <Picture 1>{motion}. The clothing, "
                    "background, lighting and framing of <Picture 1> are not present in the target video."),
        summary=("[reference generation] The head of <Subject 1> takes the place of one person's head in a "
                 "scene that is already lit, framed and cut, on that person's own body and in that "
                 "person's own clothes, {voice}, while everything else in the scene stays as it is."),
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - retain the same face, hair and "
                   "headwear in every frame, at every distance from the camera and from every side; the "
                   "body, the clothing and the setting are the scene's own."),
        scene=("The target video is photorealistic live-action, and its setting, its lighting, its framing, "
               "every other person and object in it, and the body, the clothing and the movement of the "
               "one person whose head is replaced stay exactly as they already are from the first frame to "
               "the last. Only that person's head changes: it is the head of <Subject 1>."),
        place=("[Shot 1] The head of <Subject 1> sits on that person's neck for the whole take, at the same "
               "size and in the same place, and meets the collar of the clothing with no gap and no second "
               "neckline."),
        shot=(
            "The head always faces the way the body beneath it faces: when the body turns, the head turns "
            "with it at the same moment and by the same amount, and when the body has its back to the "
            "camera the camera sees only the back of the head of <Subject 1> and whatever is worn on it, "
            "never the face.",
            "The hair of <Subject 1> is the only hair: none of the original hair remains on the shoulders, "
            "the chest or the back, where the clothing, its print and its folds carry on unbroken.",
            "The scene's own light falls on the head exactly as it falls on the body below it and on what "
            "is beside it: the same direction, the same softness and the same colour, and whenever the "
            "light changes colour or brightness, the light on the face changes with it at the same moment.",
            "Where the frame is close, the face is sharp and evenly exposed, with the skin texture, the "
            "hairline and the eyes of <Picture 1>.",
        ),
        performance={
            VOICE_MAIN: (
                LIP_SYNC,
                EYES + ", and the head tips and nods with the delivery while the body carries on moving "
                "exactly as before.",
            ),
            VOICE_SILENT: (
                LIPS_STILL,
                EYES + ", and the head moves only as the body beneath it moves.",
            ),
        },
        close=("Everyone and everything else is untouched: the other people, the walls, the furniture and "
               "the ground are visible exactly as before, steady and in focus."),
    ),
}


def resolve_gives(picture_gives: str, replace: str | None) -> str:
    """What the still provides: the choice itself, or read off the Masked Source's `replace`."""
    if picture_gives in ROLES:
        return picture_gives
    if picture_gives != GIVES_FOLLOW:
        raise ValueError(f"unknown picture_gives {picture_gives!r}; one of {list(GIVES)}")
    if replace is None or replace == REPLACE_WHOLE:
        return GIVES_WHOLE
    if replace == REPLACE_PART:
        return GIVES_HEAD
    raise ValueError(
        f"the Masked Source replaces `{replace}`, which can be any part of the subject: set `picture_gives` "
        f"to what <Picture 1> provides, one of {list(ROLES)}")


def assemble(subject: str = SUBJECT_PERSON, voice: str = VOICE_MAIN, picture_gives: str = GIVES_FOLLOW,
             extra: str = "", replace: str | None = None, motion_reference: str | None = None) -> str:
    """The prompt for one masked render.

    `replace` and `motion_reference` are the Masked Source's own values, or
    None when no source is wired (the whole person, no `<Video 1>`). `extra`
    is added to the shot as written, after the performance sentences.
    """
    if subject not in SUBJECTS:
        raise ValueError(f"unknown subject {subject!r}; one of {list(SUBJECTS)}")
    if voice not in VOICES:
        raise ValueError(f"unknown voice {voice!r}; one of {list(VOICES)}")
    role = ROLES[resolve_gives(picture_gives, replace)]
    moving = motion_reference not in (None, MOTION_NONE)
    noun, poss, obj = SUBJECTS[subject]
    fill = dict(noun=noun, poss=poss, obj=obj, motion=MOTION_CLAUSE if moving else "",
                voice=SUMMARY_VOICE[voice])
    shot = [role["place"]] + ([MOVES] if moving else []) + list(role["shot"]) + list(role["performance"][voice])
    added = re.sub(r"\s+", " ", extra or "").strip()
    if added:
        shot.append(added)
    shot += [role["close"], CAMERA]
    sections = [
        "subject_definitions:\n" + "\n".join([role["definition"]] + ([VIDEO_DEFINITION] if moving else [])),
        "summary:\n" + role["summary"],
        "retention_analysis:\n" + "\n".join([role["retention"]] + ([VIDEO_RETENTION] if moving else [])),
        "detailed_description:\n" + role["scene"] + "\n" + " ".join(shot),
        SOUNDSCAPE,
        MUSIC,
    ]
    return "\n\n".join(sections).format(**fill)
