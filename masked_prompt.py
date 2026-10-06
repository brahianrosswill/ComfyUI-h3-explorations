"""The node that writes the masked video-to-video prompt.

`masked_prompt_text.py` holds every sentence and says where each came from;
this is the node on top of it. Wire `prompt` into the song node's `prompt`
and the Masked Source's `source` into this node: the text then names the
motion reference exactly when the Masked Source builds one, and describes
what it replaces, so the two cannot disagree (until 2026-10-06 turning the
motion reference on with an older text left `<Video 1>` unnamed, and nothing
said so).

**What the `source` wire costs.** The song node's `preview` asks for no
loader and no model, but it does ask for its prompt, and this node asks for
the Masked Source. So with `source` wired, a preview tracks the subject the
first time (the mask is kept, and the render that follows tracks nothing).
Reading the two settings from the queued graph instead would avoid that and
was not done: core computes a node's cache fingerprint without the graph
(`execution.py`, the `fingerprint_inputs` call passes no prompt), so a
changed setting on the Masked Source would have left this node's old text in
the cache.

The text is shown on the node on every run. To change one sentence for
every render, edit the constant in `masked_prompt_text.py`; to add to one
render, use `extra`; to write the whole text by hand, type it into the song
node's `prompt` as before and leave this node out.
"""
from __future__ import annotations

import logging

from comfy_api.latest import io, ui

from . import masked_prompt_text as text
from .video_mask import H3MaskedSource

logger = logging.getLogger(__name__)


class MiniMaxH3MaskedPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaskedPrompt",
            display_name="MiniMax H3 Masked Prompt (video to video)",
            category="model/latent/minimax",
            description=(
                "Writes the prompt for a masked video-to-video render from a few choices, so the long "
                "reference-format text is never typed. Wire `prompt` into the song node's `prompt` and the "
                "Masked Source's `source` into this node. The text is shown here on every run."),
            inputs=[
                io.Combo.Input("subject", options=list(text.SUBJECTS), default=text.SUBJECT_PERSON,
                               tooltip="Who the reference still shows. Sets the noun and the pronouns."),
                io.Combo.Input("voice", options=list(text.VOICES), default=text.VOICE_MAIN,
                               tooltip=("`the main voice on the track`: the subject sings or speaks it, lips "
                                        "in time.\n\n`silent`: the subject's lips stay closed.")),
                io.Combo.Input("picture_gives", options=list(text.GIVES), default=text.GIVES_FOLLOW,
                               tooltip=("What the reference still provides.\n\n"
                                        "`what the Masked Source replaces`: read from the wired `source`.\n\n"
                                        "`the whole person`: face, hair, build and clothing.\n\n"
                                        "`the head and hair`: the head only, on the original's body and "
                                        "clothes.")),
                io.String.Input("extra", multiline=True, default="",
                                tooltip=("Sentences added to the shot as written, for example what the "
                                         "subject does. Call the subject <Subject 1>.")),
                H3MaskedSource.Input("source", optional=True,
                                     tooltip=("The Masked Source's `source` output. The prompt then names "
                                              "the motion reference when it is on, and what is replaced.")),
            ],
            outputs=[io.String.Output(display_name="prompt", tooltip="The text, for the song node's `prompt`.")],
        )

    @classmethod
    def execute(cls, subject=text.SUBJECT_PERSON, voice=text.VOICE_MAIN, picture_gives=text.GIVES_FOLLOW,
                extra="", source=None) -> io.NodeOutput:
        replace = source.get("replace") if source is not None else None
        motion = source.get("motion_reference") if source is not None else None
        prompt = text.assemble(subject, voice, picture_gives, extra, replace, motion)
        logger.info("[h3] MiniMaxH3MaskedPrompt: %s, %s, the still gives %s, %s", subject, voice,
                    text.resolve_gives(picture_gives, replace),
                    "movement from <Video 1>" if motion not in (None, text.MOTION_NONE) else "no motion reference")
        return io.NodeOutput(prompt, ui=ui.PreviewText(prompt))
