#!/usr/bin/env python3
"""`step_x0_observer.py` saves each step's prediction and changes nothing.

CPU only, no server. Cases:
  returns_input     the post-CFG function returns the very tensor it was given
  per_step_files    three calls at falling sigma write steps 00, 01, 02, each
                    the video half of a packed AV prediction, unpacked by core's
                    own `unpack_latents`, with the sigma in the file's metadata
  new_render_stamp  a call at a sigma not below the last starts a new render
                    stamp, so a second render never overwrites the first
  audio_optional    `save_audio` adds the audio half, and off it is absent
  node_appends      the node installs through the patcher's own setter, which
                    appends, so a function already there still runs

    <comfy-venv-python> bench/check_step_x0_observer.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parents[1]))
sys.path.insert(1, str(REPO))

import safetensors  # noqa: E402
import torch  # noqa: E402

import comfy.utils  # noqa: E402
import step_x0_observer as so  # noqa: E402

results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<17} {detail}")


def main() -> int:
    video = torch.randn(1, 24, 3, 4, 6)
    audio = torch.randn(1, 32, 2, 5)
    packed, shapes = comfy.utils.pack_latents([video, audio])
    model = types.SimpleNamespace(latent_shapes=shapes)

    with tempfile.TemporaryDirectory() as tmp:
        fn = so.make_observer(tmp, "t", with_audio=False)
        outs = [fn({"denoised": packed, "sigma": torch.tensor([s]), "model": model})
                for s in (0.9, 0.5, 0.1)]
        case("returns_input", all(o is packed for o in outs), "same object back on every call")

        files = sorted(Path(tmp).glob("t_*_video.latent"))
        ok = len(files) == 3 and [f.name.rsplit("_", 2)[-2] for f in files] == ["00", "01", "02"]
        t0 = safetensors.torch.load_file(str(files[0]))["latent_tensor"] if files else None
        with safetensors.safe_open(str(files[1]), "pt") as f:
            meta = f.metadata()
        ok = ok and t0 is not None and torch.equal(t0, video) and meta.get("sigma") == "0.5"
        case("per_step_files", ok, f"{[f.name for f in files]}, step 01 sigma {meta.get('sigma')}")

        fn({"denoised": packed, "sigma": torch.tensor([0.9]), "model": model})
        stamps = {f.name.split("_")[1] + f.name.split("_")[2] for f in Path(tmp).glob("t_*_video.latent")}
        case("new_render_stamp", len(stamps) == 2 and len(list(Path(tmp).glob("t_*_video.latent"))) == 4,
             f"{len(stamps)} render stamps, no overwrite")

        fa = so.make_observer(tmp, "a", with_audio=True)
        fa({"denoised": packed, "sigma": torch.tensor([0.9]), "model": model})
        a_files = list(Path(tmp).glob("a_*_audio.latent"))
        a_ok = len(a_files) == 1 and torch.equal(
            safetensors.torch.load_file(str(a_files[0]))["latent_tensor"], audio)
        case("audio_optional", a_ok and not list(Path(tmp).glob("t_*_audio.latent")),
             "audio saved when asked, absent otherwise")

    class Patcher:
        def __init__(self):
            self.model_options = {"sampler_post_cfg_function": ["existing"]}

        def clone(self):
            c = Patcher()
            c.model_options = {k: list(v) for k, v in self.model_options.items()}
            return c

        def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
            self.model_options["sampler_post_cfg_function"] = \
                self.model_options.get("sampler_post_cfg_function", []) + [fn]

    import folder_paths
    src = Patcher()
    res = so.MiniMaxH3StepX0Observer.execute(src, filename_prefix="latents/x", save_audio=False)
    m = res.result[0] if hasattr(res, "result") else res[0]
    fns = m.model_options["sampler_post_cfg_function"]
    case("node_appends", m is not src and fns[0] == "existing" and len(fns) == 2
         and src.model_options["sampler_post_cfg_function"] == ["existing"],
         f"output dir under {Path(folder_paths.get_output_directory()).name}/latents")

    print(f"\n{sum(results)} of {len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
