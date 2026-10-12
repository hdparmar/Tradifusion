"""
Generate the example recordings shown on hdparmar.github.io/Tradifusion.

Every clip goes through the thesis pipeline unchanged (Sections 5.4 and 5.5):

    seed spectrogram --VAE--> latent --noise (denoising)--> img2img diffusion, text = lerp(start, end, alpha)
        --VAE--> 512x512 mel spectrogram --inverse mel + Griffin-Lim--> waveform --Tukey--> 5.12 s clip

An experiment is a list of keyframes (prompt + seed). With one keyframe it is a single clip.
With more, `steps` clips are made between each pair of keyframes and stitched into one piece.

Usage, from the repository root:
    python examples/generate_examples.py --out examples/output --device mps
    python examples/generate_examples.py --out examples/output --only t01 i01

Finished clips are skipped on a rerun, so an interrupted run picks up where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "tradifusion_pipeline"))

from inference import SEED_IMAGE_DIR, TradifusionPipeline  # noqa: E402
from imgs2audio import wav_bytes_from_spectrogram_image  # noqa: E402

SAMPLE_RATE = 44100


def frame_plan(experiment: dict) -> list[dict]:
    """
    List the clips of one experiment: which pair of keyframes it sits between and its alpha.

    Two keyframes and steps = N gives N clips with alpha = 0, 1/(N-1), ..., 1, the same
    spacing as TradifusionPipeline.txt2audio_tradfusion. With K keyframes the segments
    share their end points, so the total is (K - 1) * (N - 1) + 1 clips.
    """
    keyframes = experiment["keyframes"]
    if len(keyframes) == 1:
        return [{"start": keyframes[0], "end": keyframes[0], "alpha": 0.0}]

    steps = experiment["steps"]
    plan = []
    for k in range(len(keyframes) - 1):
        for j in range(steps - 1):
            plan.append({"start": keyframes[k], "end": keyframes[k + 1], "alpha": j / (steps - 1)})
    plan.append({"start": keyframes[-2], "end": keyframes[-1], "alpha": 1.0})
    return plan


def clip_key(frame: dict, settings: dict) -> str:
    """Everything that decides the pixels of a clip, so identical clips are only made once."""
    return json.dumps([frame["start"], frame["end"], round(frame["alpha"], 6), settings], sort_keys=True)


def render_clip(pipeline: TradifusionPipeline, frame: dict, settings: dict) -> tuple[Image.Image, np.ndarray]:
    """Run one img2img interpolation step and turn the spectrogram into int16 audio."""
    seed_image = Image.open(os.path.join(SEED_IMAGE_DIR, settings["seed_image"] + ".png")).convert("RGB")
    image = pipeline.tradfuse(
        start_prompt=frame["start"]["prompt"],
        end_prompt=frame["end"]["prompt"],
        num_inference_steps=settings["num_inference_steps"],
        alpha=frame["alpha"],
        init_image=seed_image,
        start_seed=frame["start"]["seed"],
        end_seed=frame["end"]["seed"],
        denoising=settings["denoising"],
        guidance=settings["guidance"],
    )
    wav_bytes, _ = wav_bytes_from_spectrogram_image(
        image, duration=5119, nmels=512, maxvol=100, power_for_image=0.25, device="cpu"
    )
    _, samples = wavfile.read(wav_bytes)
    return image, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(HERE, "experiments.json"))
    parser.add_argument("--out", default=os.path.join(HERE, "output"))
    parser.add_argument("--device", default="mps", help="mps, cuda or cpu")
    parser.add_argument("--only", nargs="*", help="Experiment ids to run (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    experiments = [e for e in config["experiments"] if not args.only or e["id"] in args.only]

    pipeline = TradifusionPipeline.load_checkpoint("hdparmar/tradfusion-v2", device=args.device)
    pipeline.set_progress_bar_config(disable=True)
    # Attention one slice at a time: fits a 4 GB GPU, costs a little speed.
    pipeline.enable_attention_slicing()

    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    made = {}  # clip_key -> (png path, wav path), to reuse a clip that two experiments share
    for experiment in experiments:
        settings = {k: experiment.get(k, v) for k, v in config["defaults"].items()}
        exp_dir = os.path.join(args.out, experiment["id"])
        os.makedirs(exp_dir, exist_ok=True)

        clips = []
        for n, frame in enumerate(frame_plan(experiment), start=1):
            png = os.path.join(exp_dir, f"{n:02d}.png")
            wav = os.path.join(exp_dir, f"{n:02d}.wav")
            key = clip_key(frame, settings)
            t0 = time.time()
            if not (os.path.exists(png) and os.path.exists(wav)):
                if key in made:
                    Image.open(made[key][0]).save(png)
                    wavfile.write(wav, SAMPLE_RATE, wavfile.read(made[key][1])[1])
                else:
                    image, samples = render_clip(pipeline, frame, settings)
                    image.save(png)
                    wavfile.write(wav, SAMPLE_RATE, samples)
            made[key] = (png, wav)
            clips.append({
                "n": n,
                "alpha": round(frame["alpha"], 4),
                "start": frame["start"],
                "end": frame["end"],
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[{experiment['id']}] clip {n} alpha={frame['alpha']:.2f} {time.time() - t0:.0f}s", flush=True)

        # Stitch the clips end to end, as txt2audio_tradfusion does.
        full = np.concatenate([wavfile.read(os.path.join(exp_dir, f"{c['n']:02d}.wav"))[1] for c in clips])
        wavfile.write(os.path.join(exp_dir, "full.wav"), SAMPLE_RATE, full)

        manifest[experiment["id"]] = {**experiment, "settings": settings, "clips": clips,
                                      "duration_s": round(len(full) / SAMPLE_RATE, 2)}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
