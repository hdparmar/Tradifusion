"""
Build the project page (index.html + media/) from the clips made by generate_examples.py.

    spectrogram PNGs --> WebP frames and strips      (media/*.webp)
    clip / full WAVs --> 128 kbit/s mono MP3          (media/*.mp3)
    manifest.json    --> index.html, in the look of hdparmar.github.io ("Edge Print")

Usage, from the repository root:
    python examples/build_page.py --results examples/output

Needs ffmpeg on the PATH, Pillow with WebP, numpy, scipy and torchaudio.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess

import numpy as np
import torch
import torchaudio
from PIL import Image
from scipy.io import wavfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA = os.path.join(REPO, "media")
SEEDS = os.path.join(REPO, "src", "tradifusion_pipeline", "seed_images")

TILE_PX = 448   # single clips: shown at up to ~220 px, sharp on 2x screens
STRIP_PX = 256  # height of each frame in a strip


# ---------------------------------------------------------------- media

def to_mp3(wav: str, mp3: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-codec:a", "libmp3lame",
                    "-b:a", "128k", mp3], check=True)


def to_webp(img: Image.Image, path: str, quality: int = 82) -> None:
    img.convert("L").save(path, "WEBP", quality=quality, method=6)


def strip_image(pngs: list[str]) -> Image.Image:
    frames = [Image.open(p).convert("L").resize((STRIP_PX, STRIP_PX), Image.LANCZOS) for p in pngs]
    strip = Image.new("L", (STRIP_PX * len(frames), STRIP_PX), 255)
    for i, f in enumerate(frames):
        strip.paste(f, (i * STRIP_PX, 0))
    return strip


def spectrogram_picture(wav: str) -> Image.Image:
    """
    Draw a recording the way the dataset images were drawn (Table 1 of the thesis):
    100 ms Hann window, 10 ms hop, 512 mel bands from 60 to 8,500 Hz, power 0.25, dark = loud.
    Only used for the 2024 clips, which were published without their spectrograms.
    """
    sr, x = wavfile.read(wav)
    x = torch.from_numpy(x.astype(np.float32) / 32768.0)
    spec = torchaudio.transforms.Spectrogram(n_fft=17640, win_length=4410, hop_length=441, power=1.0)(x)
    mel = torchaudio.transforms.MelScale(n_mels=512, sample_rate=sr, f_min=60, f_max=8500, n_stft=8821,
                                         norm=None, mel_scale="htk")(spec)
    data = np.power(mel.numpy(), 0.25)
    data = 255 - data / data.max() * 255
    return Image.fromarray(np.flipud(data).astype(np.uint8)).resize((512 * max(1, round(len(x) / sr / 5.11)), 512))


# ---------------------------------------------------------------- page pieces

def e(s: str) -> str:
    return html.escape(str(s), quote=True)


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def tile(exp_id: str, label: str, prompt: str, duration: float, inset: str | None = None, src: str | None = None,
         img: str | None = None) -> str:
    """One playable 5-second clip: a spectrogram frame with film-edge metadata under it."""
    src = src or f"media/{exp_id}.mp3"
    img = img or f"media/{exp_id}.webp"
    inset_html = (f'<img class="inset" src="{e(inset)}" width="128" height="128" alt="" loading="lazy">'
                  if inset else "")
    return f"""
      <figure class="clip" data-audio="{e(src)}" data-duration="{duration:.2f}" data-frames="1">
        <button type="button" class="frame" aria-label="Play {e(prompt)}">
          <img class="spec" src="{e(img)}" width="{TILE_PX}" height="{TILE_PX}" alt="Generated mel spectrogram for: {e(prompt)}" loading="lazy">
          {inset_html}<span class="playhead" aria-hidden="true"></span>
          <span class="state mono" aria-hidden="true">▶</span>
        </button>
        <figcaption>
          <span class="mono edge"><span class="accent">▸ {e(exp_id.upper())}</span> {e(label)}</span>
          <span class="prompt">{e(prompt)}</span>
        </figcaption>
      </figure>"""


def strip(exp_id: str, head: str, title_html: str, note: str, labels: list[str], duration: float,
          src: str | None = None, img: str | None = None) -> str:
    """A run of clips stitched together: one wide image, one audio file, a playhead across both."""
    n = len(labels)
    src = src or f"media/{exp_id}.mp3"
    img = img or f"media/{exp_id}-strip.webp"
    cells = "".join(f'<span data-i="{i}">{e(l)}</span>' for i, l in enumerate(labels))
    note_html = f'<p class="note">{e(note)}</p>' if note else ""
    return f"""
      <figure class="strip" data-audio="{e(src)}" data-duration="{duration:.2f}" data-frames="{n}">
        <figcaption>
          <span class="mono edge"><span class="accent">▸ {e(exp_id.upper())}</span> {e(head)}</span>
          <span class="prompt">{title_html}</span>
          {note_html}
        </figcaption>
        <div class="scroller">
          <div class="strip-inner" style="--n:{n}">
            <button type="button" class="frame" aria-label="Play {e(head)}. Click a frame to jump to it.">
              <img class="spec" src="{e(img)}" width="{STRIP_PX * n}" height="{STRIP_PX}" alt="{n} generated spectrograms side by side" loading="lazy">
              <span class="playhead" aria-hidden="true"></span>
              <span class="state mono" aria-hidden="true">▶</span>
            </button>
            <div class="mono ticks" aria-hidden="true">{cells}</div>
          </div>
        </div>
        <p class="mono time" aria-live="off"><span class="t">0:00</span> / {mmss(duration)}</p>
      </figure>"""


def heading(frame: str, text: str, hid: str, aside: str = "") -> str:
    return f"""
      <div class="sh"><h2 id="{hid}" class="mono"><span class="accent">▸ {frame}</span><span>{e(text)}</span></h2>{aside}</div>"""


# ---------------------------------------------------------------- build

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="folder written by generate_examples.py")
    args = parser.parse_args()

    with open(os.path.join(args.results, "manifest.json")) as f:
        manifest = json.load(f)
    os.makedirs(MEDIA, exist_ok=True)

    def frames_of(exp_id: str) -> list[str]:
        return [os.path.join(args.results, exp_id, f"{c['n']:02d}.png") for c in manifest[exp_id]["clips"]]

    # Media for every experiment that finished.
    for exp_id, exp in manifest.items():
        pngs = frames_of(exp_id)
        if len(pngs) == 1:
            to_webp(Image.open(pngs[0]).resize((TILE_PX, TILE_PX), Image.LANCZOS), os.path.join(MEDIA, f"{exp_id}.webp"))
        else:
            to_webp(strip_image(pngs), os.path.join(MEDIA, f"{exp_id}-strip.webp"))
        to_mp3(os.path.join(args.results, exp_id, "full.wav"), os.path.join(MEDIA, f"{exp_id}.mp3"))

    used_seeds = sorted({exp["settings"]["seed_image"] for exp in manifest.values()})
    for s in used_seeds:
        to_webp(Image.open(os.path.join(SEEDS, s + ".png")).resize((STRIP_PX, STRIP_PX), Image.LANCZOS),
                os.path.join(MEDIA, f"seed-{s}.webp"))

    # The four clips from the 2024 page, re-encoded (two were WAV files named .mp3) and drawn.
    legacy = [("2024-a", "audio/t2.mp3"), ("2024-b", "audio/t9.mp3"), ("2024-c", "audio/tff0.mp3"),
              ("2024-d", "audio/final_transition_audio.wav")]
    legacy_dur = {}
    for key, path in legacy:
        src = os.path.join(REPO, path)
        if not os.path.exists(src):
            continue
        tmp = os.path.join(MEDIA, f"{key}.tmp.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "44100", tmp], check=True)
        to_mp3(tmp, os.path.join(MEDIA, f"{key}.mp3"))
        pic = spectrogram_picture(tmp)
        legacy_dur[key] = wavfile.read(tmp)[1].shape[0] / 44100
        os.remove(tmp)
        if pic.width == 512:
            to_webp(pic.resize((TILE_PX, TILE_PX), Image.LANCZOS), os.path.join(MEDIA, f"{key}.webp"))
        else:
            to_webp(pic.resize((STRIP_PX * pic.width // 512, STRIP_PX), Image.LANCZOS), os.path.join(MEDIA, f"{key}-strip.webp"))

    write_social_image(manifest, frames_of)
    page = render(manifest, legacy_dur)
    with open(os.path.join(REPO, "index.html"), "w") as f:
        f.write(page)
    total = sum(os.path.getsize(os.path.join(MEDIA, m)) for m in os.listdir(MEDIA))
    print(f"index.html written; media/ holds {len(os.listdir(MEDIA))} files, {total / 1e6:.1f} MB")


def write_social_image(manifest: dict, frames_of) -> None:
    """1200 x 630 card for link previews: five frames of the generated set on warm paper."""
    exp_id = "w02" if "w02" in manifest else next(iter(manifest))
    pngs = frames_of(exp_id)[:5]
    card = Image.new("RGB", (1200, 630), (243, 241, 237))
    size = 216
    x0 = (1200 - len(pngs) * size - (len(pngs) - 1) * 12) // 2
    for i, p in enumerate(pngs):
        card.paste(Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS), (x0 + i * (size + 12), 207))
    card.save(os.path.join(MEDIA, "social.jpg"), quality=85)


def render(manifest: dict, legacy_dur: dict) -> str:
    m = manifest
    has = lambda k: k in m  # noqa: E731
    sections = []

    # Hero: the generated set.
    hero = ""
    if has("w02"):
        x = m["w02"]
        names = ["Jig", "Reel", "Air", "Hornpipe"]
        steps = x["steps"]
        labels = []
        for c in x["clips"]:
            k = (c["n"] - 1) // (steps - 1)
            labels.append(names[k] if (c["n"] - 1) % (steps - 1) == 0 else "·")
        hero = strip("w02", f"{len(labels)} clips · seed {x['keyframes'][0]['seed']} · {mmss(x['duration_s'])}",
                     "A generated set: jig <span class='arrow'>→</span> reel <span class='arrow'>→</span> air "
                     "<span class='arrow'>→</span> hornpipe",
                     "Four prompts, one seed. Each 5-second frame sits a little further along the path between "
                     "two prompts, the way the thesis makes longer pieces (Section 5.4).", labels, x["duration_s"])

    # 01 Text to audio.
    tiles = [tile(k, f"seed {v['keyframes'][0]['seed']}", v["keyframes"][0]["prompt"], v["duration_s"])
             for k, v in m.items() if v["section"] == "text"]
    if tiles:
        sections.append(("text", "Text to audio", f"""
      <p class="lede">One prompt, one 5-second clip. Every training image had the same caption,
      <q>An Irish Traditional Tune</q>, so the words mostly nudge the result; the seed spectrogram it starts
      from shapes it more. Click a frame to play it.</p>
      <div class="grid">{''.join(tiles)}
      </div>"""))

    # 02 Between two prompts.
    strips = []
    for k, v in m.items():
        if v["section"] != "interp":
            continue
        a, b = v["keyframes"][0]["prompt"], v["keyframes"][-1]["prompt"]
        labels = [("α " if i == 0 else "") + f"{c['alpha']:.2f}".rstrip("0").rstrip(".") for i, c in enumerate(v["clips"])]
        strips.append(strip(k, f"{v['steps']} steps · seed {v['keyframes'][0]['seed']} · {mmss(v['duration_s'])}",
                            f"{e(a)} <span class='arrow'>→</span> {e(b)}", v.get("note", ""), labels, v["duration_s"]))
    if strips:
        sections.append(("interp", "Between two prompts", f"""
      <p class="lede">The text embedding slides from the first prompt to the second in even steps of α while the
      seed image and noise stay fixed. More steps give a smoother walk, and more repetition.</p>
      <div class="stack">{''.join(strips)}
      </div>"""))

    # 03 Walking the seed.
    if has("w01"):
        v = m["w01"]
        steps = v["steps"]
        seeds = [kf["seed"] for kf in v["keyframes"]]
        labels = [f"seed {seeds[(c['n'] - 1) // (steps - 1)]}" if (c["n"] - 1) % (steps - 1) == 0 else "·"
                  for c in v["clips"]]
        sections.append(("walk", "Walking the seed", f"""
      <p class="lede">Same prompt the whole way; only the noise moves, by spherical interpolation between three seeds.
      This is how Riffusion makes endless loops, and where the repetition the thesis mentions comes from.</p>
      <div class="stack">{strip("w01", f"{len(labels)} clips · {mmss(v['duration_s'])}", e(v['keyframes'][0]['prompt']), v.get('note', ''), labels, v['duration_s'])}
      </div>"""))

    # 04 What the settings do.
    rows = []
    s_ids = [k for k in ("s030", "s050", "s075", "s095") if has(k)]
    if s_ids:
        seed = m[s_ids[0]]["settings"]["seed_image"]
        seed_tile = f"""
      <figure class="clip still">
        <div class="frame"><img class="spec" src="media/seed-{e(seed)}.webp" width="{STRIP_PX}" height="{STRIP_PX}" alt="The seed spectrogram, a real recording from the dataset" loading="lazy"></div>
        <figcaption><span class="mono edge"><span class="accent">▸ SEED</span> from the dataset</span><span class="prompt">Where it starts</span></figcaption>
      </figure>"""
        rows.append(f"""
      <h3 class="mono sub">Denoising strength: how far it may wander from the seed</h3>
      <div class="grid five">{seed_tile}{''.join(tile(k, f"denoising {m[k]['settings']['denoising']:.2f}", 'An Irish traditional tune', m[k]['duration_s']) for k in s_ids)}
      </div>""")
    g_ids = [k for k in ("g01", "g03", "g07", "g12") if has(k)]
    if g_ids:
        rows.append(f"""
      <h3 class="mono sub">Guidance: how hard the prompt pulls</h3>
      <div class="grid">{''.join(tile(k, f"guidance {m[k]['settings']['guidance']:g}", 'An Irish traditional tune', m[k]['duration_s']) for k in g_ids)}
      </div>""")
    m_ids = [k for k in ("m1", "m2", "m3", "m4") if has(k)]
    if m_ids:
        rows.append(f"""
      <h3 class="mono sub">Seed image: the same prompt from four starting points</h3>
      <div class="grid">{''.join(tile(k, f"seed image {chr(65 + i)}", 'An Irish traditional tune', m[k]['duration_s'], inset=f"media/seed-{m[k]['settings']['seed_image']}.webp") for i, k in enumerate(m_ids))}
      </div>""")
    if rows:
        sections.append(("settings", "What the settings do", f"""
      <p class="lede">Prompt <q>An Irish traditional tune</q> and seed 42 throughout; one setting changes per row.
      Defaults are denoising 0.75 and guidance 7.</p>{''.join(rows)}"""))

    # 05 How it works.
    sections.append(("how", "How it works", """
      <ol class="tracks">
        <li><span class="mono accent">S1</span><div><h3>Cut</h3><p>About 1,700 recordings from Bob L. T. Sturm's collection, made mono at 44.1 kHz and cut into 5.12-second pieces, with silence removed.</p></div></li>
        <li><span class="mono accent">S2</span><div><h3>Draw</h3><p>Each piece becomes a 512 × 512 picture: a mel spectrogram with 512 bands from 60 to 8,500 Hz, 100 ms Hann windows, 10 ms apart. Dark means loud.</p></div></li>
        <li><span class="mono accent">S3</span><div><h3>Caption</h3><p>Every picture gets the same caption, <q>An Irish Traditional Tune</q>: about 9,000 image and text pairs.</p></div></li>
        <li><span class="mono accent">S4</span><div><h3>Fine-tune</h3><p>Starting from Riffusion v1, the U-Net trains for 122k steps (202 epochs) on two RTX A6000s, about four days. The CLIP text encoder stays frozen.</p></div></li>
        <li><span class="mono accent">S5</span><div><h3>Generate</h3><p>A real spectrogram from the dataset is encoded, noised (denoising 0.75) and denoised over 50 steps, steered by the prompt. Sliding α between prompts or seeds gives the next frame.</p></div></li>
        <li><span class="mono accent">S6</span><div><h3>Listen</h3><p>Inverse mel scale, then 32 rounds of Griffin-Lim to guess the phase, then a Tukey window on the first and last 100 ms. Frames are joined end to end.</p></div></li>
      </ol>
      <dl class="facts mono">
        <div><dt>Pairs</dt><dd>≈ 9,000</dd></div>
        <div><dt>Frame</dt><dd>512 × 512 · 5.12 s</dd></div>
        <div><dt>Training</dt><dd>122k steps · 2 × A6000 · 4 days</dd></div>
        <div><dt>One clip here</dt><dd>≈ 90 s on a Radeon 5500M</dd></div>
      </dl>"""))

    # 06 The 2024 examples.
    old = []
    for i, key in enumerate(("2024-a", "2024-b", "2024-c")):
        if key in legacy_dur:
            old.append(tile(key, "single clip", "An Irish traditional tune", legacy_dur[key]))
    old_strip = ""
    if "2024-d" in legacy_dur:
        old_strip = strip("2024-d", f"2 steps · {mmss(legacy_dur['2024-d'])}",
                          "An Irish traditional tune <span class='arrow'>→</span> with a fiddle lead", "",
                          ["α 0", "1"], legacy_dur["2024-d"])
    if old or old_strip:
        sections.append(("2024", "The 2024 examples", f"""
      <p class="lede">The clips this page launched with in March 2024, made on the CPU of this same laptop at about
      ten minutes each. Their spectrograms are redrawn here from the audio.</p>
      <div class="grid">{''.join(old)}
      </div>
      <div class="stack">{old_strip}
      </div>"""))

    # 07 Limits.
    sections.append(("limits", "What it can't do", """
      <ul class="plain">
        <li><strong>It doesn't write tunes.</strong> Each frame is 5 seconds with no idea of the AABB form of a reel or jig. Long pieces are frames placed side by side.</li>
        <li><strong>Griffin-Lim guesses the phase,</strong> so the sound is smeared and a little metallic. A trained vocoder (MelGAN, HiFi-GAN) would sound much cleaner.</li>
        <li><strong>The prompt is a weak handle.</strong> With one caption for all 9,000 images, words like <q>banjo</q> or <q>harp</q> come from Stable Diffusion's text encoder rather than from the music. The seed image matters more.</li>
        <li><strong>Joins are audible.</strong> Every frame is faded in and out over 100 ms before it is joined to the next.</li>
        <li><strong>The recordings stay private.</strong> The model learned from commercial recordings, so only its outputs are published here, not the music it learned from.</li>
      </ul>"""))

    body = []
    for n, (hid, title, inner) in enumerate(sections, start=1):
        body.append(f"""
    <section aria-labelledby="{hid}" class="section">
      <div class="col">{heading(f'{n:02d}', title, hid)}{inner}
      </div>
    </section>""")

    if hero:
        hero = f"""
      <section aria-labelledby="listen" class="band">
        <div class="col">{heading('00', 'Listen first', 'listen')}
          <div class="stack">{hero}
          </div>
        </div>
      </section>"""
    clips_total = sum(len(v["clips"]) for v in m.values())
    with open(os.path.join(os.path.dirname(__file__), "page_template.html")) as f:
        page = f.read()
    for token, value in {
        "%%HERO%%": hero,
        "%%SECTIONS%%": "".join(body),
        "%%CLIPS%%": str(clips_total),
        "%%FILM%%": " ▸ ".join(f"{i} ▸ {i}A" for i in range(1, 49)),
    }.items():
        page = page.replace(token, value)
    return page

if __name__ == "__main__":
    main()
