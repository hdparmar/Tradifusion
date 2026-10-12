"""
Build the project page (index.html + media/) from the clips made by generate_examples.py.

    spectrogram PNGs --> WebP frames and strips      (media/*.webp)
    clip / full WAVs --> 128 kbit/s mono MP3          (media/*.mp3)
    manifest.json    --> index.html, in the look of hdparmar.github.io ("Edge Print")

Usage, from the repository root:
    python examples/build_page.py --results examples/output

Needs ffmpeg on the PATH, Pillow with WebP, numpy and scipy.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess

import numpy as np
from PIL import Image
from scipy.io import wavfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA = os.path.join(REPO, "media")
SEEDS = os.path.join(REPO, "src", "tradifusion_pipeline", "seed_images")

TILE_PX = 448   # single clips: shown at up to ~220 px, sharp on 2x screens
STRIP_PX = 256  # height of each frame in a strip


# ---------------------------------------------------------------- media

def to_mp3(wav: str, mp3: str, target_rms_db: float = -20.0, peak_db: float = -1.0) -> None:
    """
    Encode for the web at a common level. One fixed gain per file brings its RMS to target_rms_db,
    lowered if needed so the peak stays under peak_db. No compression, so the dynamics are untouched.
    """
    sr, x = wavfile.read(wav)
    x = x.astype(np.float64) / 32768.0
    rms = np.sqrt(np.mean(x ** 2)) or 1e-9
    gain = min(10 ** (target_rms_db / 20) / rms, 10 ** (peak_db / 20) / (np.abs(x).max() or 1e-9))
    tmp = mp3 + ".level.wav"
    wavfile.write(tmp, sr, (x * gain * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-codec:a", "libmp3lame",
                    "-b:a", "128k", mp3], check=True)
    os.remove(tmp)


def to_webp(img: Image.Image, path: str, quality: int = 82) -> None:
    img.convert("L").save(path, "WEBP", quality=quality, method=6)


def strip_image(pngs: list[str]) -> Image.Image:
    frames = [Image.open(p).convert("L").resize((STRIP_PX, STRIP_PX), Image.LANCZOS) for p in pngs]
    strip = Image.new("L", (STRIP_PX * len(frames), STRIP_PX), 255)
    for i, f in enumerate(frames):
        strip.paste(f, (i * STRIP_PX, 0))
    return strip


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
    # .mono is uppercase, which would turn α into a capital alpha that reads as "A".
    cells = "".join(f'<span data-i="{i}">{e(l).replace("α", "<i class=greek>α</i>")}</span>'
                    for i, l in enumerate(labels))
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

    # generate_examples.py scores each frame's colour; a grey spectrogram scores 2-3.
    for exp_id, exp in manifest.items():
        for c in exp["clips"]:
            if c.get("colour", 0) > 6:
                print(f"WARNING: {exp_id} clip {c['n']} looks broken (colour {c['colour']})")

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


    write_social_image(manifest, frames_of)
    page = render(manifest)
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


def render(manifest: dict) -> str:
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
                     "Four prompts, one after the other. Each five-second frame moves a little closer to the next.",
                     labels, x["duration_s"])

    # 01 Text to audio.
    tiles = [tile(k, f"seed {v['keyframes'][0]['seed']}", v["keyframes"][0]["prompt"], v["duration_s"])
             for k, v in m.items() if v["section"] == "text"]
    if tiles:
        sections.append(("text", "Text to audio", f"""
      <p class="lede">One prompt, five seconds each. Every training image had the same caption,
      <q>An Irish Traditional Tune</q>, so the words only nudge it. The spectrogram it starts from matters more.</p>
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
                            f"{e(a)} <span class='arrow'>→</span> {e(b)}", "", labels, v["duration_s"]))
    if strips:
        sections.append(("interp", "Between two prompts", f"""
      <p class="lede">The prompt slides from one to the other while everything else stays put. α is how far along
      it is, from 0 to 1. More steps make the change smoother, but it also repeats more.</p>
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
      <p class="lede">Same prompt the whole way. Only the random noise changes, drifting between three seeds.
      It's how Riffusion makes its endless loops.</p>
      <div class="stack">{strip("w01", f"{len(labels)} clips · {mmss(v['duration_s'])}", e(v['keyframes'][0]['prompt']), '', labels, v['duration_s'])}
      </div>"""))

    # 04 What the settings do.
    rows = []
    s_ids = [k for k in ("s030", "s050", "s075", "s095") if has(k)]
    if s_ids:
        seed = m[s_ids[0]]["settings"]["seed_image"]
        seed_tile = f"""
      <figure class="clip still">
        <div class="frame"><img class="spec" src="media/seed-{e(seed)}.webp" width="{STRIP_PX}" height="{STRIP_PX}" alt="The seed spectrogram, a real recording from the dataset" loading="lazy"></div>
        <figcaption><span class="mono edge"><span class="accent">▸ START</span> from the dataset</span><span class="prompt">Where it starts</span></figcaption>
      </figure>"""
        rows.append(f"""
      <h3 class="mono sub">Denoising: how far it can drift from where it starts</h3>
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
      <h3 class="mono sub">Starting spectrogram: four different starts</h3>
      <div class="grid">{''.join(tile(k, f"start {chr(65 + i)}", 'An Irish traditional tune', m[k]['duration_s'], inset=f"media/seed-{m[k]['settings']['seed_image']}.webp") for i, k in enumerate(m_ids))}
      </div>""")
    if rows:
        sections.append(("settings", "What the settings do", f"""
      <p class="lede">Same prompt and seed throughout, one setting changed per row. The rest of the page uses
      denoising 0.75 and guidance 7.</p>{''.join(rows)}"""))

    # 05 How it works.
    sections.append(("how", "How it works", """
      <ol class="tracks">
        <li><span class="mono accent">S1</span><div><h3>Cut</h3><p>About 1,700 recordings from Bob L. T. Sturm's collection, cut into five-second pieces.</p></div></li>
        <li><span class="mono accent">S2</span><div><h3>Draw</h3><p>Each piece becomes a 512 × 512 mel spectrogram, a picture of the sound.</p></div></li>
        <li><span class="mono accent">S3</span><div><h3>Caption</h3><p>Every picture gets the same caption, <q>An Irish Traditional Tune</q>.</p></div></li>
        <li><span class="mono accent">S4</span><div><h3>Fine-tune</h3><p>Starting from Riffusion, the model trains on those pairs for about four days on two GPUs.</p></div></li>
        <li><span class="mono accent">S5</span><div><h3>Generate</h3><p>Take a real spectrogram, add noise, and let the model redraw it with the prompt as a guide.</p></div></li>
        <li><span class="mono accent">S6</span><div><h3>Listen</h3><p>Griffin-Lim turns the picture back into sound, and the five-second frames are joined end to end.</p></div></li>
      </ol>
      <dl class="facts mono">
        <div><dt>Pairs</dt><dd>≈ 9,000</dd></div>
        <div><dt>Frame</dt><dd>512 × 512 · 5.12 s</dd></div>
        <div><dt>Training</dt><dd>122k steps · 4 days</dd></div>
      </dl>"""))


    # 06 Limits.
    sections.append(("limits", "What it can't do", """
      <ul class="plain">
        <li><strong>It doesn't write tunes.</strong> Each frame is five seconds long and knows nothing about the AABB shape of a reel or jig. Longer pieces are frames placed side by side.</li>
        <li><strong>It sounds a bit metallic.</strong> Griffin-Lim has to guess the phase. A neural vocoder like HiFi-GAN would sound cleaner.</li>
        <li><strong>The prompt doesn't do much.</strong> All 9,000 images had the same caption, so words like <q>banjo</q> or <q>harp</q> don't mean much to it.</li>
        <li><strong>You can hear the joins.</strong> Each frame fades in and out before it's joined to the next.</li>
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
    with open(os.path.join(os.path.dirname(__file__), "page_template.html")) as f:
        page = f.read()
    for token, value in {
        "%%HERO%%": hero,
        "%%SECTIONS%%": "".join(body),
                "%%FILM%%": " ▸ ".join(f"{i} ▸ {i}A" for i in range(1, 49)),
    }.items():
        page = page.replace(token, value)
    return page

if __name__ == "__main__":
    main()
