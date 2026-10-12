# StickerMotion

[Українська версія](README.uk.md)

StickerMotion turns a still drawing of a character (a PNG with a transparent background) into animated stickers and emotes for Twitch, Discord, Telegram and OBS. It describes a natural action for the character, has an image-to-video model animate it, cuts the character back out into full transparency, makes a seamless loop and exports a file for every platform that fits its size and weight limits. Finished stickers can be published to a Telegram pack and a Discord server in one click.

It is a small self-hosted service for one person: a password-protected web UI plus a command-line tool.

## Features

- **AI animation from a single PNG.** A vision model proposes what the character does (yawns, waves, blinks), you can edit the text, then Wan 2.2 image-to-video animates it.
- **Full transparency.** The clip is generated on a flat green background and keyed back to RGBA with soft edges; skin tones that drift in the video are repaired.
- **Seamless loops.** First frame equals last frame, the loop is trimmed at the closest match, with a ping-pong fallback.
- **Decor as separate layers.** Stars, a moon, hearts or "Zzz" next to the character are detected and animated on their own (twinkle, sway, float, pulse) instead of moving with the head.
- **Emote focus.** Emotes shown at 28 px are cropped to the face and gesture so they stay readable in chat.
- **One master, many exporters.** Twitch GIF (auto and 28/56/112 manual), Discord sticker (APNG) and emoji, Telegram video sticker and custom emoji (WEBM VP9 with alpha), OBS overlay. Every file is validated against platform limits; weight is tuned to 95% of the limit.
- **Variants.** Generate 2-3 clips with different seeds and pick the best motion; reuse a liked motion later without paying again.
- **Publishing.** Telegram sticker pack and custom emoji pack, Discord animated emoji and sticker, a ready-made archive with instructions for Twitch.
- **Spend control.** Per-generation and daily budgets checked before every paid call, a cost journal for every request.
- **Procedural presets** (bounce, squash and stretch, wiggle) as a free fallback that needs no API key.

## Example result

Input: a drawn cartoon fox sitting with one paw raised, with three small hearts floating around it, on a transparent background.

Output after a minute or two: the fox waves its paw twice, blinks and tilts its head, then settles back into the starting pose, while the hearts float and pulse on their own. You get a 512 px Telegram video sticker under 256 KB, a 100 px Telegram emoji, a 320 px Discord sticker, a 128 px Discord emoji, Twitch GIFs at 28, 56, 112 and 224 px (the emote versions cropped to the face), and a large WEBM for OBS, all looping without a visible seam.

## How it works

```
PNG with transparency
  -> action description (vision LLM gpt-5-nano, editable)
  -> framing on a green 16:9 canvas, decor taken out
  -> Wan 2.2 i2v fast, 480p, first frame = last frame
  -> chroma key back to RGBA, skin hue repair
  -> seamless loop, decor laid back as separate animated layers
  -> master animation (RGBA frames)
  -> exporters per platform, weight fitted to 95% of the limit, validator
  -> download or publish (Telegram, Discord, Twitch archive)
```

More detail: [docs/how-it-works.md](docs/how-it-works.md).

## Quick start (Docker)

You need Docker with Docker Compose and a [Replicate](https://replicate.com) API token.

```bash
git clone https://github.com/webmosaic/stickermotion.git stickermotion
cd stickermotion
cp .env.example .env
# edit .env: set APP_PASSWORD and REPLICATE_API_TOKEN
docker compose up -d --build
```

Open http://127.0.0.1:8000, log in with `APP_PASSWORD` and upload a PNG. The port is bound to localhost only; to reach the service from the internet put it behind nginx with HTTPS (see [docs/getting-started.md](docs/getting-started.md)).

Without Docker: Python 3.12, ffmpeg with libvpx-vp9, then `pip install .` and `stickermotion serve`.

## Documentation

- [Getting started](docs/getting-started.md): requirements, local run, Docker Compose, VPS behind nginx with HTTPS, updates
- [Configuration](docs/configuration.md): all environment variables
- [Usage](docs/usage.md): the web UI step by step, the CLI, input PNG requirements
- [Publishing](docs/publishing.md): Telegram and Discord bots, Twitch upload
- [How it works](docs/how-it-works.md): architecture and the processing chain
- [Platform specs](docs/platform-specs.md): file limits of every platform
- [Prompting](docs/prompting.md): how to write action descriptions that work with Wan
- [Troubleshooting](docs/troubleshooting.md): common errors and what to do
- [Design decisions](docs/design-decisions.md): why things are the way they are

## Cost

Generation runs on Replicate with your own token; you pay Replicate directly.

- One Wan 2.2 i2v fast clip at 480p (81 frames) costs about **$0.05**. 720p costs about $0.11 and gives no visible gain at sticker sizes.
- Helper calls to `openai/gpt-5-nano` (action description, translation, decor check, emote focus) cost fractions of a cent, around $0.0002 each.
- So one sticker is about $0.05, or $0.10-0.15 if you generate 2-3 variants. Reprocessing a stored clip, downloads and publishing are free.
- Requests rejected by Replicate's moderation (E005) are not billed.

Prices were checked in October 2026 and may change; see the model pages on Replicate.

## Limitations

- Single user: one password, no accounts or roles.
- The web UI and the messages it shows are in Ukrainian; you can type action descriptions in Ukrainian or English. The CLI and documentation are in English.
- The input must be a PNG with real transparency; background removal is not part of the default path.
- AI generation needs a Replicate account and internet access; there is no local GPU mode. The service itself runs on a CPU-only server.
- Wan 2.2 fast has no negative prompt, so results vary between seeds; generating variants and picking one is part of the workflow.
- Twitch has no API for emote uploads: the service prepares files and instructions, the upload is manual.
- Telegram custom emoji are capped at 64 KB in practice, which limits detail and frame rate for emoji.
- One generation runs at a time; paid calls are serialized to stay within Replicate's rate limits.

## Development

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest                 # fast suite, no network, no spend
pytest -m slow         # full render matrix through the CLI
ruff check . && ruff format --check .
```

Tests use fake Replicate, Telegram and Discord transports; no request leaves the machine. Test images in `tests/fixtures/` are synthetic and can be regenerated with `python scripts/make_fixtures.py`.

## License

[MIT](LICENSE). The bundled Onest font is under the SIL Open Font License (`src/stickermotion/web/static/OFL.txt`).
