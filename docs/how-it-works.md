# How it works

## Principle: one master, many exporters

The engine renders **one master animation**: a list of RGBA frames at high resolution with full transparency, an fps value and a loop flag. Every platform then gets its own **exporter** that turns the master into a file of the right format, size and weight. Adding a platform means adding one exporter; the engine does not change.

```
input PNG
  -> engine (AI chain or procedural preset) -> MasterAnimation (RGBA frames)
  -> exporters: twitch_gif, discord_sticker, discord_emoji,
                telegram_sticker, telegram_emoji, obs_overlay
  -> validator (every file against the platform limits)
  -> delivery: downloads, Telegram and Discord publishers, Twitch archive
```

## Modules

| Module | Responsibility |
|---|---|
| `core/models`, `core/specs`, `core/alpha` | Shared types (`MasterAnimation`, `ExportResult`, `ExportTarget`), platform limits as data, premultiplied-alpha resize and transparency thresholds |
| `preprocess` | Alpha check, crop by alpha, square canvas with headroom (procedural path) |
| `engine/presets` | Procedural presets: bounce, squash and stretch, wiggle |
| `engine/ai` | AI chain: description, framing, decor, generation, cut, skin repair, loop, emote focus; Replicate client with budgets and cost journal |
| `exporters` | One exporter per target, timing plan and weight-fit loop |
| `validator` | Reads a file back and checks it against `core/specs` |
| `ai_pipeline`, `pipeline` | End-to-end runs for the AI and procedural modes, manifest and comparison page |
| `variants`, `raws`, `reprocess` | Several seeds at once, library of raw clips, free re-export of a stored clip |
| `publish` | Telegram, Discord and Twitch publishers (outgoing HTTP only) |
| `web` | FastAPI app, background worker, SQLite store, templates |
| `cli` | Typer command-line interface |

## The AI chain

### 1. Description

A cheap vision LLM (`openai/gpt-5-nano` on Replicate) looks at the character and writes what it does: a natural, small action that follows from its pose and emotion, plus what the decor does. In the web UI the text is shown in Ukrainian and translated back to English before generation. Fixed requirements are appended in code, so they are always present: static camera, flat background, return to the starting pose, no new objects, same drawing style, same skin and hand colours, mouth and hands keeping the size and shape of the drawing. All of them are worded positively, see [prompting.md](prompting.md).

If Replicate's moderation flags the green frame (E005), the description is retried once with the character on neutral grey.

### 2. Framing

The character is cropped to its alpha, scaled and placed on a flat **green** 16:9 canvas (832x480 for 480p). Margins depend on the action: more headroom only when the action raises arms, stretches or jumps; otherwise the character fills the full width so the clip carries more detail.

### 3. Decor separation

Connected components of the input alpha are analysed. The largest is the character; smaller components that stand apart from it with a transparent gap (and are not specks or detached body parts) are decor candidates. The vision model confirms each one is decor and not something the character holds, and picks a motion: **twinkle**, **sway**, **float** or **pulse**. If the model fails, a shape heuristic decides.

Decor is removed from the model input, so the video model animates the character alone. Later each decor element is resampled once from the original full-resolution pixels and moved by its own periodic pose, with its own phase and a whole number of cycles per loop, then laid back at its place in the composition. The decor no longer drags along with the head, and the loop stays seamless.

### 4. Generation

`wan-video/wan-2.2-i2v-fast` at 480p, 81 frames played at 30 fps (2.7 s). The model input frame is passed as both the **first and the last image**, so the clip returns to its start. The raw clip is cached by a hash of the input frame, prompt, model and parameters: a repeat run with the same inputs costs nothing.

Several variants use the same prompt with different seeds. Every clip goes to the raw library with its `model_input.png` and `meta.json` (prompt, seed, framing, focus zone, cost), so a liked motion can be reused.

### 5. Cut back to transparency

The background was painted green, so alpha comes from how green a pixel is (`G - max(R, B)`), which stays linear across soft edges. The key colour is measured on each frame's border, because models drift the background slightly. Colours are fixed only in the edge band: semi-transparent pixels take the colour of the nearby opaque interior, and a narrow band inside the contour has green capped at `max(R, B)` to remove spill. The rest of the character is left untouched.

An optional `rembg` cut is available for shadows and smudges a key cannot separate.

### 6. Skin hue repair

Video models sometimes turn a moving hand lilac in part of the frames. The skin tone is measured on the input PNG; frame pixels close to it in saturation and brightness but shifted towards magenta get their hue rotated back, with a soft ramp at the edge of the band. Colours the character really has in that range (pink clothes, lips) are protected.

### 7. Loop

`trim_to_match` cuts the clip at the frame most similar to frame 0. A seam score compares the wrap jump with a regular frame step; if the trimmed seam is not clean, a **ping-pong** loop over the start of the clip is used instead. The frames are then cropped to a tight square around the union of alpha of all frames, which becomes the master.

### 8. Emote focus

Emotes (Twitch GIF, Discord emoji, Telegram emoji) are shown at about 28 px, where a full composition turns into a blur. With `--emote-focus auto` (default) the vision model returns the key zone (face and gesture) as a rectangle; an invalid reply or an implausible rectangle falls back to the full composition. The zone is mapped onto the video through the framing, grown to hold the motion of every loop frame, and made square. At 28 px the face then takes about 11 px instead of 4-5. Stickers and OBS always get the full composition.

## Exporters and weight fitting

Each exporter:

1. **Plans the timing.** Duration and frame count are brought within the target limits (for example at most 60 frames for Twitch, 3 s and 30 fps for Telegram) while keeping the loop seamless.
2. **Downscales** with premultiplied alpha to the target size. The Telegram sticker gets a light unsharp mask after the downscale to 512 px.
3. **Encodes** with decreasing quality and, if that is not enough, decreasing fps, until the file fits **95% of the weight limit**. Files do not sit right at the edge, and the quality difference is invisible.
4. **Validates** the written file by reading it back.

Formats:

- **GIF** (Twitch): binary transparency, so semi-transparent edges are composed over a dark matte close to the Twitch dark chat (`#18181B`).
- **APNG** (Discord sticker): full alpha.
- **Animated WEBP** (Discord emoji): full alpha.
- **WEBM VP9 with alpha** (Telegram sticker and emoji, OBS): the Telegram sticker uses two-pass constrained quality with a bitrate cap derived from the weight limit; the Telegram emoji aims at 64 KB and thins frames no lower than 15 fps.

## Procedural mode

`stickermotion render` skips the AI chain: the PNG is cropped by alpha and placed on a 1024 px square canvas with 18% headroom, and a preset (bounce, squash and stretch, wiggle) transforms it per frame. Presets come in two classes: **emote** (wide, simple, high-contrast motion that reads at 28 px) and **sticker** (finer motion). The same exporters and validator are used.

## Web service

- **FastAPI** with server-rendered Jinja2 pages and a little local JavaScript; no CDN, fonts and assets are served locally.
- **SQLite** store under `DATA_DIR` for stickers, descriptions, costs and publication state.
- **Background worker** in the same process: descriptions run on a small pool, generations go through one worker thread and a FIFO queue. All paid Replicate steps of all stickers are serialized by one lock, so a batch upload does not hit Replicate's rate limit.
- **Budgets:** every paid call goes through a ledger whose limit is the smaller of the per-run budget and what is left of the daily budget. The check happens before the call.
- **Security:** one password, signed session cookie, CSRF tokens on every form, request-size guard, tokens redacted from every message, log line and database row.

## Replicate client

- The budget is checked before every paid call; the real cost is booked from Replicate's metrics afterwards (an upper bound when no metrics are reported).
- **402** (insufficient credit) is never retried.
- **429** (rate limit): Replicate may have created the prediction anyway. Before the delayed retry the client lists recent predictions and cancels the ones this run created that are still queued or running; predictions of other runs in the same process are never touched.
- Every call is logged in `costs.jsonl` with the token redacted and image data replaced by short markers.
