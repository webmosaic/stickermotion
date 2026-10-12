# Platform specs

File limits StickerMotion exports for. Checked in October 2026; platforms change their rules, so check the official pages before relying on a number. The same limits live in code in `src/stickermotion/core/specs.py` and can be printed with `stickermotion targets`.

Units: 1 KB = 1024 bytes, 1 MB = 1024 KB (conservative for limits). Sizes marked "recommended" produce a validator warning, not an error.

## Twitch: animated emotes

| Parameter | Value |
|---|---|
| Format | GIF, square |
| Size (auto mode) | One file from 112x112 to 4096x4096 (StickerMotion exports 224) |
| Size (manual mode) | 28x28, 56x56, 112x112 |
| Weight | Up to 1 MB (auto), up to 512 KB per file (manual) |
| Frames | At most 60 |
| Flashing | No more than 3 times per second |
| Access | Affiliates and Partners |
| Upload | Manual in the creator dashboard, no public API |

The main display size is 28 px. GIF has only binary transparency, so soft edges are composed over a dark matte (`#18181B`, the Twitch dark chat) to hide the fringe; on the light theme the edge looks 1 px darker.

## Discord: stickers

| Parameter | Value |
|---|---|
| Format | APNG (exported), GIF, PNG, Lottie |
| Size | 320x320 |
| Weight | Up to 512 KB |
| Duration | Up to 5 seconds |
| Access | Custom stickers come with server boost slots |
| Upload | API (bot with Manage Expressions) |

## Discord: emoji

| Parameter | Value |
|---|---|
| Format | GIF, WEBP, APNG (animated), PNG (static); StickerMotion exports animated WEBP |
| Size | Square, recommended 128x128 |
| Weight | Up to 256 KB |
| Access | Using animated emoji requires Nitro on the user's side |
| Upload | API |

## Telegram: video stickers

| Parameter | Value |
|---|---|
| Format | WEBM, VP9 codec, with alpha, no audio |
| Size | 512 px on the longer side |
| Duration | Up to 3 seconds |
| Frame rate | Up to 30 fps |
| Weight | Up to 256 KB |
| Upload | Bot API (`createNewStickerSet`, `addStickerToSet`) |

## Telegram: custom emoji

| Parameter | Value |
|---|---|
| Format | WEBM VP9 with alpha |
| Size | 100x100 |
| Weight | 256 KB per the documentation; **about 64 KB in practice** (see below) |
| Other | Same as video stickers |

**Practical limit.** In testing, Telegram rejected a 173 KB video emoji when creating a pack, although the documentation says 256 KB. StickerMotion therefore targets 64 KB for custom emoji. To get there the exporter lowers quality first and then the frame rate, but never below 15 fps and never shortens the loop.

## OBS: overlay

| Parameter | Value |
|---|---|
| Format | WEBM VP9 with alpha (or a PNG sequence) |
| Size | Any, matches the master |
| Weight | No hard limit |
| Display | Browser Source or Media Source |

## Result classes

- **Emote** (small display): Twitch, Discord emoji, Telegram custom emoji. Simple, wide, high-contrast motion that reads at 28 px. Cropped to the face and gesture by default.
- **Sticker** (large display): Telegram video sticker, Discord sticker. Finer details are fine. Always the full composition.
- **Overlay**: large size, no hard weight limit.
