# Usage

StickerMotion has a web UI for everyday work and a CLI for batch runs and experiments. The UI labels are in Ukrainian; below each step gives the English meaning and the Ukrainian label in quotes so you can find the button.

## Input PNG requirements

The result is only as good as the input. Prepare the PNG like this:

- **Real transparency.** A PNG with an alpha channel and a fully transparent background. A white or coloured background is not removed by default, and files without any transparency are rejected.
- **One character.** One drawn character, not a scene. The largest opaque shape is treated as the character.
- **Decor as separate islands.** Stars, a moon, hearts, "Zzz" and other small elements should not touch the character: leave a visible transparent gap around each. Then they are detected and animated on their own. Decor that touches the character becomes part of it and moves with it.
- **Objects the character holds** (a cup, a phone, a pillow) should touch the hand; then they stay part of the character.
- **Enough resolution.** 512-2048 px on the longer side works well. Accepted: 64 to 4096 px per side, up to `MAX_UPLOAD_MB` (10 MB by default).
- **Clean edges.** Soft anti-aliased edges are fine; stray semi-transparent specks around the drawing are ignored but are better removed.
- **A clear face.** Emotes are cropped to the face and gesture; a readable face at small size gives better emotes.

## Web UI step by step

1. **Log in** ("Вхід") with `APP_PASSWORD`.

2. **Upload a PNG** ("Новий стікер"). Drag the file onto the drop zone ("Перетягніть PNG сюди") or click to pick it, then press "Завантажити й описати" (upload and describe). The file is checked first; if it has no transparency or is too large you see "Файл не підходить" with the reason.

3. **Automatic description.** The vision model looks at the character and proposes a natural motion that follows from its pose and emotion ("Описуємо дію", usually 5-15 seconds, costs a fraction of a cent). The text appears in Ukrainian in the field "Що робить персонаж" (what the character does).

4. **Edit the description.** Keep it or rewrite it in Ukrainian or English. Before generation the text is translated to English for the video model; fixed requirements (static camera, flat background, return to the start pose, same style, keep the original mouth and hands) are always appended in code. Tips for good descriptions are in [prompting.md](prompting.md). "Описати ще раз" asks the model for a new description.

5. **Animate decor separately** ("Анімувати декор окремо"). This checkbox appears when separate decor was found, with a list of what was found (for example "moon, star x3"). On (default): the decor is taken out of the video and animated on its own, independent of the head. Off: the decor stays in the video as part of the scene.

6. **Number of variants.** Choose 1, 2 or 3 clips next to the generate button. Several variants use the same description with different seeds; the button shows the total price. Wan has no negative prompt, so 2-3 variants are the most reliable way to get a good motion.

7. **Generate** ("Генеруємо стікер"). One generation runs at a time; others wait in a queue. You can close the page: the work continues on the server and the sticker appears in the history ("Історія"). A clip takes one to a few minutes.

8. **Pick a variant** ("Виберіть варіант"). With 2-3 variants you get a Telegram-sticker preview of each. "Вибрати" (choose) prepares all platform files from that clip. This is free; the other clips stay below and can be used later.

9. **Review the result.** Previews on dark ("Темний"), light ("Світлий") and checkerboard ("Шаховий") backgrounds, and a chat row with emotes at their real 28 px size. Files that exceed a platform limit are marked "поза лімітом"; normally all files pass ("Усі файли пройшли перевірку").

10. **Use this motion** ("Використати цей рух"). The "Попередні рухи" block (previous motions) keeps the last 5 clips of the sticker. Taking one of them re-cuts and re-exports it for free, for example after you changed the decor setting. "Перегенерувати" (regenerate) makes a new paid clip with the same or an edited description.

11. **Download.** Each platform has its files and an archive ("Завантажити"); "Завантажити все разом" downloads everything in one ZIP. "Підготувати для Twitch" gives a ZIP with the manual 28/56/112 GIFs, the auto-mode GIF and an instruction.

12. **Publish** ("Опублікувати"). One click adds the sticker to your Telegram pack (optionally also to the custom emoji pack: "Також додати в пак кастомних емодзі") and uploads an animated emoji and a sticker to your Discord server. Set the name and the emoji before publishing; they are suggested from the description. Publishing again does not create a duplicate; "Прибрати з платформи" removes it. Setup is in [publishing.md](publishing.md).

The settings page ("Налаштування") shows which platforms are connected, the bot, pack names, the Discord server and free slots. It is read-only: platforms are configured with environment variables, and tokens are never shown.

The history page lists every sticker with its description, files and cost, plus the total spent.

## Command line

Install the package (`pip install .`) and run `stickermotion --help`. All commands work offline except `animate` and `reprocess --emote-focus auto`.

### `targets`: platform limits

```bash
stickermotion targets
```

Prints every export target with its format, sizes, weight limit, frame, duration and fps limits.

### `presets`: procedural presets

```bash
stickermotion presets
```

Lists the free procedural presets: `bounce`, `squash_stretch`, `wiggle`.

### `render`: procedural animation (free)

```bash
stickermotion render character.png                                  # all presets, all targets
stickermotion render character.png --preset wiggle --target twitch_gif --out out
stickermotion render character.png --preset bounce --preset wiggle --target telegram_sticker
```

Writes `out/<input name>/<preset>/<files>`, a `manifest.json` and a comparison page `index.html`. Exit code 1 if any file fails validation.

### `animate`: AI animation (paid)

Needs `REPLICATE_API_TOKEN`.

```bash
stickermotion animate character.png                          # automatic description, all targets
stickermotion animate character.png --prompt "The fox waves its raised paw twice"
stickermotion animate character.png --target telegram_sticker --target discord_emoji
stickermotion animate character.png --budget 0.10 --out out  # spend cap for this run, USD
stickermotion animate character.png --regenerate             # ignore the cached clip, pay again
stickermotion animate character.png --variants 3 --budget 0.20   # 3 seeds, previews only
stickermotion animate character.png --seed 1234567           # repeat a clip with a known seed
stickermotion animate character.png --emote-focus full       # no face crop for emotes
stickermotion animate character.png --emote-focus 150,200,550,420   # manual X,Y,W,H crop
stickermotion animate character.png --decor off              # remove decor from the result
stickermotion animate character.png --decor together         # keep decor in the video
stickermotion animate character.png --frames 97 --fps 24     # Wan num_frames 81..121, fps 5..30
stickermotion animate character.png --resolution 720p        # $0.11 instead of $0.05
stickermotion animate character.png --cut rembg              # needs pip install ".[rembg]"
```

Defaults: automatic description, all targets, 81 frames at 30 fps (2.7 s), 480p, `--emote-focus auto`, `--decor auto`, chroma-key cut, budget $0.10. The Wan price depends only on `--frames`, not on `--fps`.

Output in `out/<input name>/ai/`: the target files, `raw.mp4`, `model_input.png`, `prompt.txt`, `manifest.json` and `index.html`. With an emote focus the full-composition emotes are kept in `full/` for comparison. Raw clips are cached in `out/.cache/ai/`: a repeated run with the same input and prompt does not pay again. Every paid call and cache hit is logged in `out/costs.jsonl`.

With `--variants 2` or `3` each clip is stored in `out/<input name>/raw/<clip id>/` with `meta.json` and a Telegram sticker preview; the command prints the `reprocess` line for a full export of the chosen clip.

Exit codes: 0 ok, 1 a file failed validation, 2 bad arguments, 3 provider or budget error (the cost journal path is printed).

### `reprocess`: export an existing clip again (free)

```bash
stickermotion reprocess out/character/raw/<clip id>/raw.mp4 \
  --input character.png \
  --model-input out/character/raw/<clip id>/model_input.png \
  --out out --name character_take2
```

Cuts and exports a clip again without a new generation. With `--decor auto` (default) decor that was part of the video is tracked frame by frame and removed, and procedural decor from the original PNG is laid in its place. `--emote-focus stored` (default) reuses the zone saved in the clip's `meta.json`; `full`, `X,Y,W,H` or `auto` (one cheap vision call) are also accepted. `--target` limits the targets.

### `validate`: check a file

```bash
stickermotion validate out/character/ai/telegram_sticker.webm telegram_sticker
stickermotion validate out/character/wiggle/twitch_gif_28.gif twitch_gif --variant manual
```

Checks one file against the limits of a target. Exit code 1 when it does not fit.

### `serve`: web UI

```bash
stickermotion serve --host 127.0.0.1 --port 8000
```

Needs `APP_PASSWORD`; see [configuration.md](configuration.md).

### Helper scripts

```bash
python scripts/make_fixtures.py                    # regenerate synthetic test PNGs
python scripts/preview_presets.py character.png --out out/preview   # contact sheets of presets
python scripts/compare_quality.py grid.jpg A=a.webm B=b.webm        # frame grid and sharpness
```
