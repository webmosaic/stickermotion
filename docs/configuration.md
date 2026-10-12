# Configuration

StickerMotion is configured only through environment variables. Secrets never live in code or in the repository.

The web UI and the CLI also read a `.env` file (`KEY=value` lines, `#` for comments) from the current directory. A variable that is already set in the environment wins over `.env`. Start from [`.env.example`](../.env.example):

```bash
cp .env.example .env
```

An empty value means "not set": the default applies, or the feature is turned off.

## Web UI and generation

| Variable | Purpose | Required | Default | Example |
|---|---|---|---|---|
| `APP_PASSWORD` | The only login of the web UI. The service does not start without it. | yes (web UI) | - | `long-random-passphrase` |
| `REPLICATE_API_TOKEN` | Replicate API token for the vision model and Wan 2.2. Without it the UI opens, but describing and generating fail with a clear message. The CLI `animate` command needs it too. | yes (AI mode) | - | `<your Replicate token>` |
| `DATA_DIR` | Directory for uploaded PNGs, generated files, the SQLite database `app.db`, the clip cache `.cache/ai/`, stored raw clips and the cost journal `costs.jsonl`. Created if missing. Docker Compose sets it to `/data` (a volume). | no | `data` | `/var/lib/stickermotion` |
| `DAILY_BUDGET_USD` | Spend cap per calendar day (server time) across all generations and descriptions. A paid call that would exceed it is not made; the UI shows what was spent today. | no | `1` | `2.5` |
| `RUN_BUDGET_USD` | Spend cap for one generation: description, decor check, emote focus and the video clip together. With 2-3 variants it applies to each variant. The effective limit of a call is the smaller of this and what is left of the daily budget. | no | `0.10` | `0.15` |
| `MAX_UPLOAD_MB` | Maximum size of an uploaded PNG, in MB. Keep nginx `client_max_body_size` a bit above it. | no | `10` | `20` |
| `SECRET_KEY` | Key that signs the session cookie. When empty, a random key is generated on first start and kept in `DATA_DIR/.secret_key` (mode 600), so sessions survive restarts. Set it explicitly if `DATA_DIR` is not persistent. | no | generated | `<64 random characters>` |
| `COOKIE_SECURE` | `1`, `true`, `yes` or `on` marks the session cookie as Secure (sent over HTTPS only). Turn it on behind HTTPS; leave it off for plain `http://127.0.0.1`. | no | off | `1` |
| `STICKERMOTION_PORT` | Docker Compose only: host port bound on `127.0.0.1`. Without Docker use `stickermotion serve --port`. | no | `8000` | `8080` |

The session cookie lives for 30 days. Logging out clears it.

### Budgets

Every paid call goes through a ledger that checks the budget **before** the call, using the published price as an upper bound, and books the real cost afterwards from Replicate's metrics. Typical costs:

| Step | Model | Cost |
|---|---|---|
| Action description, translation | `openai/gpt-5-nano` | about $0.0002 |
| Decor check, emote focus | `openai/gpt-5-nano` | about $0.0002 each |
| Video clip, 480p, 81 frames | `wan-video/wan-2.2-i2v-fast` | $0.05 |
| Video clip, 720p (CLI only) | `wan-video/wan-2.2-i2v-fast` | $0.11 |

The default `RUN_BUDGET_USD=0.10` covers one 480p clip with all helper calls. Raise it only if you use 720p or longer clips in the CLI.

## Publishing

All publishing variables are optional. A platform with missing variables is shown as "not configured" on the settings page, and its publish button lists what is missing. Tokens are never shown in the UI, logs or the database.

| Variable | Purpose | Required | Default | Example |
|---|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Token of the bot that creates and fills the packs, from @BotFather. | for Telegram | - | `123456789:AA...` |
| `TELEGRAM_OWNER_ID` | Numeric Telegram user ID of the pack owner. Telegram creates packs on behalf of a user; that user must have sent `/start` to the bot once. | for Telegram | - | `123456789` |
| `TELEGRAM_PACK_TITLE` | Pack title shown in Telegram. The emoji pack gets the same title with a suffix. | no | `StickerMotion` | `My Pack` |
| `TELEGRAM_PACK_SLUG` | Base of the pack link name. The sticker pack is `<slug>_by_<bot username>`, the emoji pack `<slug>_emoji_by_<bot username>` (Telegram requires the `_by_<bot>` suffix). Latin letters, digits and underscores; change it if the name is taken. | no | `stickermotion` | `mypack` |
| `DISCORD_BOT_TOKEN` | Token of a Discord bot that is a member of your server with the Manage Expressions permission. | for Discord | - | `<bot token>` |
| `DISCORD_GUILD_ID` | Numeric ID of the server (guild) that receives emoji and stickers. | for Discord | - | `112233445566778899` |
| `TWITCH_CHANNEL` | Channel login, only used for the link to the emote page of the creator dashboard in the UI and the archive instructions. Nothing is sent to Twitch. | no | - | `mychannel` |

See [publishing.md](publishing.md) for how to create the bots and get the IDs.

## Command line

The CLI reads `REPLICATE_API_TOKEN` (environment, then `.env`). Spend limits are set per run with `--budget` (default `0.10`); the daily budget and the other web variables do not apply. Output goes to `--out` (default `out/`), the cost journal to `out/costs.jsonl`.

## Files under `DATA_DIR`

```
DATA_DIR/
  app.db             SQLite: stickers, descriptions, costs, publication state
  costs.jsonl        one JSON line per paid call or cache hit (no tokens)
  .secret_key        generated cookie signing key, when SECRET_KEY is not set
  .cache/ai/         raw clips keyed by input frame, prompt and model parameters
  stickers/<id>/     input PNG, outputs per platform, raw/ with the last 5 clips
```

Deleting a sticker in the UI removes its files; whether it is also removed from Telegram and Discord is up to you in the delete dialog.
