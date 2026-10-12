# Publishing

From the result page a sticker can go straight to Telegram and Discord. Twitch has no upload API, so the service prepares the files and you upload them by hand.

The service makes **outgoing calls only**: it never reads a Telegram bot's updates and never connects to the Discord gateway. You can therefore reuse a bot that already runs elsewhere (a bot that polls updates or uses a webhook keeps working).

Publishing state is stored per sticker and platform. Publishing again does not create a duplicate, and "Прибрати з платформи" (remove from platform) deletes the item there. The settings page shows what is connected; tokens are never displayed.

## Telegram

Telegram creates sticker packs on behalf of a **user** (the pack owner) through a **bot**. The pack link name always ends with `_by_<bot username>`.

### 1. Create a bot

1. Open [@BotFather](https://t.me/BotFather) and send `/newbot`.
2. Choose a name and a username ending in `bot`.
3. Copy the token into `TELEGRAM_BOT_TOKEN`.

No privacy settings or webhooks are needed.

### 2. Find the owner's user ID

The pack owner is the Telegram account that will own and manage the packs. Get its numeric ID, for example by sending a message to [@userinfobot](https://t.me/userinfobot), and put it into `TELEGRAM_OWNER_ID`.

### 3. The owner writes /start to the bot

**Important:** from the owner's account, open the bot and send `/start` once. Telegram does not allow a bot to create a pack for a user who has never talked to it; without this you get "Telegram не знає власника для цього бота" (Telegram does not know the owner for this bot).

### 4. Pack names

- `TELEGRAM_PACK_TITLE`: the title users see (default `StickerMotion`).
- `TELEGRAM_PACK_SLUG`: the link base (default `stickermotion`). The video sticker pack becomes `t.me/addstickers/<slug>_by_<bot>`, the custom emoji pack `t.me/addemoji/<slug>_emoji_by_<bot>`.

The first published sticker creates the pack; later ones are added to it. If the name is taken by someone else, change the slug.

### What is published

- **Video sticker:** the 512 px WEBM VP9 with alpha, up to 3 seconds and 256 KB, with the emoji you choose (1-20, suggested from the description).
- **Custom emoji (optional):** tick "Також додати в пак кастомних емодзі" to also add the 100 px emoji to the custom emoji pack. Custom emoji packs can be added by anyone; using custom emoji in messages requires Telegram Premium on the user's side.

Limits: up to 120 stickers per video sticker pack and 200 per emoji pack (Telegram may change these).

## Discord

Discord emoji and stickers are uploaded to one **server** (guild) by a **bot** over the REST API.

### 1. Create a bot

1. Open the [Discord Developer Portal](https://discord.com/developers/applications), create an application, open **Bot** and press **Reset Token**. Copy the token into `DISCORD_BOT_TOKEN`.
2. No privileged gateway intents are needed.

### 2. Invite it with the right permission

In **OAuth2 > URL Generator** pick the scope `bot` and the permissions **Manage Expressions** (on newer servers also **Create Expressions**). Open the generated URL and add the bot to your server. Alternatively give the bot's role that permission in the server settings.

Without it uploads fail with "Боту бракує права Create/Manage Expressions на сервері" (the bot lacks the permission).

### 3. Server ID

In Discord turn on **User Settings > Advanced > Developer Mode**, right-click the server icon and choose **Copy Server ID**. Put it into `DISCORD_GUILD_ID`.

### What is published

- **Animated emoji:** 128 px animated WEBP (retried as GIF if Discord refuses WEBP), up to 256 KB. Name: 2-32 characters, Latin letters, digits and underscores; a taken name gets a `_2` suffix.
- **Sticker:** 320 px APNG, up to 512 KB and 5 seconds, with a related emoji as its tag. Name: 2-30 characters.

Using animated emoji in other servers requires Discord Nitro on the user's side; that is a Discord rule, not a limit of the service.

### Slots per boost level

Every server has a fixed number of slots. Animated and static emoji have separate pools of the same size; stickers have their own pool.

| Boost level | Animated emoji | Stickers |
|---|---|---|
| 0 (no boost) | 50 | 5 |
| 1 | 100 | 15 |
| 2 | 150 | 30 |
| 3 | 250 | 60 |

The settings page shows the server's boost level and free slots. When slots run out, publishing fails with a clear message: delete old items or raise the boost level.

## Twitch

Twitch has no public API for emote uploads, so nothing is sent to Twitch.

1. On the result page press **"Підготувати для Twitch"** (prepare for Twitch). You get a ZIP with:
   - `manual/twitch_gif_28.gif`, `twitch_gif_56.gif`, `twitch_gif_112.gif` for the manual mode;
   - `auto/twitch_gif_224.gif` for the auto mode;
   - `README.txt` with the steps below and the link to your emote page.
2. Open the emote page of the creator dashboard: `https://dashboard.twitch.tv/u/<your channel>/viewer-rewards/emotes` (set `TWITCH_CHANNEL` to get a direct link in the UI).
3. Add an emote and choose an **animated** emote.
4. **Auto-resize** (recommended): upload the single file from `auto/`. Twitch resizes it; the file is up to 1 MB and 60 frames.
5. **Manual:** upload the three files from `manual/` into the 28, 56 and 112 slots (each up to 512 KB).
6. Set the emote code, the tier and save.

Animated emotes are available to Affiliates and Partners. All GIFs already pass Twitch's file limits (at most 60 frames, 1 MB in auto mode, 512 KB per file in manual mode). Twitch also forbids flashing more than 3 times per second; that depends on the motion itself, so check it before uploading.
