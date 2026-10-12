# Troubleshooting

Messages in the web UI are in Ukrainian; the quoted text below helps you match them.

## E005: request flagged as sensitive

> "Модерація Replicate помилково позначила запит як чутливий (E005), гроші не списано."

Replicate's moderation sometimes flags harmless drawings, most often on the green model frame. Flagged requests are **not billed**.

What the service does on its own:

- **Description:** retried once with the character on a neutral grey background.
- **Decor check and emote focus:** retried once on grey; if it fails again, the step falls back (shape heuristic for decor, full composition for emotes) and the generation goes on.
- **Video generation:** not retried automatically.

What you can do: retry (moderation results vary between attempts); write the action description yourself instead of the automatic one; generate another variant.

## 402: insufficient credit

> "На рахунку Replicate закінчились кошти. Поповніть баланс і повторіть."

Your Replicate account has no credit left. The call is never retried, so nothing is charged. Top up the balance on [replicate.com](https://replicate.com/account/billing) and retry.

## 429: rate limit, queue

> "Replicate тимчасово обмежив частоту запитів. Спробуйте за хвилину."

Replicate limits how often an account can create predictions, stricter for accounts with low credit. The service:

- runs paid calls one at a time across all stickers; during a batch upload descriptions show "У черзі на опис" (in queue for description) and generations wait in the queue;
- retries a 429 a few times with a delay, cancelling any prediction of its own that Replicate created despite the 429, so you do not pay twice.

If the message still appears, wait a minute and retry. Adding credit to the account usually raises the limit.

## Budget messages

> "Денний ліміт витрат ... не дозволяє цей крок" / "Ліміт на одну генерацію ... не дозволяє цей крок"

The daily (`DAILY_BUDGET_USD`) or per-generation (`RUN_BUDGET_USD`) cap would be exceeded, so the paid call was not made. Wait for the next day or raise the variable and restart.

## A file is too big for a platform

Every exported file is fitted to 95% of the platform limit and validated. If a file is marked "поза лімітом" (over the limit):

- **Telegram custom emoji** is the tightest (about 64 KB in practice). Very detailed characters, many colours or fast motion make it hard to fit. Use a simpler, smaller motion, or regenerate.
- **Discord emoji** (256 KB) and **Telegram sticker** (256 KB) suffer from large moving areas. Keep the motion small; keep gesturing hands still.
- **Twitch** allows at most 60 frames; the exporter thins frames automatically.

If a platform rejects a file at publishing time ("Файл стікера завеликий для Telegram", "Файл завеликий для Discord"), the platform's real limit may have changed; please open an issue with the file size.

## Discord: no free slots

> "На сервері закінчились слоти емодзі." / "...слоти стікерів." / "немає вільних слотів"

The server has used all emoji or sticker slots of its boost level (see [publishing.md](publishing.md#slots-per-boost-level)). Delete old emoji or stickers in the server settings, or raise the boost level, then publish again. The settings page shows free slots.

Other Discord errors:

- "Боту бракує права Create/Manage Expressions" - give the bot's role the Manage Expressions permission.
- "Бот не має доступу до сервера" or "Сервер не знайдено" - check `DISCORD_GUILD_ID` and that the bot was invited to that server.
- "Discord не прийняв токен бота (401)" - check `DISCORD_BOT_TOKEN`, reset it in the Developer Portal if needed.

## Telegram: the bot cannot create the pack

> "Telegram не знає власника для цього бота. Напишіть боту /start з акаунта власника (TELEGRAM_OWNER_ID)..."
> "Бот не може звернутися до власника. Напишіть боту /start і повторіть."

Telegram does not let a bot create a pack for a user who never talked to it. Open the bot from the account whose ID is in `TELEGRAM_OWNER_ID`, send `/start`, then publish again. Also check that `TELEGRAM_OWNER_ID` is the numeric user ID, not a username.

Other Telegram errors:

- "Ім'я пака вже зайняте" - the pack name is taken; change `TELEGRAM_PACK_SLUG`.
- "У паку вже максимум стікерів" - the pack is full; use a new slug for a new pack.
- "Telegram не прийняв емодзі" - use 1-20 regular emoji.

## The web UI does not start

- `APP_PASSWORD is not set` - set it in `.env` or the environment.
- `... must be a number` - a budget or size variable has a non-numeric value.
- In Docker check `docker compose logs` and `docker compose ps` (the health check calls `/healthz`).

## Login works over HTTP but not over HTTPS (or the reverse)

`COOKIE_SECURE=1` makes the browser send the session cookie over HTTPS only. Use it behind HTTPS and leave it off for plain `http://127.0.0.1`.

## The generation was interrupted

If the service restarts during a generation, the sticker is marked as interrupted. It is not retried automatically, because a retry is paid; open the sticker and retry by hand.

## The decor moves with the head

The decor touches the character in the PNG, so it was treated as part of it, or the decor box was unchecked. Leave a transparent gap between the character and each decor element and keep "Анімувати декор окремо" on. For an existing clip, "Використати цей рух" with the box on re-cuts it with separate decor for free.

## The hand changes colour

Skin hue repair fixes most cases of skin drifting to lilac. If a gesturing hand still breaks, keep it still in the description and animate the face instead (see [prompting.md](prompting.md)).
