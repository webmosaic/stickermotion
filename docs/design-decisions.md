# Design decisions

A short record of the key decisions behind StickerMotion and why they were made. Numbers match the references (`ADR-0xx`) in code comments.

## ADR-001. One master animation, separate exporters

Twitch, Discord, Telegram and OBS need different formats and limits. The engine renders one master animation (RGBA frames at high resolution) and every platform gets an exporter with a common interface. A new platform is one new exporter; the engine does not change.

## ADR-002. Two classes of motion: emote and sticker

Emotes are shown at 28 px, stickers at 160-512 px; the same motion does not read equally well at both sizes. Each preset takes a class that controls amplitude, detail and speed.

## ADR-003. Procedural engine first (superseded by ADR-010)

AI animation is more expensive, slower and less predictable, so the first versions used procedural presets only, behind the same `MasterAnimation` contract. ADR-010 later made AI the main engine; the presets remain as a fallback.

## ADR-004. Python

Python 3.12 with FastAPI and Pydantic: the best ecosystem for image processing, background removal and AI APIs.

## ADR-005. Single user, PNG with transparency as input

The service is meant for one creator's own characters, drawn with a transparent background. No accounts, roles or billing. Preprocessing works with the existing alpha; background removal is only an optional fallback (`rembg` extra). Full transparency in every export is mandatory.

## ADR-006. CPU-only server

The service runs in Docker on an ordinary VPS without a GPU. Everything heavy that needs a GPU (video generation) goes through an external API; keying, looping and encoding run on the CPU.

## ADR-007. Weight units and recommended sizes

1 KB = 1024 bytes. Recommended sizes (such as 128 px for Discord emoji) produce validator warnings; hard limits are errors.

## ADR-008. Shared alpha utilities and thresholds

Premultiply, unpremultiply and resize live in one module (`core/alpha.py`). The "transparent pixel" threshold (alpha < 128) and the GIF binarization threshold are shared constants, so presets, exporters and the validator treat transparency the same way.

## ADR-009. Dark matte for Twitch GIFs

GIF has binary transparency, so soft edges need a background colour. The default is `#18181B`, the Twitch dark chat; on the light theme the edge looks 1 px darker.

## ADR-010. AI animation as the main engine

Moving the whole picture (procedural presets) cannot make a character yawn or wave. The main path is: action description (vision model or by hand) -> image-to-video on a flat background -> cut back to transparency -> loop -> the existing exporters. Procedural presets stay as a free fallback. Target cost: under $0.10 per sticker, hard cap below $1, so the work stays at 480p and 2-3 seconds.

## ADR-011. Video model: Wan 2.2 i2v fast

Six image-to-video models on Replicate were compared on the same character. Wan 2.2 i2v fast at 480p was chosen: good style preservation, first frame = last frame for loops, chroma-key cut on green, $0.05 per clip. Seedance 1.5 Pro has livelier motion but drifts the face style; Hailuo changed the background and redrew the character. A raised arm reached the top edge in early tests, so framing leaves headroom when the action needs it.

## ADR-012. 95% weight target

The fit loop aims at 95% of the platform limit in both AI and procedural modes. Files do not sit on the edge of a limit, and the quality difference is invisible.

## ADR-013. Face focus for emotes by default

Emote targets get `--emote-focus auto`: the vision model returns the key zone, with a fallback to the full composition. Stickers and OBS always use the full composition. At 28 px the face takes about 11 px instead of 4-5.

## ADR-014. Web UI instead of a chat bot

A simple password-protected web page: upload -> free automatic description (editable) -> paid generation -> previews on dark and light backgrounds -> downloads per platform -> history. Telegram and Discord are only destinations for stickers, not the control interface.

## ADR-015. Default generation parameters

Wan generates 81 frames played at 30 fps: a 2.7 s clip at the same $0.05 price that fits Telegram's 3 s without time compression. The fixed prompt requirements include keeping skin and hand colours, because Wan recoloured a moving fist.

## ADR-016. Description on grey after a moderation flag

The vision model on Replicate sometimes returned a moderation error (E005) for a frame on green while the same character on grey passed. Moderation errors are not billed. On E005 the description is retried once on grey; video generation stays on green. The decor and focus steps follow the same rule and fall back instead of failing the run.

## ADR-017. Publishing through bots, outgoing calls only

- **Telegram:** a bot creates and fills a video sticker pack (and optionally a custom emoji pack) through the Bot API on behalf of the pack owner.
- **Discord:** a bot uploads an animated emoji and a sticker to one server over REST, without a gateway connection.
- **Twitch:** no public upload API, so the service prepares an archive with files and instructions.

The service never reads bot updates and never connects to the Discord gateway, so it does not conflict with other software already using the same bots. Tokens stay in environment variables; publication state is stored per sticker and platform, and publishing again does not create a duplicate.

## ADR-018. Decor as separate layers, 480p by default

Video models treat a moon and stars next to the character as part of one scene and drag them along with the head. Detached decor is taken out of the model input and animated procedurally from the original pixels, independent of the character. Generation stays at 480p: 720p costs $0.11 and gives no gain at 512 px. Sharpness was improved instead with tighter framing, better downscaling and encoding (Laplacian variance about 2.9x higher).

## Positive-only prompts

Wan 2.2 fast has no negative prompt and reads named actions as cues even when negated. The fixed prompt requirements and the instruction for the vision model are worded positively ("the mouth keeps its size and shape from the drawing") and never name mouth actions to avoid. See [prompting.md](prompting.md).

## Variants and reusing clips

Because results vary a lot between seeds, the service can generate 2-3 variants of one description and lets the user pick. Every raw clip is stored with its model input and parameters; re-exporting a stored clip (for example with separate decor) costs nothing.

## Paid calls one at a time

Replicate rate-limits prediction creation, and a 429 can still create a prediction. All paid calls are serialized by one lock, and after a 429 only the predictions created by the same run are cancelled, never those of a neighbouring run.
