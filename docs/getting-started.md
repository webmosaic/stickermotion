# Getting started

This guide covers three ways to run StickerMotion: locally with Python, with Docker Compose, and on a VPS behind nginx with HTTPS.

## Requirements

Either:

- **Docker** with Docker Compose v2 (the image includes Python and ffmpeg), or
- **Python 3.12+** and **ffmpeg** built with the `libvpx-vp9` encoder (VP9 with alpha for Telegram and OBS) and the `apng` encoder. Debian and Ubuntu packages have both. Check with:

  ```bash
  ffmpeg -hide_banner -encoders | grep -E "libvpx-vp9|apng"
  ```

For AI animation you also need a [Replicate](https://replicate.com) account with credit and an API token. The procedural presets (`stickermotion render`) work without it.

No GPU is needed: generation runs on Replicate, everything else (keying, looping, encoding) runs on the CPU. A small VPS with 2 vCPU and 2-4 GB of RAM is enough.

## Local run (Python)

```bash
git clone https://github.com/webmosaic/stickermotion.git stickermotion
cd stickermotion
python3.12 -m venv .venv
. .venv/bin/activate
pip install .            # or: pip install -e ".[dev]" for development
cp .env.example .env
```

Edit `.env` and set at least `APP_PASSWORD` and `REPLICATE_API_TOKEN` (see [configuration.md](configuration.md)). Then start the web UI:

```bash
stickermotion serve                      # http://127.0.0.1:8000
stickermotion serve --host 0.0.0.0 --port 8080
```

`serve` reads `.env` from the current directory; variables already set in the environment win. Data goes to `DATA_DIR` (default `./data`).

The CLI works from the same environment, see [usage.md](usage.md#command-line).

## Docker Compose

```bash
git clone https://github.com/webmosaic/stickermotion.git stickermotion
cd stickermotion
cp .env.example .env     # set APP_PASSWORD and REPLICATE_API_TOKEN
docker compose up -d --build
docker compose ps        # wait for "healthy"
docker compose logs -f   # follow the logs
```

What the bundled `docker-compose.yml` does:

- builds the image from the `Dockerfile` (Python 3.12 slim plus Debian ffmpeg), runs as a non-root user;
- reads all settings from `.env`;
- stores data in the named volume `stickermotion-data` mounted at `/data` (`DATA_DIR` is set to `/data` inside the container);
- publishes the port on `127.0.0.1` only (`STICKERMOTION_PORT`, default 8000), so the service is not exposed to the network by itself;
- checks `/healthz` every 30 seconds and restarts the container unless stopped.

Back up the volume to keep your stickers and history:

```bash
docker run --rm -v stickermotion-data:/data -v "$PWD":/backup busybox \
  tar czf /backup/stickermotion-data.tgz -C /data .
```

## VPS behind nginx with HTTPS

The service has a single password and no rate limiting of its own at the HTTP level, so always serve it over HTTPS. A typical setup on a Debian or Ubuntu VPS:

1. Install Docker, nginx and certbot:

   ```bash
   sudo apt install nginx certbot python3-certbot-nginx
   # Docker: follow https://docs.docker.com/engine/install/
   ```

2. Put the project in a neutral place, for example `/opt/stickermotion`, create `.env` and set `COOKIE_SECURE=1`, then start it with `docker compose up -d --build`.

3. Point a DNS record (for example `stickers.example.com`) to the server.

4. Create `/etc/nginx/sites-available/stickermotion`:

   ```nginx
   server {
       listen 80;
       listen [::]:80;
       server_name stickers.example.com;
       return 301 https://$host$request_uri;
   }

   server {
       listen 443 ssl;
       listen [::]:443 ssl;
       http2 on;
       server_name stickers.example.com;

       ssl_certificate     /etc/letsencrypt/live/stickers.example.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/stickers.example.com/privkey.pem;

       # Must be at least MAX_UPLOAD_MB.
       client_max_body_size 12m;

       add_header X-Content-Type-Options nosniff always;
       add_header Referrer-Policy same-origin always;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_read_timeout 120s;
       }
   }
   ```

5. Get a certificate and enable the site. The simplest way is to start with only the port 80 block, run certbot, then add the 443 block:

   ```bash
   sudo ln -s /etc/nginx/sites-available/stickermotion /etc/nginx/sites-enabled/
   sudo certbot --nginx -d stickers.example.com
   sudo nginx -t && sudo systemctl reload nginx
   ```

6. Open `https://stickers.example.com` and log in.

The service does not throttle login attempts itself. Use a long random `APP_PASSWORD`, and optionally let nginx slow down guessing:

```nginx
# in the http {} block, e.g. /etc/nginx/conf.d/limits.conf
limit_req_zone $binary_remote_addr zone=sm_login:10m rate=10r/m;

# inside the 443 server {} block, before "location /"
location = /login {
    limit_req zone=sm_login burst=5 nodelay;
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Generations run in the background on the server, so long requests are not held open by nginx: the page polls the status and you can close the tab while a sticker is being made.

## Updating

With Docker:

```bash
cd /opt/stickermotion
git pull
docker compose up -d --build
docker image prune -f
```

The data volume is kept. The SQLite database is migrated automatically on start.

With Python:

```bash
git pull
. .venv/bin/activate
pip install .
# restart `stickermotion serve` (or your systemd unit)
```

A generation that was running when the service restarted is marked as interrupted; it is not retried automatically, because it would cost money. Retry it by hand from its page.
