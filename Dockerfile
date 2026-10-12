# StickerMotion web UI. Debian ffmpeg ships libvpx-vp9 and the apng encoder.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -hide_banner -encoders | grep -q libvpx-vp9 \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data && chown app:app /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"
CMD ["stickermotion", "serve", "--host", "0.0.0.0", "--port", "8000"]
