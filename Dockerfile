# syntax=docker/dockerfile:1

# ---- builder: install Python deps into a venv ----
# Isolated from the runtime stage so build-only tooling (a compiler, pip's
# own cache) never ends up in the final image (see docs/decisions.md #45).
FROM python:3.11-slim AS builder

WORKDIR /app

# build-essential: some transitive dependencies (e.g. under paddleocr/
# paddlepaddle) don't ship a manylinux wheel for every platform and fall
# back to a source build, which needs a C compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only the install-time metadata + source first, so this (multi-GB,
# slowest) layer is cache-hit on rebuilds that only touch scripts/tests/docs.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[detection,ocr,serving]"

# ---- runtime: slim base + the built venv + app source ----
FROM python:3.11-slim AS runtime

# opencv-python-headless still needs these two at import time despite
# being the "headless" build (libgl1: GL runtime several codec paths
# probe for; libglib2.0-0: glib, pulled in transitively by those codecs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY . .

# Bakes the plate-detector checkpoint into the image at build time (not
# fetched on first request) — requires the GitHub Release referenced by
# scripts/download_weights.py's defaults to exist; see docs/decisions.md
# #44/#45 for why it doesn't yet.
RUN python scripts/download_weights.py

EXPOSE 8000
CMD ["python", "scripts/run_web.py", "--host", "0.0.0.0"]
