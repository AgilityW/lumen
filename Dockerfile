# Lumen — Book Deconstruction Engine
#
# Multi-stage build: slim image with PDF, EPUB, and Markdown support.

# ── Builder stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime dependencies for pymupdf (MuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source
COPY . .

# Install package itself
RUN pip install --no-cache-dir -e . 2>&1 | tail -3

# Default entry point
ENTRYPOINT ["lumen"]
CMD ["--help"]
