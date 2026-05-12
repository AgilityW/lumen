#!/bin/bash
# Lumen runner — strips proxy, loads .env, activates venv, runs python (unbuffered)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
cd "$(dirname "$0")"

# Load .env if present
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

source .venv/bin/activate
exec python3 -u "$@"
