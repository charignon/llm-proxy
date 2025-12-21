#!/bin/sh
set -e

# Default UID/GID if not provided
PUID=${PUID:-1000}
PGID=${PGID:-1000}

echo "Running as user ${PUID} and group ${PGID}"

# Ensure data directory exists and has correct permissions
mkdir -p /app/data
chown -R ${PUID}:${PGID} /app/data 2>/dev/null || true

/app/llm-proxy
