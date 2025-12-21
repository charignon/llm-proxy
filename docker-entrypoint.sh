#!/bin/sh
set -e

# Default UID/GID if not provided
PUID=${PUID:-1000}
PGID=${PGID:-1000}

echo "Running as user ${PUID} and group ${PGID}"

# Ensure we're running as root for user/group creation
if [ "$(id -u)" != "0" ]; then
    echo "Error: Entrypoint must run as root to create user/group" >&2
    echo "Do not use --user flag or USER directive. The entrypoint will switch to the non-root user." >&2
    exit 1
fi

# Create group "llm" if it doesn't exist
if ! getent group llm > /dev/null 2>&1; then
    addgroup -g ${PGID} -S llm
fi

# Create user "llm" if it doesn't exist
if ! getent passwd llm > /dev/null 2>&1; then
    adduser -u ${PUID} -G llm -S -D -h /app llm
fi

# Ensure data directory exists and has correct permissions
mkdir -p /app/data
chown -R ${PUID}:${PGID} /app/data 2>/dev/null || true

# Ensure we have a command to execute
if [ $# -eq 0 ]; then
    # Default to running llm-proxy if no command provided
    set -- /app/llm-proxy
fi

# Switch to the user and execute the application
# su-exec syntax: su-exec user command [args...]
exec su-exec llm "$@"
