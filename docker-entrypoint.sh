#!/bin/sh
set -e

# Default UID/GID if not provided
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Check if group with GID exists, if not create it
if ! getent group ${PGID} > /dev/null 2>&1; then
    # Try to create group with name llmproxy
    if ! addgroup -g ${PGID} -S llmproxy 2>/dev/null; then
        # If that fails, create with a numeric name
        addgroup -g ${PGID} -S "g${PGID}" 2>/dev/null || true
    fi
fi

# Get the group name (might be llmproxy, g${PGID}, or existing group)
GROUP_NAME=$(getent group ${PGID} | cut -d: -f1)

# Check if user with UID exists, if not create it
if ! getent passwd ${PUID} > /dev/null 2>&1; then
    # Try to create user with name llmproxy
    if ! adduser -u ${PUID} -G ${GROUP_NAME} -S -D -h /app llmproxy 2>/dev/null; then
        # If that fails, create with a numeric name
        adduser -u ${PUID} -G ${GROUP_NAME} -S -D -h /app "u${PUID}" 2>/dev/null || true
    fi
fi

# Get the username (might be llmproxy, u${PUID}, or existing user)
USER_NAME=$(getent passwd ${PUID} | cut -d: -f1)

# Ensure data directory exists and has correct permissions
mkdir -p /app/data
chown -R ${PUID}:${PGID} /app/data 2>/dev/null || true

# Switch to the user and execute the application
exec su-exec ${USER_NAME} "$@"

