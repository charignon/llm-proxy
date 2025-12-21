# Build stage
FROM golang:1.22-alpine AS builder

# Install build dependencies for CGO (required for sqlite3)
RUN apk add --no-cache \
    gcc \
    musl-dev \
    sqlite-dev

# Set working directory
WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary with CGO enabled
# Use -ldflags to reduce binary size
RUN CGO_ENABLED=1 GOOS=linux go build \
    -ldflags="-w -s" \
    -o llm-proxy \
    .

# Runtime stage - minimal Alpine
FROM alpine:latest

# Install only runtime dependencies
# sqlite-libs is needed for sqlite3 CGO bindings
# wget is needed for healthcheck (very small, ~50KB)
# su-exec is needed for switching users at runtime (very small, ~10KB)
RUN apk add --no-cache \
    ca-certificates \
    sqlite-libs \
    wget \
    su-exec \
    shadow \
    && rm -rf /var/cache/apk/*

# Create data directory (permissions set at runtime)
RUN mkdir -p /app/data

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/llm-proxy /app/llm-proxy

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Make scripts executable
RUN chmod +x /app/llm-proxy /usr/local/bin/docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Default command (executed as the runtime user)
CMD ["/app/llm-proxy"]

