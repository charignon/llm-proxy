# Docker Usage

Alpine-based image (~25-35MB) with runtime UID/GID support.

## Build

```bash
docker build -t llm-proxy .
```

## Run

```bash
docker run -d \
  --name llm-proxy \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -e PUID=$(id -u) \
  -e PGID=$(id -g) \
  -e OPENAI_API_KEY=sk-xxx \
  -e DATA_DIR=/app/data \
  llm-proxy
```

**Environment Variables:**
- `PUID` / `PGID`: User/group IDs (default: 1000/1000)
- `PORT`: HTTP port (default: 8080)
- `DATA_DIR`: Data directory (default: /app/data)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`: API keys
- `OLLAMA_HOST`: Ollama server (default: localhost:11434)

## Docker Compose

```yaml
version: '3.8'
services:
  llm-proxy:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```
