# LLM Proxy

A unified AI gateway that routes requests to multiple LLM providers (OpenAI, Anthropic, Ollama, Gemini) based on sensitivity and quality requirements.

## Features

- **Smart Routing**: Automatically routes to the best provider based on `sensitive` and `precision` flags
- **Privacy Controls**: Sensitive requests stay local (Ollama only)
- **Cost Tracking**: Logs all requests with token counts and costs
- **Response Caching**: Reduces costs and latency for repeated queries
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API
- **Web Dashboard**: Real-time analytics and request history
- **Multi-Modal**: Supports text, vision, speech-to-text, and text-to-speech

## Requirements

- Go 1.22 or later
- SQLite (included via go-sqlite3)
- (Optional) [Ollama](https://ollama.ai) for local models
- (Optional) API keys for cloud providers

## Installation

```bash
# Clone the repository
git clone https://github.com/charignon/llm-proxy.git
cd llm-proxy

# Build
go build -o llm-proxy .

# Run
./llm-proxy
```

## Configuration

Set environment variables or create key files:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP port |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `OLLAMA_HOST` | `localhost:11434` | Ollama server address |
| `DATA_DIR` | `./data` | Directory for SQLite database |

Alternatively, create files in the working directory:
- `openai_key.txt`
- `anthropic_key.txt`
- `gemini_key.txt`

## Usage

### Basic Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "usecase": "my-app",
    "sensitive": false,
    "precision": "medium"
  }'
```

### Parameters

- `model`: `"auto"` for smart routing, or specify a model directly
- `usecase`: Required identifier for your application (for analytics)
- `sensitive`: `true` = local only (Ollama), `false` = can use cloud
- `precision`: `"low"`, `"medium"`, `"high"`, or `"very_high"`

### Routing Table

| Sensitive | Precision | Provider | Model |
|-----------|-----------|----------|-------|
| false | very_high | Anthropic | claude-sonnet-4-5 |
| false | high | OpenAI | gpt-4o |
| false | medium | OpenAI | gpt-4o-mini |
| false | low | Ollama | mistral:7b |
| true | high | Ollama | llama3.3:70b |
| true | medium | Ollama | qwen3-vl:30b |
| true | low | Ollama | mistral:7b |

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard |
| `GET /analytics` | Analytics page |
| `GET /stats` | Performance stats |
| `GET /test` | Test playground |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /v1/messages` | Anthropic-compatible messages |
| `POST /v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `POST /v1/audio/speech` | Text-to-speech |

## Development

```bash
# Run locally
make run

# Run tests
make test

# Clean build artifacts
make clean
```

## License

MIT
