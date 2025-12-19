# LLM Proxy

A unified AI gateway that routes requests to multiple LLM providers (OpenAI, Anthropic, Ollama, Gemini, llama.cpp) based on sensitivity and quality requirements.

## Features

- **Smart Routing**: Automatically routes to the best provider based on `sensitive` and `precision` flags
- **Privacy Controls**: Sensitive requests stay local (Ollama only)
- **Cost Tracking**: Logs all requests with token counts and costs
- **Response Caching**: Reduces costs and latency for repeated queries
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API
- **Web Dashboard**: Real-time analytics and request history
- **Multi-Modal**: Supports text, vision, speech-to-text, and text-to-speech

## Quick Start (Local Development)

### Prerequisites

- Go 1.22 or later
- (Optional) [Ollama](https://ollama.ai) for local models
- (Optional) API keys for cloud providers

### Build and Run

```bash
# Clone the repository
git clone https://github.com/charignon/llm-proxy.git
cd llm-proxy

# Build
go build -o llm-proxy .

# Run (default port 8080)
./llm-proxy
```

Open http://localhost:8080 to see the dashboard.

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP port to listen on |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `OLLAMA_HOST` | `localhost:11434` | Ollama server address |
| `LLAMACPP_HOST` | - | llama.cpp server address (e.g., `localhost:8081`) |
| `DATA_DIR` | `./data` | Directory for SQLite database |
| `WHISPER_SERVER_URL` | `http://localhost:8890` | Whisper STT server |
| `TTS_SERVER_URL` | `http://localhost:7788` | TTS server |

### API Keys

You can provide API keys either as environment variables or as files in the working directory:
- `openai_key.txt`
- `anthropic_key.txt`
- `gemini_key.txt`

## Deployment

### Option 1: Simple Manual Deployment

```bash
# On your dev machine: build for Linux
GOOS=linux GOARCH=amd64 go build -o llm-proxy .

# Copy to server
scp llm-proxy yourserver:/home/youruser/llm-proxy/

# SSH in and run
ssh yourserver
cd /home/youruser/llm-proxy
PORT=8080 OPENAI_API_KEY=sk-xxx ./llm-proxy
```

### Option 2: systemd Service (Linux)

1. Copy the binary to the server:
```bash
GOOS=linux GOARCH=amd64 go build -o llm-proxy .
scp llm-proxy yourserver:/usr/local/bin/
```

2. Create `/etc/systemd/system/llm-proxy.service`:
```ini
[Unit]
Description=LLM Proxy
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/llm-proxy
Environment=PORT=8080
Environment=OPENAI_API_KEY=sk-xxx
Environment=ANTHROPIC_API_KEY=sk-ant-xxx
Environment=OLLAMA_HOST=localhost:11434
Environment=DATA_DIR=/home/youruser/llm-proxy/data
ExecStart=/usr/local/bin/llm-proxy
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable llm-proxy
sudo systemctl start llm-proxy
```

4. Check status:
```bash
sudo systemctl status llm-proxy
journalctl -u llm-proxy -f
```

### Option 3: launchd Service (macOS)

1. Build and copy the binary:
```bash
go build -o llm-proxy .
sudo cp llm-proxy /usr/local/bin/
```

2. Create `/Library/LaunchDaemons/com.llm-proxy.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.llm-proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/llm-proxy</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/var/lib/llm-proxy</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PORT</key>
        <string>8080</string>
        <key>OPENAI_API_KEY</key>
        <string>sk-xxx</string>
        <key>DATA_DIR</key>
        <string>/var/lib/llm-proxy/data</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/llm-proxy.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/llm-proxy.err</string>
</dict>
</plist>
```

3. Load the service:
```bash
sudo mkdir -p /var/lib/llm-proxy/data
sudo launchctl load /Library/LaunchDaemons/com.llm-proxy.plist
```

## API Usage

### Chat Completion (OpenAI-compatible)

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | `"auto"` for smart routing, or a specific model name |
| `messages` | array | Standard OpenAI message format |
| `usecase` | string | Required. Application identifier for analytics |
| `sensitive` | bool | `true` = local only (Ollama), `false` = can use cloud |
| `precision` | string | `"low"`, `"medium"`, `"high"`, or `"very_high"` |

### Smart Routing Table

| Sensitive | Precision | Provider | Model |
|-----------|-----------|----------|-------|
| false | very_high | Anthropic | claude-sonnet-4-5 |
| false | high | OpenAI | gpt-4o |
| false | medium | OpenAI | gpt-4o-mini |
| false | low | Ollama | mistral:7b |
| true | high | Ollama | llama3.3:70b |
| true | medium | Ollama | qwen3-vl:30b |
| true | low | Ollama | mistral:7b |

### Direct Model Access

You can also specify models directly:

```bash
# Use a specific OpenAI model
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "gpt-4o", "messages": [...]}'

# Use Ollama
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "ollama/llama3.3:70b", "messages": [...]}'
```

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

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Prometheus Metrics

Available at `/metrics` - includes:
- Request counts by provider/model
- Latency histograms
- Token counts
- Error rates

## Development

```bash
# Run locally with hot reload (using make)
make run

# Test the chat endpoint
make test

# Clean build artifacts
make clean
```

## Troubleshooting

### "Connection refused" to Ollama

Make sure Ollama is running:
```bash
ollama serve
```

### No response from cloud providers

Check your API keys are set correctly:
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Database errors

The SQLite database is created automatically in `DATA_DIR`. Make sure the directory exists and is writable:
```bash
mkdir -p ./data
```

## License

MIT
