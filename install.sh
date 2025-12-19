#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions must be defined before use
install_macos_service() {
    echo -e "\n${YELLOW}Installing launchd service...${NC}"

    INSTALL_DIR="$HOME/llm-proxy"
    PLIST_PATH="$HOME/Library/LaunchAgents/com.llm-proxy.plist"

    # Copy files to install directory
    mkdir -p "$INSTALL_DIR/data"
    cp llm-proxy "$INSTALL_DIR/"
    cp .env "$INSTALL_DIR/" 2>/dev/null || true

    # Source env file for values
    if [ -f .env ]; then
        set -a; source .env; set +a
    fi

    # Create plist
    cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.llm-proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>$INSTALL_DIR/llm-proxy</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PORT</key>
        <string>${PORT:-8080}</string>
        <key>OLLAMA_HOST</key>
        <string>${OLLAMA_HOST:-localhost:11434}</string>
        <key>DATA_DIR</key>
        <string>$INSTALL_DIR/data</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/data/llm-proxy.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/data/llm-proxy.err</string>
</dict>
</plist>
EOF

    # Load service
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    launchctl load "$PLIST_PATH"

    echo -e "  Installed to: ${GREEN}$INSTALL_DIR${NC}"
    echo -e "  Service: ${GREEN}com.llm-proxy${NC}"
    echo
    echo "Commands:"
    echo -e "  Stop:    ${BLUE}launchctl unload $PLIST_PATH${NC}"
    echo -e "  Start:   ${BLUE}launchctl load $PLIST_PATH${NC}"
    echo -e "  Logs:    ${BLUE}tail -f $INSTALL_DIR/data/llm-proxy.log${NC}"
    echo
    echo -e "${GREEN}Service started!${NC} Open: http://localhost:${PORT:-8080}"
}

install_linux_service() {
    echo -e "\n${YELLOW}Installing systemd service...${NC}"

    INSTALL_DIR="$HOME/llm-proxy"
    SERVICE_PATH="$HOME/.config/systemd/user/llm-proxy.service"

    # Copy files
    mkdir -p "$INSTALL_DIR/data"
    mkdir -p "$(dirname "$SERVICE_PATH")"
    cp llm-proxy "$INSTALL_DIR/"
    cp .env "$INSTALL_DIR/" 2>/dev/null || true

    # Source env file for values
    if [ -f .env ]; then
        set -a; source .env; set +a
    fi

    # Create service file
    cat > "$SERVICE_PATH" << EOF
[Unit]
Description=LLM Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
Environment=PORT=${PORT:-8080}
Environment=OLLAMA_HOST=${OLLAMA_HOST:-localhost:11434}
Environment=DATA_DIR=$INSTALL_DIR/data
ExecStart=$INSTALL_DIR/llm-proxy
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

    # Enable and start
    systemctl --user daemon-reload
    systemctl --user enable llm-proxy
    systemctl --user start llm-proxy

    echo -e "  Installed to: ${GREEN}$INSTALL_DIR${NC}"
    echo -e "  Service: ${GREEN}llm-proxy (user)${NC}"
    echo
    echo "Commands:"
    echo -e "  Status:  ${BLUE}systemctl --user status llm-proxy${NC}"
    echo -e "  Stop:    ${BLUE}systemctl --user stop llm-proxy${NC}"
    echo -e "  Start:   ${BLUE}systemctl --user start llm-proxy${NC}"
    echo -e "  Logs:    ${BLUE}journalctl --user -u llm-proxy -f${NC}"
    echo
    echo -e "${GREEN}Service started!${NC} Open: http://localhost:${PORT:-8080}"
}

# Main script starts here
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  LLM Proxy Installer${NC}"
echo -e "${BLUE}================================${NC}"
echo

# Detect OS
OS=$(uname -s)
case "$OS" in
    Darwin) OS_TYPE="macos" ;;
    Linux)  OS_TYPE="linux" ;;
    *)      echo -e "${RED}Unsupported OS: $OS${NC}"; exit 1 ;;
esac
echo -e "Detected OS: ${GREEN}$OS_TYPE${NC}"

# Check for Go
echo -e "\n${YELLOW}Checking prerequisites...${NC}"
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | grep -oE 'go[0-9]+\.[0-9]+' | sed 's/go//')
    MAJOR=$(echo "$GO_VERSION" | cut -d. -f1)
    MINOR=$(echo "$GO_VERSION" | cut -d. -f2)
    if [ "$MAJOR" -ge 1 ] && [ "$MINOR" -ge 22 ]; then
        echo -e "  Go: ${GREEN}$GO_VERSION (OK)${NC}"
    else
        echo -e "  Go: ${RED}$GO_VERSION (need 1.22+)${NC}"
        echo -e "  Install from: https://go.dev/dl/"
        exit 1
    fi
else
    echo -e "  Go: ${RED}not found${NC}"
    echo -e "  Install from: https://go.dev/dl/"
    exit 1
fi

# Detect available providers
echo -e "\n${YELLOW}Detecting available providers...${NC}"
PROVIDERS_FOUND=0

# Check Ollama
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        OLLAMA_MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | head -3 | sed 's/"name":"//g;s/"//g' | tr '\n' ', ' | sed 's/,$//')
        echo -e "  Ollama: ${GREEN}running${NC} (models: $OLLAMA_MODELS)"
        PROVIDERS_FOUND=$((PROVIDERS_FOUND + 1))
    else
        echo -e "  Ollama: ${YELLOW}installed but not running${NC} (run 'ollama serve')"
    fi
else
    echo -e "  Ollama: ${YELLOW}not installed${NC} (optional - https://ollama.ai)"
fi

# Check OpenAI
if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "  OpenAI: ${GREEN}API key found${NC}"
    PROVIDERS_FOUND=$((PROVIDERS_FOUND + 1))
elif [ -f "openai_key.txt" ]; then
    echo -e "  OpenAI: ${GREEN}key file found${NC}"
    PROVIDERS_FOUND=$((PROVIDERS_FOUND + 1))
else
    echo -e "  OpenAI: ${YELLOW}no API key${NC} (optional)"
fi

# Check Anthropic
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "  Anthropic: ${GREEN}API key found${NC}"
    PROVIDERS_FOUND=$((PROVIDERS_FOUND + 1))
elif [ -f "anthropic_key.txt" ]; then
    echo -e "  Anthropic: ${GREEN}key file found${NC}"
    PROVIDERS_FOUND=$((PROVIDERS_FOUND + 1))
else
    echo -e "  Anthropic: ${YELLOW}no API key${NC} (optional)"
fi

# Check Gemini
if [ -n "$GEMINI_API_KEY" ]; then
    echo -e "  Gemini: ${GREEN}API key found${NC}"
    PROVIDERS_FOUND=$((PROVIDERS_FOUND + 1))
elif [ -f "gemini_key.txt" ]; then
    echo -e "  Gemini: ${GREEN}key file found${NC}"
    PROVIDERS_FOUND=$((PROVIDERS_FOUND + 1))
else
    echo -e "  Gemini: ${YELLOW}no API key${NC} (optional)"
fi

if [ $PROVIDERS_FOUND -eq 0 ]; then
    echo -e "\n${YELLOW}Warning: No providers detected.${NC}"
    echo "You'll need at least one of:"
    echo "  - Ollama running locally (ollama serve)"
    echo "  - OPENAI_API_KEY environment variable"
    echo "  - ANTHROPIC_API_KEY environment variable"
    echo
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build
echo -e "\n${YELLOW}Building llm-proxy...${NC}"
go build -o llm-proxy .
echo -e "  Binary: ${GREEN}./llm-proxy${NC}"

# Create data directory
mkdir -p data
echo -e "  Data dir: ${GREEN}./data${NC}"

# Create env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cat > .env << 'EOF'
# LLM Proxy Configuration
# Uncomment and set values as needed

PORT=8080

# Provider API Keys (set at least one, or use Ollama)
# OPENAI_API_KEY=sk-xxx
# ANTHROPIC_API_KEY=sk-ant-xxx
# GEMINI_API_KEY=xxx

# Local providers
OLLAMA_HOST=localhost:11434
# LLAMACPP_HOST=localhost:8081

# Data storage
DATA_DIR=./data

# Optional: Speech services
# WHISPER_SERVER_URL=http://localhost:8890
# TTS_SERVER_URL=http://localhost:7788
EOF
    echo -e "  Created: ${GREEN}.env${NC} (edit to add API keys)"
fi

# Ask about service installation
echo -e "\n${YELLOW}Install as system service?${NC}"
echo "This will start llm-proxy automatically on boot."
read -p "Install service? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ "$OS_TYPE" = "macos" ]; then
        install_macos_service
    else
        install_linux_service
    fi
else
    echo -e "\n${GREEN}Installation complete!${NC}"
    echo
    echo "To run manually:"
    echo -e "  ${BLUE}source .env && ./llm-proxy${NC}"
    echo
    echo "Or with specific settings:"
    echo -e "  ${BLUE}PORT=8080 OPENAI_API_KEY=sk-xxx ./llm-proxy${NC}"
    echo
    echo "Then open: http://localhost:8080"
fi
