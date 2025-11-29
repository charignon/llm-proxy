.PHONY: build run test clean deploy

PORT ?= 8080
OLLAMA_HOST ?= localhost:11434

build:
	go build -o llm-proxy .

run: build
	PORT=$(PORT) OLLAMA_HOST=$(OLLAMA_HOST) ./llm-proxy

test:
	@echo "Testing chat completions endpoint..."
	curl -s http://localhost:$(PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "user", "content": "Say hello"}], "sensitive": false, "precision": "low"}' | jq .

test-routing:
	@echo "Testing routing..."
	curl -s http://localhost:$(PORT)/api/routes | jq .

test-stats:
	@echo "Testing stats..."
	curl -s http://localhost:$(PORT)/api/stats | jq .

clean:
	rm -f llm-proxy
	rm -rf data/

# Deploy to studio.lan
deploy: build
	scp llm-proxy studio.lan:/home/laurent/
	ssh studio.lan "sudo systemctl restart llm-proxy || echo 'Service not configured yet'"
