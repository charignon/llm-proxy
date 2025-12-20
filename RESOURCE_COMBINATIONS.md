# Studio.lan Resource Combinations Guide

## Hardware Specs (M2 Ultra)
- **CPU:** 12 cores (8 performance + 4 efficiency)
- **RAM:** 192 GB unified memory
- **Disk:** SSD storage

## Model Sizes (on-disk → in-memory)

### Large Models (30-70B parameters)
| Model | Size | Loaded | CPU Impact | Use Case |
|-------|------|--------|-----------|----------|
| Llama 3.3 70B | 42.5 GB | ~51 GB | Very High (3-4 cores) | Reasoning, coding, complex tasks |
| Llama 3.1 70B | 42.5 GB | ~51 GB | Very High (3-4 cores) | General purpose large |
| DeepSeek R1 70B | 42.5 GB | ~51 GB | Very High (3-4 cores) | Reasoning, step-by-step |
| Qwen 3 32B | 20.2 GB | ~24 GB | Medium-High (2-3 cores) | Fast, efficient |
| Qwen 2.5-Coder 32B | 19.8 GB | ~24 GB | Medium-High (2-3 cores) | Code generation |
| DeepSeek Coder 33B | 18.8 GB | ~23 GB | Medium-High (2-3 cores) | Code-focused |
| Qwen 3-VL 30B | 19.6 GB | ~24 GB | Medium-High (2-3 cores) | Vision + text |
| Exaone Deep 32B | 19.3 GB | ~23 GB | Medium-High (2-3 cores) | General purpose |
| Qwen 3-Coder 30B | 18.6 GB | ~22 GB | Medium (2-3 cores) | Code specialized |

### Medium Models (15-25B parameters)
| Model | Size | Loaded | CPU Impact | Use Case |
|-------|------|--------|-----------|----------|
| DeepSeek Coder V2 16B | 8.9 GB | ~11 GB | Low-Medium (1-2 cores) | Code, fast |
| Codestral 24B | 12.6 GB | ~15 GB | Low-Medium (1-2 cores) | Code generation |
| Phi 4 14B | 9.1 GB | ~11 GB | Low (1-2 cores) | Lightweight, fast |

### Small Models (3-8B parameters)
| Model | Size | Loaded | CPU Impact | Use Case |
|-------|------|--------|-----------|----------|
| Mistral 7B | 4.4 GB | ~5 GB | Low (1 core) | Lightweight, general |
| Llama 3.1 8B | 4.9 GB | ~6 GB | Low (1 core) | Small, efficient |
| Granite 3.1 MOE 3B | 2.0 GB | ~2.5 GB | Minimal (<1 core) | Tiny, fast |

### Vision Models
| Model | Size | Loaded | CPU Impact | Special Notes |
|-------|------|--------|-----------|----------|
| Qwen 3-VL 235B | 143.3 GB | ~172 GB | CRITICAL | Only loadable alone (uses almost all RAM) |
| Qwen 3-VL 30B | 19.6 GB | ~24 GB | Medium-High | Practical vision model |
| Qwen 3-VL 8B | 6.1 GB | ~7 GB | Low-Medium | Lightweight vision |

### Audio Services
| Service | RAM | CPU Impact | Notes |
|---------|-----|-----------|-------|
| Whisper (per instance) | 2-3 GB | Medium (1-2 cores) | STT, can run 3-5 concurrent |
| Kokoro (per instance) | 1-2 GB | Medium (1-2 cores) | TTS, 1-2 concurrent recommended |

---

## Viable Combinations

### 🔥 Scenario 1: Maximum Reasoning Power (1 Large + Whisper/Kokoro)
```
1x Llama 3.3 70B    ~51 GB  (very high precision reasoning)
+ 3x Whisper         ~9 GB  (concurrent STT)
+ 1x Kokoro          ~2 GB  (TTS)
─────────────────────────
Total RAM:          ~62 GB  (68% utilization)
CPU: 12 cores fully saturated
Status: ✅ OPTIMAL for reasoning-heavy workloads
```

**When to use:** Complex reasoning, code generation, step-by-step problem solving

---

### 🎯 Scenario 2: Multi-Model Processing (3x Qwen + Audio)
```
3x Qwen 3 32B       ~72 GB  (3 independent processing streams)
+ 3x Whisper         ~9 GB  (concurrent transcription)
+ 1x Kokoro          ~2 GB  (TTS)
─────────────────────────
Total RAM:          ~83 GB  (43% utilization)
CPU: All 12 cores active (balanced)
Status: ✅ OPTIMAL for parallel processing
```

**When to use:** Multiple concurrent LLM requests, speech I/O heavy

---

### 🚀 Scenario 3: Balanced Multi-Model (Your Current Setup + 1 More)
```
4x Qwen 3 32B       ~96 GB  (4 parallel processing pipelines)
+ 2x Whisper         ~6 GB  (concurrent STT)
+ 1x Kokoro          ~2 GB  (TTS)
─────────────────────────
Total RAM:         ~104 GB  (54% utilization)
CPU: All 12 cores very active
Status: ✅ FEASIBLE with your requested "4 Qwen"
```

**When to use:** Maximum parallelism, moderate CPU load, multi-tenant serving

---

### 💡 Scenario 4: Hybrid Large + Small (Best CPU/RAM Balance)
```
1x Llama 3.3 70B    ~51 GB  (reasoning)
2x Mistral 7B       ~10 GB  (fast responses)
1x Phi 4 14B        ~11 GB  (lightweight processing)
+ 4x Whisper        ~12 GB  (concurrent transcription)
+ 2x Kokoro          ~4 GB  (concurrent TTS)
─────────────────────────
Total RAM:          ~88 GB  (46% utilization)
CPU: Mixed load, 10-12 cores active
Status: ✅ OPTIMAL - versatile, responsive
```

**When to use:** Production serving mixed workloads, high concurrency

---

### 🎬 Scenario 5: Vision-Heavy (Best for Image Analysis)
```
1x Qwen 3-VL 30B    ~24 GB  (images + text)
1x Llama 3.3 70B    ~51 GB  (text reasoning)
2x Mistral 7B       ~10 GB  (lightweight text)
+ 2x Whisper         ~6 GB  (transcription)
+ 1x Kokoro          ~2 GB  (TTS)
─────────────────────────
Total RAM:          ~93 GB  (48% utilization)
CPU: All 12 cores active
Status: ✅ OPTIMAL for vision + reasoning
```

**When to use:** Image analysis with text reasoning, document processing

---

### 📊 Scenario 6: Maximum Audio Processing (Speech I/O Focus)
```
1x Qwen 3 32B       ~24 GB  (routing/response generation)
+ 5x Whisper        ~15 GB  (max concurrent STT from concurrency mgr)
+ 2x Kokoro          ~4 GB  (concurrent TTS)
─────────────────────────
Total RAM:          ~43 GB  (22% utilization)
CPU: CPU-bound (audio processing)
Status: ✅ OPTIMAL for voice-first applications
```

**When to use:** Voice assistants, call centers, speech-heavy workflows

---

### 🔬 Scenario 7: Code Specialization (Developer-Focused)
```
1x DeepSeek Coder V2 16B  ~11 GB  (code completion)
1x Qwen 2.5-Coder 32B     ~24 GB  (code generation)
1x Mistral 7B             ~5 GB   (fast responses)
+ 2x Whisper               ~6 GB  (voice commands)
─────────────────────────
Total RAM:          ~46 GB  (24% utilization)
CPU: 8-10 cores active
Status: ✅ FEASIBLE
```

**When to use:** IDE plugins, code review, pair programming

---

### ⚠️ Scenario 8: Not Recommended - Too Heavy
```
❌ 1x Llama 70B + 1x Qwen 32B + 1x DeepSeek 70B
   = ~118 GB alone (missing audio services)

❌ 1x Qwen 3-VL 235B (alone uses 172GB, leaves only 20GB for OS/services)

❌ 5x Qwen 32B (120GB) + 2x Whisper + other services
   = RAM thrashing, very slow
```

---

## STT Concurrency Limits by Scenario

The dynamic concurrency manager adjusts based on CPU load:

| Scenario | Base Limit | Under Heavy Load | Min Concurrent |
|----------|-----------|-----------------|----------------|
| Scenario 1 (1 Large) | 5 Whisper | Drops to 3 | 1 |
| Scenario 2 (3 Qwen) | 4 Whisper | Drops to 2 | 1 |
| Scenario 3 (4 Qwen) | 3 Whisper | Drops to 2 | 1 |
| Scenario 4 (Hybrid) | 4 Whisper | Drops to 2 | 1 |
| Scenario 5 (Vision) | 3 Whisper | Drops to 2 | 1 |
| Scenario 6 (Audio) | 5 Whisper | Drops to 3 | 2 |

---

## Tuning the Concurrency Manager

The current settings are:
```go
sttConcurrencyMgr = loadmanager.NewConcurrencyManager(5, 20)
```

To adjust for a specific scenario:

```go
// For Scenario 3 (4 Qwen - will be CPU-saturated):
sttConcurrencyMgr = loadmanager.NewConcurrencyManager(3, 20)

// For Scenario 6 (Audio focus - lots of free CPU):
sttConcurrencyMgr = loadmanager.NewConcurrencyManager(5, 20)  // Keep default
```

The first parameter is the **maximum concurrent** (will auto-scale down under load).
The second parameter is the **queue size** before rejecting with 429.

---

## CPU Core Allocation

With 12 cores, think about distribution:

```
Scenario 1 (1 large LLM):
├─ Llama 70B:    3-4 cores
├─ Whisper (3):  3 cores (1 each)
├─ Kokoro:       2 cores
├─ System:       2 cores
└─ Free:         1-2 cores

Scenario 3 (4 Qwen):
├─ Qwen (4):     8 cores (2 each)
├─ Whisper (2):  2 cores (1 each)
├─ Kokoro:       1 core
└─ System:       1 core
```

---

## Memory Headroom Recommendations

Always reserve 10-15% for:
- OS and system processes (10-15 GB)
- Transient buffers and caching (5-10 GB)
- Safety margin (5-10 GB)

**Safe utilization:** 60-70% of total RAM
- Scenario 1: 62 GB = 32% ✅ (very safe)
- Scenario 2: 83 GB = 43% ✅ (good)
- Scenario 3: 104 GB = 54% ✅ (still safe)
- Scenario 4: 88 GB = 46% ✅ (optimal)

---

## Switching Between Scenarios

To change at runtime:

```bash
# Stop current Ollama
pkill ollama

# Load new models
ollama run llama3.3:70b &
ollama run qwen3:32b &
ollama run mistral:7b &

# Monitor memory
watch free -h
```

Or use Ollama's API to unload models:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.3:70b",
  "keep_alive": 0
}'
```

---

## Monitoring

Check current load:
```bash
curl http://studio.lan:8080/metrics | grep stt_

# Watch in real-time:
watch -n 2 'curl -s http://studio.lan:8080/metrics | grep stt_'
```

Key metrics to watch:
- `llm_proxy_stt_allowed_concurrent` - how many can run now
- `llm_proxy_stt_current_concurrent` - how many are running
- `llm_proxy_stt_queue_pending` - requests waiting
- `llm_proxy_stt_cpu_percent` - system load
- `llm_proxy_stt_rejection_rate` - 429 errors

---

## Summary Table: Quick Reference

| Scenario | Models | Total RAM | CPU Load | Audio | Best For |
|----------|--------|-----------|----------|-------|----------|
| 1 | 1×70B | 62 GB | Very High | Good | Reasoning, complex tasks |
| 2 | 3×32B | 83 GB | High | Good | Parallel processing |
| 3 | 4×32B | 104 GB | Saturated | Fair | Multi-tenant, your request |
| 4 | 1×70B+2×7B+1×14B | 88 GB | High | Excellent | Production balanced |
| 5 | Vision+70B | 93 GB | High | Good | Image + reasoning |
| 6 | 1×32B+Audio | 43 GB | CPU-bound | Excellent | Voice-first apps |
| 7 | Code models | 46 GB | Medium | Good | Development |

**Recommendation:** Start with **Scenario 4 (Hybrid)** - it gives you versatility while leaving 50% RAM free for spikes.
