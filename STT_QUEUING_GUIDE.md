# STT Request Queuing Guide

## Configuration
- **Max Concurrent:** 5 (dynamic, scales down under CPU load)
- **Max Queue Size:** 200 requests
- **Request Timeout:** 120 seconds (waits in queue + executes)

## Request Flow

```
┌─ Incoming STT Request ─────────┐
│                                │
│  AcquireSlot(120s timeout)    │
│  ├─ Try: Get concurrent slot  │
│  ├─ If busy: Try queue (200)  │
│  └─ If full: Return 429       │
│                                │
└────────────────────────────────┘
         ↓
    [3 Outcomes]
         ↓
    ┌────┬────┬────┐
    ↓    ↓    ↓
  OK    OK  QUEUED 429
execute execute wait  reject
immediate 0.5-10ms  0-120s
```

## Dynamic Concurrency Scaling

```
CPU Usage    │ Allowed Concurrent │ Queue Size │ Reject When
─────────────┼────────────────────┼────────────┼─────────────
< 30%        │ 5                  │ 200        │ 201+
30-50%       │ 4                  │ 200        │ 201+
50-70%       │ 3                  │ 200        │ 201+
70-85%       │ 2                  │ 200        │ 201+
> 85%        │ 1                  │ 200        │ 201+

Memory
─────────────┼────────────────────┼────────────┼─────────────
< 10GB free  │ 1                  │ 200        │ 201+
10-20GB free │ -2 from CPU limit  │ 200        │ 201+
20-40GB free │ -1 from CPU limit  │ 200        │ 201+
```

## Scenarios

### Low Load (CPU 20%, Memory 40%)
```
5 concurrent slots available
6th request → Enters queue (199 slots remaining)
205th request → Enters queue (1 slot remaining)
206th request → Returns 429
```

### Peak Load (CPU 75%, Memory 65%)
```
2 concurrent slots available (due to CPU throttling)
Request 1-2 → Execute immediately
Request 3-202 → Queue (200 slots)
Request 203 → Returns 429 (queue full)
```

### Critical Load (CPU 90%, Memory 85%)
```
1 concurrent slot available
Request 1 → Execute immediately
Request 2-201 → Queue (200 slots)
Request 202 → Returns 429 (queue full)
```

## HTTP Response Examples

### Success (Slot Available)
```
HTTP 200 OK
Content-Type: application/json
X-LLM-Proxy-Latency-Ms: 4250

{
  "text": "hello world"
}
```

### Queued (Processing)
```
HTTP 200 OK
Content-Type: application/json
X-LLM-Proxy-Latency-Ms: 45000  (includes queue wait)

{
  "text": "transcribed text"
}
```

### Queue Full (Reject)
```
HTTP 429 Too Many Requests
Retry-After: 30

server queue full, please retry after 30 seconds
```

### Timeout (120s exceeded)
```
HTTP 503 Service Unavailable

Request timeout waiting for available slot
```

## Monitoring

### Check Current Queue Status
```bash
curl http://studio.lan:8080/metrics | grep -E "stt_(current|queue|allowed|cpu|mem)"

# Output:
llm_proxy_stt_cpu_percent 45.67
llm_proxy_stt_mem_percent 62.34
llm_proxy_stt_mem_free_gb 72.50
llm_proxy_stt_allowed_concurrent 4
llm_proxy_stt_current_concurrent 3
llm_proxy_stt_queue_pending 12
llm_proxy_stt_requests_total 542
llm_proxy_stt_rejections_total 2
llm_proxy_stt_rejection_rate 0.0037
```

### Watch in Real-Time
```bash
watch -n 1 'curl -s http://studio.lan:8080/metrics | grep stt_ | grep -E "(current|queue|allowed|cpu|rejection)"'
```

### Set Alert When Queue > 50
```
if curl -s http://studio.lan:8080/metrics | grep stt_queue_pending | awk '{if ($NF > 50) exit 1}'; then
  echo "Queue healthy"
else
  echo "ALERT: Queue depth > 50"
fi
```

## Tuning Guide

### If You See Many 429 Errors
```
Problem: Queue filling up too quickly
Solution: Increase queue size or reduce concurrent requests

# Option 1: Increase queue (max practical: 500)
NewConcurrencyManager(5, 500)

# Option 2: Reduce max concurrent (add headroom)
NewConcurrencyManager(3, 200)
```

### If CPU is Always High (>85%)
```
Problem: LLM models too large, causing stalls
Solution: Load smaller models or fewer concurrent models

# Example: Remove one Qwen model
# 4x Qwen → 3x Qwen (reduces CPU by ~25%)
```

### If Memory Usage > 80%
```
Problem: Not enough headroom for spikes
Solution: Load fewer models or use smaller variants

# Example: Use Qwen 3 8B instead of 32B
# Memory: 24GB → 7GB per instance
```

## Client Implementation

### Retry with Exponential Backoff
```python
import time
import requests

def transcribe_with_queue(audio_file, max_retries=5):
    url = "http://studio.lan:8080/v1/audio/transcriptions"

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                files={"file": audio_file},
                timeout=180  # 3 minutes for queue wait + processing
            )

            if response.status_code == 429:
                # Queue full, back off and retry
                retry_after = int(response.headers.get("Retry-After", 30))
                wait_time = retry_after * (2 ** attempt)  # Exponential backoff
                print(f"Queue full, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

    raise Exception(f"Failed after {max_retries} retries")
```

## Performance Characteristics

### Average Transcription Times
```
Audio Length │ Processing Time │ Queue Wait (peak)
──────────────┼─────────────────┼──────────────────
5 seconds     │ 2-3 seconds     │ 0-120 seconds
30 seconds    │ 10-15 seconds   │ 0-120 seconds
60 seconds    │ 20-30 seconds   │ 0-120 seconds
```

### Queue Draining
```
Scenario: 100 requests queued at peak, then load drops

Time │ CPU Load │ Concurrent │ Queue Depth │ Drain Rate
─────┼──────────┼────────────┼─────────────┼──────────
0s   │ 85%      │ 2          │ 100         │ 0
30s  │ 45%      │ 4          │ 90          │ 1/min
60s  │ 25%      │ 5          │ 70          │ 4/min
90s  │ 20%      │ 5          │ 40          │ 6/min
120s │ 15%      │ 5          │ 10          │ 6/min
150s │ 10%      │ 5          │ 0           │ ✅ empty

Total drain time: ~2.5 minutes
```

## Limits and Constraints

```
Physical Limits (M2 Ultra)
├─ Max Concurrent:        5 (limited by dynamic scaling algo)
├─ Max Queue:            200 (practical limit, memory per request ~5MB)
├─ Max Total Pending:    205 (5 executing + 200 queued)
├─ Max Request Timeout:  120 seconds (hardcoded)
└─ Max Queue Wait:       120 seconds (before 503 Service Unavailable)

Memory for Queue
├─ Per Request (queued): ~5MB (metadata only, not audio data)
├─ 200 requests:        ~1GB overhead
├─ Safe headroom:       192GB - 1GB queue - LLM models > 50GB free
```

## Summary

- ✅ Up to 200 requests can queue before rejection
- ✅ Dynamic limits adapt to CPU load (1-5 concurrent)
- ✅ 120 second timeout for queue + execution
- ✅ Graceful 429 response with Retry-After header
- ✅ Observable via Prometheus metrics
- ✅ No data loss, queue survives if request times out (client retries)
