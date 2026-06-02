# MythoMax API — Deployment Wiki

## Architecture

```
XinMate App (Azure) ──HTTPS──→ mythomax.parallaxstar.com (Hetzner CPX62)
                                        ↓
                                  Caddy (auto HTTPS + API key gate)
                                        ↓
                                  MythoMax CPU (:8000)
                                  ├── Q4_K_M GGUF model (~7.4GB)
                                  ├── 5 threads, mlock, n_ctx=2048
                                  └── ~15GB RAM usage
```

---

## Local Testing (No HTTPS, No Caddy)

```bash
# Start MythoMax locally (use docker-compose.local.yml)
docker compose -f docker-compose.local.yml up -d --build

# Wait for model to load (~8-10 min)
docker logs -f mythomax-cpu

# Check health
curl http://localhost:8000/health

# Test inference
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"You are Scarlett. User: Hello!","max_tokens":64,"temperature":0.85}'

# Check performance stats
docker logs mythomax-cpu 2>&1 | grep "perf_context"

# Stop
docker compose -f docker-compose.local.yml down
```

---

## Production Deploy — Hetzner CPX62

### 1. Server Setup (one time)

```bash
# SSH into Hetzner
ssh root@your-hetzner-ip

# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone repo
git clone https://github.com/your-org/mythomax-api-deployment.git
cd mythomax-api-deployment
```

### 2. Get the Model

```bash
mkdir -p models

# Option A: Download from HuggingFace (~7.4GB, takes ~10 min)
apt install -y python3-pip
pip3 install huggingface-hub
huggingface-cli download TheBloke/MythoMax-L2-13B-GGUF \
  mythomax-l2-13b.Q4_K_M.gguf --local-dir ./models

# Option B: Copy from your local machine
scp /path/to/mythomax-l2-13b.Q4_K_M.gguf root@hetzner-ip:~/mythomax-api-deployment/models/

# Verify
ls -lh models/mythomax-l2-13b.Q4_K_M.gguf
# Should show ~7.4G
```

### 3. Configure

```bash
cp .env.template .env
nano .env
```

Set a strong API key:
```bash
# Generate random key
echo "MYTHOMAX_API_KEY=$(openssl rand -hex 24)" > .env
cat .env
# Save this key — you'll need it in XinMate's .env.prod as RUNPOD_LLM_API_KEY
```

### 4. DNS Setup

Point `mythomax.parallaxstar.com` → Hetzner server IP (A record)

### 5. Deploy

```bash
bash deploy-hetzner.sh
```

### 6. Verify

```bash
# Check containers
docker compose -f docker-compose.hetzner.yml ps

# Health (no auth needed)
curl https://mythomax.parallaxstar.com/health

# Wait for model_ready: true (takes ~10 min after deploy)
watch -n 10 'curl -sf https://mythomax.parallaxstar.com/health'

# Test inference (requires API key)
curl -X POST https://mythomax.parallaxstar.com/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"You are Scarlett. User: Hey!","max_tokens":64,"temperature":0.85}'

# Check Caddy SSL
docker compose -f docker-compose.hetzner.yml logs caddy | grep "certificate"
```

---

## Day-to-Day Commands

```bash
COMPOSE="docker-compose.hetzner.yml"

# View logs
docker compose -f $COMPOSE logs -f mythomax
docker compose -f $COMPOSE logs -f caddy

# Check model performance
docker logs mythomax-api 2>&1 | grep "perf_context" | tail -5

# Check memory usage
docker stats mythomax-api --no-stream

# Restart (model takes ~10 min to reload!)
docker compose -f $COMPOSE restart mythomax

# Update code (rebuild without redownloading model)
git pull
docker compose -f $COMPOSE up -d --build mythomax

# Stop
docker compose -f $COMPOSE down

# Shell into container
docker compose -f $COMPOSE exec mythomax bash
```

---

## Performance Reference

| Metric | Value (CPX62, 16 vCPU) |
|---|---|
| Model load time | ~8-10 min |
| RAM usage | ~15 GB |
| Prompt eval | ~6-8 tok/s |
| Generation | ~3-4 tok/s |
| Short reply (40 tok) | ~12-15s |
| Medium reply (80 tok) | ~25-30s |
| Concurrent users | 1 active + queue |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `model_ready: false` | Model still loading. Wait 10 min. Check: `docker logs mythomax-api` |
| `curl: 401 Unauthorized` | Wrong API key. Check `.env` on Hetzner |
| `curl: connection refused` | Caddy not running. Check: `docker logs mythomax-caddy` |
| Very slow responses (>60s) | Check CPU: `docker stats`. Ensure mlock is working (no swapping) |
| Container OOM killed | Reduce `N_CTX` to 1024 or increase memory limit |
| Model loading stuck | Disk I/O bottleneck. Check: `iostat -x 1` |
| SSL cert error | DNS not propagated. Check: `dig mythomax.parallaxstar.com` |
