#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# MythoMax API — Hetzner CPX62 Deploy Script
# ═══════════════════════════════════════════════════════════════
#
# Pre-requisites:
#   1. Hetzner CPX62 (16 vCPU / 32GB RAM) with Ubuntu/Debian
#   2. Docker installed: curl -fsSL https://get.docker.com | sh
#   3. Model file in: ./models/mythomax-l2-13b.Q4_K_M.gguf
#   4. .env file with MYTHOMAX_API_KEY set
#   5. DNS: mythomax.parallaxstar.com → this server's IP
#
# Usage: bash deploy-hetzner.sh
#
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
COMPOSE="docker-compose.hetzner.yml"

echo "═══════════════════════════════════════════════════"
echo "  MythoMax API — Hetzner Deployment"
echo "═══════════════════════════════════════════════════"

# ── Pre-flight ────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/5] Pre-flight checks...${NC}"

[ ! -f ".env" ] && { echo -e "${RED}.env not found! Copy .env.template → .env${NC}"; exit 1; }
[ ! -f "Caddyfile" ] && { echo -e "${RED}Caddyfile not found!${NC}"; exit 1; }
[ ! -f "models/mythomax-l2-13b.Q4_K_M.gguf" ] && {
    echo -e "${RED}Model file not found at models/mythomax-l2-13b.Q4_K_M.gguf${NC}"
    echo "Download: huggingface-cli download TheBloke/MythoMax-L2-13B-GGUF mythomax-l2-13b.Q4_K_M.gguf --local-dir ./models"
    exit 1
}

echo -e "${GREEN}✓ All checks passed${NC}"

# ── Swap ──────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/5] Checking swap...${NC}"
SWAP=$(free -m | awk '/^Swap:/ {print $2}')
if [ "$SWAP" -lt 4000 ]; then
    echo "Setting up 4GB swap..."
    sudo fallocate -l 4G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
    sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
    grep -q swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    echo -e "${GREEN}✓ 4GB swap enabled${NC}"
else
    echo -e "${GREEN}✓ Swap OK (${SWAP}MB)${NC}"
fi

# ── System limits for mlock ───────────────────────────────────
echo -e "\n${YELLOW}[3/5] Setting memlock limits...${NC}"
grep -q "memlock" /etc/security/limits.conf 2>/dev/null || {
    echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf >/dev/null
    echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf >/dev/null
}
echo -e "${GREEN}✓ memlock unlimited${NC}"

# ── Build & Start ─────────────────────────────────────────────
echo -e "\n${YELLOW}[4/5] Building & starting...${NC}"
docker compose -f "$COMPOSE" up -d --build
echo -e "${GREEN}✓ Containers started${NC}"

# ── Status ────────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/5] Status check...${NC}"
sleep 5
docker compose -f "$COMPOSE" ps

echo ""
echo "═══════════════════════════════════════════════════"
echo -e "${GREEN}  MythoMax API Deployed!${NC}"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  URL:     https://mythomax.parallaxstar.com"
echo "  Health:  https://mythomax.parallaxstar.com/health"
echo ""
echo "  Model will take ~10 min to load into RAM."
echo "  Monitor: docker compose -f $COMPOSE logs -f mythomax"
echo ""
echo "  Test:"
echo '  curl -X POST https://mythomax.parallaxstar.com/chat \'
echo '    -H "Authorization: Bearer YOUR_API_KEY" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"prompt":"Hello!","max_tokens":64}'"'"
echo "═══════════════════════════════════════════════════"
