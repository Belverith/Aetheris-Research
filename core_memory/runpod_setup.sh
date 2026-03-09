#!/bin/bash
# ============================================================================
# RunPod Setup & Run Script for 7B HIS Safety Experiment
# ============================================================================
# Usage:
#   1. Create a RunPod GPU Pod (recommended: RTX 4090 or A40 — best $/speed)
#   2. SSH into the pod or use the web terminal
#   3. Upload this script + llm_experiment_7b.py to the pod
#   4. Run:  bash runpod_setup.sh
#
# The script will:
#   - Install Ollama
#   - Pull qwen2.5:7b
#   - Install Python dependencies
#   - Run the experiment with full logging
#
# Recommended Pod Config:
#   - GPU: RTX 4090 (24GB VRAM) or A40 (48GB) — 7B model fits easily
#   - Template: RunPod Pytorch 2.x (has Python, CUDA pre-installed)
#   - Disk: 30GB minimum (model ~4.7GB + overhead)
#   - Expected runtime: ~30-60 min on RTX 4090 vs ~6-12 hrs on local RTX 5050
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_SCRIPT="$SCRIPT_DIR/llm_experiment_7b.py"
LOG_FILE="$SCRIPT_DIR/experiment_7b_log.txt"

echo "============================================="
echo "  HIS 7B Experiment — RunPod Setup"
echo "============================================="
echo "  Time: $(date)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================="

# ── Step 1: Install Ollama ──────────────────────────────────────────────────
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "[1/4] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed."
else
    echo ""
    echo "[1/4] Ollama already installed."
fi

# ── Step 2: Start Ollama server ─────────────────────────────────────────────
echo ""
echo "[2/4] Starting Ollama server..."
# Kill any existing ollama serve process
pkill ollama 2>/dev/null || true
sleep 1

# Start ollama serve in background
ollama serve &> /tmp/ollama_serve.log &
OLLAMA_PID=$!
echo "  Ollama server started (PID: $OLLAMA_PID)"

# Wait for it to be ready
echo "  Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if ollama list &> /dev/null; then
        echo "  Ollama ready after ${i}s."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: Ollama failed to start. Check /tmp/ollama_serve.log"
        exit 1
    fi
    sleep 1
done

# ── Step 3: Pull model ─────────────────────────────────────────────────────
echo ""
echo "[3/4] Pulling qwen2.5:7b (~4.7 GB)..."
if ollama list | grep -q "qwen2.5:7b"; then
    echo "  Model already pulled."
else
    ollama pull qwen2.5:7b
    echo "  Model pulled successfully."
fi

# ── Step 4: Install Python dependencies ─────────────────────────────────────
echo ""
echo "[4/4] Installing Python dependencies..."
pip install --quiet ollama numpy matplotlib scipy 2>/dev/null || \
    pip3 install --quiet ollama numpy matplotlib scipy
echo "  Dependencies installed."

# ── Verify everything is ready ──────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Setup complete. Verification:"
echo "============================================="
echo "  Ollama: $(ollama --version 2>/dev/null || echo 'installed')"
echo "  Model:  $(ollama list | grep qwen2.5:7b | head -1)"
echo "  Python: $(python3 --version 2>/dev/null || python --version)"
echo "  NumPy:  $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'ok')"
echo "  Script: $EXPERIMENT_SCRIPT"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null && echo "" || true

if [ ! -f "$EXPERIMENT_SCRIPT" ]; then
    echo ""
    echo "  ERROR: llm_experiment_7b.py not found at $EXPERIMENT_SCRIPT"
    echo "  Upload it to the same directory as this script."
    exit 1
fi

# ── Create assets directory ─────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/assets"

# ── Run the experiment ──────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Starting experiment..."
echo "  Log: $LOG_FILE"
echo "  Results: $SCRIPT_DIR/assets/llm_7b_results.json"
echo "============================================="
echo ""

cd "$SCRIPT_DIR"
python3 llm_experiment_7b.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================="
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "  Experiment completed successfully!"
else
    echo "  Experiment exited with code $EXIT_CODE"
fi
echo "  Log: $LOG_FILE"
echo "  Results: $SCRIPT_DIR/assets/llm_7b_results.json"
echo "  Figure:  $SCRIPT_DIR/assets/figure7_7b_experiment.png"
echo "  Time: $(date)"
echo "============================================="
echo ""
echo "  Download your results:"
echo "    - assets/llm_7b_results.json"
echo "    - assets/figure7_7b_experiment.png"
echo "    - experiment_7b_log.txt"
