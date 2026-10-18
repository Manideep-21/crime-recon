#!/usr/bin/env bash
# No-sudo setup for a shared GPU container (no Docker, no conda, no unzip, system Python 3.8).
#   bash setup_hpc.sh
# Installs everything under $HOME: uv, a Python 3.10 venv in .venv, torch + requirements,
# and a user-space Ollama with qwen2.5vl:7b. Neo4j is disabled (no Docker).
set -euo pipefail
cd "$(dirname "$0")"

PREFIX="$HOME/.local"
export PATH="$PREFIX/bin:$PATH"

# 1. uv (gives us Python 3.10 without sudo)
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

# 2. venv + packages
[ -d .venv ] || uv venv .venv --python 3.10
# cu121 stops at torch 2.5, but current diffusers needs torch>=2.6 (torch.accelerator)
uv pip install --python .venv/bin/python "torch>=2.6,<2.7" "torchvision>=0.21,<0.22" --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt zstandard

# 3. Ollama (user space). Releases are .tar.zst and there is no zstd binary, so unpack with Python.
OLLAMA_DIR="$PREFIX/ollama"
if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  mkdir -p "$OLLAMA_DIR"
  curl -fL -o "$OLLAMA_DIR/ollama.tar.zst" https://ollama.com/download/ollama-linux-amd64.tar.zst
  .venv/bin/python - "$OLLAMA_DIR" <<'EOF'
import sys, tarfile, zstandard
d = sys.argv[1]
with open(f"{d}/ollama.tar.zst", "rb") as fh, \
     zstandard.ZstdDecompressor().stream_reader(fh) as r, \
     tarfile.open(fileobj=r, mode="r|") as tar:
    tar.extractall(d)
EOF
  rm "$OLLAMA_DIR/ollama.tar.zst"
fi
ln -sf "$OLLAMA_DIR/bin/ollama" "$PREFIX/bin/ollama"

# 4. Start Ollama if not running, pull the model
if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null; then
  # keep_alive 0: free VRAM after each request so SDXL/OWLv2 can use it (only ~5 GB free here)
  OLLAMA_KEEP_ALIVE=0 nohup ollama serve > "$HOME/.ollama-serve.log" 2>&1 &
  for _ in $(seq 30); do curl -s -m 2 http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi
ollama pull qwen2.5vl:7b

# 5. No Docker -> no Neo4j
sed -i 's/^  enabled: true /  enabled: false/' config.yaml

# 6. Logic tests (no GPU needed)
.venv/bin/python tests/test_logic.py

echo
echo "Done. Use:  source .venv/bin/activate && export PATH=$PREFIX/bin:\$PATH"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || true
