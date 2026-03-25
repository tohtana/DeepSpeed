#!/usr/bin/env bash
# Create .venv at the DeepSpeed repo root with:
#   - PyTorch 2.8.0+cu128 (CUDA 12.8)
#   - requirements from requirements/requirements.txt
#   - DeepSpeed editable install from the *current* checkout (latest local code)
#   - pytest (for unit tests)
#
# Usage (from anywhere):
#   ./scripts/setup.sh
#
# Then from repo root:
#   source .venv/bin/activate
#   torchrun --standalone --nproc_per_node=1 scripts/repro_zero3_functorch_linear.py
#
# To reproduce a failure on an older DeepSpeed release instead of this tree:
#   pip install 'deepspeed==0.16.4'  # after venv is active; skip pip install -e . once or use a fresh venv
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

rm -rf .venv
python3 -m venv .venv
# shellcheck source=/dev/null
. .venv/bin/activate

python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 to match the bug report"' || {
  echo "Warning: expected Python 3.11; found $(python -V)" >&2
}

pip install -U pip setuptools wheel

# PyTorch 2.8.0 + CUDA 12.8 (matches common functorch / ZeRO-3 bug reports)
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements/requirements.txt

# Latest DeepSpeed = this git checkout (editable)
pip install -e .

pip install pytest

python -c "import torch, deepspeed; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('deepspeed', deepspeed.__file__); print('deepspeed version', deepspeed.__version__)"
