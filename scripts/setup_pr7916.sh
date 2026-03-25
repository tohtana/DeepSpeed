#!/usr/bin/env bash
# Create an isolated venv for PR 7916 repro at .venvs/pr7916 (repo root):
#   - Removes only that venv if it already exists (does not touch .venv or other envs)
#   - PyTorch 2.8.0+cu128 (CUDA 12.8)
#   - requirements from requirements/requirements.txt
#   - DeepSpeed editable install from the *current* checkout
#   - pytest
#
# Then validates the fix by:
#   1) Running the repro with the current branch (expect success + "OK" line)
#   2) Checking out master and running the same repro script (expect original RuntimeError)
#   3) Checking back to the branch you started on
#
# Usage (from repo root):
#   ./scripts/setup_pr7916.sh
#
# Activate later:
#   source .venvs/pr7916/bin/activate
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${PR7916_VENV_DIR:-$ROOT/.venvs/pr7916}"
MAIN_REF="${PR7916_MAIN_REF:-master}"

echo "==> Using venv: $VENV_DIR (only this path is removed if it already exists)"
rm -rf "$VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"

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

REPRO_SRC="$ROOT/scripts/repro_pr7916.py"
if [[ ! -f "$REPRO_SRC" ]]; then
  echo "error: missing $REPRO_SRC (need repro script on current branch)" >&2
  exit 1
fi

REPRO_TMP="$(mktemp /tmp/repro_pr7916_XXXXXX.py)"
cp "$REPRO_SRC" "$REPRO_TMP"
cleanup_repro_tmp() { rm -f "$REPRO_TMP"; }
trap cleanup_repro_tmp EXIT

FIX_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
TORCHRUN=(torchrun --standalone --nproc_per_node=1)

echo ""
echo "==> [1/2] Repro on fix branch: $FIX_BRANCH (expect success)"
"${TORCHRUN[@]}" "$REPRO_TMP"

echo ""
echo "==> [2/2] Repro on $MAIN_REF (expect original functorch / setup_context error)"
STASHED=0
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "==> Stashing local changes so checkout to $MAIN_REF can proceed..."
  git stash push -m "pr7916-setup: temp stash before main repro"
  STASHED=1
fi
if ! git checkout "$MAIN_REF"; then
  echo "error: could not checkout $MAIN_REF" >&2
  if [[ "$STASHED" -eq 1 ]]; then
    git stash pop || true
  fi
  exit 1
fi
set +e
"${TORCHRUN[@]}" "$REPRO_TMP"
MAIN_EC=$?
set -e
if [[ "$MAIN_EC" -eq 0 ]]; then
  echo "" >&2
  echo "warning: main branch run exited 0 — expected failure on unfixed tree." >&2
else
  echo ""
  echo "main branch run exited with $MAIN_EC (non-zero is expected for the unfixed tree)."
fi

echo ""
echo "==> Restoring branch: $FIX_BRANCH"
git checkout "$FIX_BRANCH"

if [[ "$STASHED" -eq 1 ]]; then
  echo "==> Restoring stashed local changes..."
  git stash pop || echo "warning: stash pop failed (resolve manually with git stash list)" >&2
fi

echo ""
echo "Done. To use this environment: source $VENV_DIR/bin/activate"
