#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — environment bootstrap for LNN_crossover_CVRP
#
# Creates a virtualenv, installs the correct PyTorch build for THIS host,
# installs the remaining dependencies, and verifies the result.
#
# Supported hosts:
#   * macOS / arm64        -> CPU + MPS build from PyPI          (dev machine)
#   * Linux / aarch64+CUDA -> CUDA wheels  (DGX Spark, GB10 Blackwell, sm_121)
#   * Linux / x86_64+CUDA  -> CUDA wheels
#   * anything else        -> CPU-only build
#
# Usage:
#   ./setup.sh                        # auto-detect
#   ./setup.sh --cpu                  # force CPU-only torch
#   ./setup.sh --venv /path/to/env    # custom venv location
#   TORCH_CUDA_CHANNEL=cu128 ./setup.sh   # override CUDA wheel channel
#   PYTHON=python3.12 ./setup.sh          # override interpreter
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
FORCE_CPU=0
MIN_PY_MINOR=10          # torch >= 2.4 requires Python >= 3.9; we ask for 3.10+

# ----------------------------- pretty output -------------------------------
if [ -t 1 ]; then
  B="\033[1m"; G="\033[32m"; Y="\033[33m"; R="\033[31m"; N="\033[0m"
else
  B=""; G=""; Y=""; R=""; N=""
fi
say()  { printf "${B}==>${N} %s\n" "$*"; }
ok()   { printf "  ${G}ok${N}    %s\n" "$*"; }
warn() { printf "  ${Y}warn${N}  %s\n" "$*"; }
die()  { printf "  ${R}error${N} %s\n" "$*" >&2; exit 1; }

# ----------------------------- arguments -----------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)   FORCE_CPU=1; shift ;;
    --venv)  VENV_DIR="${2:?--venv needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

# ----------------------------- host detection ------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
say "Host: $OS / $ARCH"

HAS_CUDA=0
if [ "$FORCE_CPU" -eq 0 ] && [ "$OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    HAS_CUDA=1
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
    ok "CUDA GPU detected: ${GPU_NAME}"
  else
    warn "nvidia-smi present but not responding; falling back to CPU build"
  fi
fi

# Choose the PyTorch wheel index.
#   GB10 / Blackwell (DGX Spark) is sm_121 and needs a CUDA 13.x build.
#   Override with TORCH_CUDA_CHANNEL if your driver needs a different one.
TORCH_INDEX=""
if [ "$HAS_CUDA" -eq 1 ]; then
  if [ "$ARCH" = "aarch64" ]; then
    CHANNEL="${TORCH_CUDA_CHANNEL:-cu130}"   # DGX Spark / Grace-Blackwell
  else
    CHANNEL="${TORCH_CUDA_CHANNEL:-cu128}"   # x86_64 CUDA host
  fi
  TORCH_INDEX="https://download.pytorch.org/whl/${CHANNEL}"
  ok "PyTorch channel: ${CHANNEL}"
elif [ "$OS" = "Darwin" ]; then
  ok "PyTorch channel: default PyPI (CPU + MPS)"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  ok "PyTorch channel: cpu"
fi

# ----------------------------- interpreter ---------------------------------
say "Locating Python >= 3.${MIN_PY_MINOR}"
PY=""
for cand in "${PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3; do
  [ -n "$cand" ] || continue
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, $MIN_PY_MINOR) else 1)" 2>/dev/null; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || die "no Python >= 3.${MIN_PY_MINOR} found. Install one, or set PYTHON=/path/to/python."
ok "using $PY ($("$PY" -V 2>&1))"

# ----------------------------- virtualenv ----------------------------------
say "Creating virtualenv at $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
  ok "already exists, reusing"
else
  "$PY" -m venv "$VENV_DIR"
  ok "created"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"

say "Upgrading pip toolchain"
"$PIP" install --quiet --upgrade pip setuptools wheel
ok "$("$PIP" --version)"

# ----------------------------- torch ---------------------------------------
say "Installing PyTorch"
if [ -n "$TORCH_INDEX" ]; then
  "$PIP" install --index-url "$TORCH_INDEX" torch \
    || die "torch install failed from $TORCH_INDEX
         Try a different channel, e.g.:  TORCH_CUDA_CHANNEL=cu128 ./setup.sh
         Or check available builds at:   https://pytorch.org/get-started/locally/"
else
  "$PIP" install torch
fi
ok "torch installed"

# ----------------------------- remaining deps ------------------------------
say "Installing remaining dependencies"
"$PIP" install -r "$PROJECT_ROOT/requirements.txt"
ok "requirements.txt satisfied"

# ----------------------------- verification --------------------------------
say "Verifying installation"
"$VENV_DIR/bin/python" - <<'PYCHECK'
import importlib, sys

failed = []
for mod in ("torch", "numpy", "pandas", "scipy", "matplotlib", "tensorboard"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok    {mod:<12} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  error {mod:<12} {e}")
        failed.append(mod)

import torch
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(i)
    print(f"  ok    device       cuda -> {torch.cuda.get_device_name(i)} "
          f"(sm_{cap[0]}{cap[1]}), CUDA {torch.version.cuda}")
    if cap[0] >= 12 and torch.version.cuda and int(torch.version.cuda.split('.')[0]) < 13:
        print("  warn  this GPU may need a CUDA 13.x build "
              "(rerun: TORCH_CUDA_CHANNEL=cu130 ./setup.sh)")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("  ok    device       mps (Apple Silicon)")
else:
    print("  ok    device       cpu")

if failed:
    sys.exit(f"\nFAILED to import: {', '.join(failed)}")
PYCHECK

# ----------------------------- project assets ------------------------------
say "Checking project assets"
[ -d Data/SCVRP ] \
  && ok "Data/SCVRP ($(ls Data/SCVRP/*.vrp 2>/dev/null | wc -l | tr -d ' ') .vrp instances)" \
  || warn "Data/SCVRP missing — place CVRPLIB .vrp/.sol files there"

[ -f GA_LNN/lnn/lnn_hyx.pt ] \
  && ok "GA_LNN/lnn/lnn_hyx.pt (pretrained LNN weights)" \
  || warn "GA_LNN/lnn/lnn_hyx.pt missing — GA.py will fall back to random init"

[ -d GA_LNN/data ] \
  && ok "GA_LNN/data ($(ls GA_LNN/data/*.pkl 2>/dev/null | wc -l | tr -d ' ') training logs)" \
  || warn "GA_LNN/data missing — generate with LNN_LOG=1 python GA.py"

mkdir -p results
ok "results/ ready"

# ----------------------------- next steps ----------------------------------
cat <<EOS

$(printf "${G}Setup complete.${N}")

  Activate the environment:
      source ${VENV_DIR#"$PROJECT_ROOT"/}/bin/activate

  Run the benchmark:
      python GA.py

  Generate LNN training data:
      LNN_LOG=1 python GA.py

  Train the LNN crossover:
      python -m GA_LNN.train_lnn --log_dir GA_LNN/data --epochs 200 --batch 128

  Note: GA.py is CPU-bound pure Python. When running benchmark blocks
  concurrently, pin the thread count so runs do not oversubscribe cores:
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

EOS
