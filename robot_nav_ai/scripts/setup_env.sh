#!/usr/bin/env bash
# setup_env.sh — Environment Setup Script
# Creates a Python virtual environment and installs all project dependencies.
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh              # default: creates .venv in project root
#   VENV_DIR=~/envs/autorobo ./scripts/setup_env.sh

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_MIN="3.11"
REQUIRED_PYTHON="python3.11"

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Check OS ───────────────────────────────────────────────────────────────────
OS="$(uname -s)"
log_info "Detected OS: ${OS}"

if [[ "${OS}" == "Darwin" ]]; then
    # macOS — MuJoCo installs via pip (no system libs needed for mujoco>=2.3)
    log_info "macOS detected — MuJoCo 3.x installs via pip on Apple Silicon/Intel."
elif [[ "${OS}" == "Linux" ]]; then
    log_info "Linux detected — checking for GL/EGL dependencies..."
    if command -v apt-get &>/dev/null; then
        log_info "Installing system dependencies for MuJoCo rendering..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            libgl1-mesa-dev \
            libgles2-mesa-dev \
            libegl1-mesa-dev \
            libglfw3-dev \
            libglew-dev \
            patchelf \
            ffmpeg
        log_ok "System dependencies installed."
    else
        log_warn "Non-apt Linux detected. Install MuJoCo rendering libs manually."
    fi
else
    log_error "Unsupported OS: ${OS}. This script supports macOS and Linux only."
    exit 1
fi

# ── Check Python version ────────────────────────────────────────────────────────
log_info "Checking Python version..."
if ! command -v "${REQUIRED_PYTHON}" &>/dev/null; then
    log_error "${REQUIRED_PYTHON} not found. Install Python ${PYTHON_MIN}+ from python.org or via pyenv."
    exit 1
fi

PYTHON_VERSION="$("${REQUIRED_PYTHON}" --version 2>&1 | awk '{print $2}')"
log_ok "Found Python ${PYTHON_VERSION}"

# ── Create virtual environment ──────────────────────────────────────────────────
if [[ -d "${VENV_DIR}" ]]; then
    log_warn "Virtual environment already exists at ${VENV_DIR}."
    read -r -p "Recreate it? [y/N] " RECREATE
    if [[ "${RECREATE}" =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_DIR}"
        log_info "Removed old venv."
    else
        log_info "Reusing existing venv."
    fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    log_info "Creating virtual environment at ${VENV_DIR}..."
    "${REQUIRED_PYTHON}" -m venv "${VENV_DIR}"
    log_ok "Virtual environment created."
fi

# ── Activate venv ──────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log_ok "Activated venv: $(which python)"

# ── Upgrade pip / build tools ──────────────────────────────────────────────────
log_info "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet
log_ok "pip upgraded: $(pip --version)"

# ── Install core dependencies ──────────────────────────────────────────────────
log_info "Installing project dependencies from pyproject.toml / requirements..."

# Core RL stack
pip install --quiet \
    "mujoco>=3.1" \
    "gymnasium>=0.29" \
    "stable-baselines3>=2.3" \
    "sb3-contrib>=2.3"

# Hydra config
pip install --quiet \
    "hydra-core>=1.3" \
    "omegaconf>=2.3"

# Perception
pip install --quiet \
    "ultralytics>=8.0" \
    "opencv-python>=4.9" \
    "pillow>=10.0"

# Deep learning
pip install --quiet \
    "torch>=2.2" \
    "torchvision>=0.17"

# Planning / Claude API
pip install --quiet \
    "anthropic>=0.28"

# Memory / vector store
pip install --quiet \
    "chromadb>=0.5"

# Data and logging
pip install --quiet \
    "numpy>=1.26" \
    "pandas>=2.2" \
    "h5py>=3.10" \
    "wandb>=0.17" \
    "tensorboard>=2.16" \
    "matplotlib>=3.8" \
    "seaborn>=0.13"

# Jupyter
pip install --quiet \
    "jupyterlab>=4.0" \
    "ipywidgets>=8.0"

# Dev / testing
pip install --quiet \
    "pytest>=8.0" \
    "pytest-cov>=5.0" \
    "ruff>=0.4" \
    "mypy>=1.10"

log_ok "All dependencies installed."

# ── Install project in editable mode ───────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    log_info "Installing project in editable mode..."
    pip install -e "${PROJECT_ROOT}" --quiet
    log_ok "Project installed (editable)."
else
    log_warn "No pyproject.toml found — skipping editable install."
fi

# ── Verify MuJoCo ──────────────────────────────────────────────────────────────
log_info "Verifying MuJoCo installation..."
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__} OK')" && log_ok "MuJoCo verified." || {
    log_error "MuJoCo import failed. Check installation."
    exit 1
}

# ── Print activation instructions ──────────────────────────────────────────────
echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""
echo "  To activate the environment:"
echo -e "    ${YELLOW}source ${VENV_DIR}/bin/activate${NC}"
echo ""
echo "  To start training navigation policy:"
echo -e "    ${YELLOW}python scripts/train_nav.py${NC}"
echo ""
echo "  To launch Jupyter Lab:"
echo -e "    ${YELLOW}jupyter lab${NC}"
echo ""
