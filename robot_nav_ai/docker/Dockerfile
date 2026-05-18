# ============================================================
# AutoRobo Physical AI — Multi-Stage Dockerfile
# ============================================================
# Stage 1: base — Python 3.11 + system deps + MuJoCo
# Stage 2: training — adds RL training stack
# Stage 3: jupyter — adds Jupyter Lab for analysis
# ============================================================

# ── Stage 1: base ────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="AutoRobo Team"
LABEL description="AutoRobo Physical AI — base image"

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies for MuJoCo rendering (EGL headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # MuJoCo GL/EGL dependencies
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    # MuJoCo linker dependency
    patchelf \
    # Video encoding for rendering
    ffmpeg \
    # Build tools
    gcc \
    g++ \
    git \
    curl \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use EGL for headless rendering (required in Docker without display)
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash autorobo
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel --no-cache-dir

# ── Stage 2: training ─────────────────────────────────────────
FROM base AS training

LABEL description="AutoRobo Physical AI — training image"

# Copy dependency spec first (layer caching)
COPY pyproject.toml ./
# If using requirements.txt instead:
# COPY requirements.txt ./

# Install core dependencies
RUN pip install --no-cache-dir \
    "mujoco>=3.1" \
    "gymnasium>=0.29" \
    "stable-baselines3>=2.3" \
    "sb3-contrib>=2.3" \
    "hydra-core>=1.3" \
    "omegaconf>=2.3" \
    "ultralytics>=8.0" \
    "opencv-python-headless>=4.9" \
    "pillow>=10.0" \
    "torch>=2.2" \
    "torchvision>=0.17" \
    "anthropic>=0.28" \
    "chromadb>=0.5" \
    "numpy>=1.26" \
    "pandas>=2.2" \
    "h5py>=3.10" \
    "wandb>=0.17" \
    "tensorboard>=2.16" \
    "matplotlib>=3.8"

# Copy project source
COPY --chown=autorobo:autorobo . .

# Install project in editable mode
RUN pip install -e . --no-cache-dir || echo "No pyproject.toml found, skipping editable install"

USER autorobo

# Default: run navigation training
CMD ["python", "scripts/train_nav.py"]

# ── Stage 3: jupyter ──────────────────────────────────────────
FROM training AS jupyter

LABEL description="AutoRobo Physical AI — Jupyter analysis image"

USER root

RUN pip install --no-cache-dir \
    "jupyterlab>=4.0" \
    "ipywidgets>=8.0" \
    "seaborn>=0.13"

USER autorobo

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/app/notebooks"]
