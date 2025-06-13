#!/usr/bin/env bash
set -euo pipefail

# Install Miniconda
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_SH="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_DIR="$HOME/miniconda"

if ! command -v conda &>/dev/null; then
  echo "▶ Downloading Miniconda…"
  wget -q "${MINICONDA_URL}" -O "${MINICONDA_SH}"
  echo "▶ Installing Miniconda (batch mode)…"
  bash "${MINICONDA_SH}" -b -p "${MINICONDA_DIR}"
  rm -f "${MINICONDA_SH}"
fi

# Initialise conda for this script and future shells
eval "$("${MINICONDA_DIR}/bin/conda" shell.bash hook)"
conda init bash >/dev/null 2>&1 || true   # won’t hurt if already initialised

# Clone the repo
if [ ! -d "lerobot" ]; then
  echo "▶ Cloning huggingface/lerobot…"
  git clone https://github.com/huggingface/lerobot.git
fi

# Create and activate the environment
echo "▶ Creating conda env ‘lerobot’ (if missing)…"
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install dependencies
echo "▶ Installing ffmpeg 7.1.1 from conda-forge…"
conda install -y -c conda-forge ffmpeg=7.1.1

echo "▶ Installing lerobot in editable mode…"
pip install -e ./lerobot

echo "▶ Installing wandb…"
pip install wandb

echo "Setup complete. To use later, just run: conda activate lerobot"
