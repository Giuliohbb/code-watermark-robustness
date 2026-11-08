#!/bin/bash
# Pilot run on one RTX 4090 of cluster (20 tasks x 3 samples x 4 conditions).
# Submit from the access machine:  sbatch ~/code-watermark-robustness/scripts/slurm/pilot.sbatch
#SBATCH --job-name=cwm-pilot
#SBATCH --partition=gpu
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/%u
#SBATCH --output=/home/%u/runs/%x-%j.out
#SBATCH --error=/home/%u/runs/%x-%j.err

set -euo pipefail
export PATH=/usr/bin:/bin
mkdir -p "$HOME/runs" "$HOME/dados"

CONFIG_NAME="${CONFIG_NAME:-pilot}"
REPO="$HOME/code-watermark-robustness"
IMG=/path/to/pytorch-image.sqsh
MNT="${HOME}:${HOME}"

srun --container-image="$IMG" \
  --container-mounts="$MNT" \
  --container-workdir="$REPO" \
  --gres=gpu:rtx4090:1 \
  bash -lc "bash '$REPO/scripts/slurm/job.sh' '$CONFIG_NAME'"
