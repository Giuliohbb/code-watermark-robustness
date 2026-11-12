#!/bin/bash
# Runs one experiment config on a single GPU with 24 GB (e.g. an RTX 4090).
#
#   cd ~ && sbatch code-watermark-robustness/scripts/slurm/run.sbatch            # pilot
#   CONFIG_NAME=humaneval_full sbatch --time=12:00:00 code-watermark-robustness/scripts/slurm/run.sbatch
#
# Partition, account and GPU type are cluster-specific: pass them on the command
# line (-p, -A, --gres). Optional settings such as a container image can live in
# a private env file (see scripts/slurm/README.md).
#SBATCH --job-name=cwm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err

set -euo pipefail

ENV_FILE="${CWM_ENV_FILE:-$HOME/.cwm-slurm.env}"
# shellcheck disable=SC1090
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi  # exported to the job

CONFIG_NAME="${CONFIG_NAME:-pilot}"
REPO="${CWM_REPO:-$HOME/code-watermark-robustness}"
JOB="bash '$REPO/scripts/slurm/job.sh' '$CONFIG_NAME'"
STEP_ARGS=()
[[ -n "${STEP_GRES:-}" ]] && STEP_ARGS+=(--gres="$STEP_GRES")

if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  # Pyxis/enroot container with CUDA PyTorch preinstalled.
  srun ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="${CONTAINER_MOUNTS:-$HOME:$HOME}" \
    --container-workdir="$REPO" \
    bash -lc "$JOB"
else
  # Any environment where `python` already has CUDA PyTorch.
  srun ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} bash -lc "$JOB"
fi
