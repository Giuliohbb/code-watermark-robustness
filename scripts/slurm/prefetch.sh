#!/usr/bin/env bash
# Fallback for when jobs have no internet access: download the model and the
# EvalPlus data on the login node, into the same cache the jobs read.
# Use the same HF_HOME as the jobs (e.g. from your env file).
set -euo pipefail
ENV_FILE="${CWM_ENV_FILE:-$HOME/.cwm-slurm.env}"
# shellcheck disable=SC1090
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi  # exported to the job
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
python3 -m pip install --user --quiet "huggingface_hub>=0.23" "evalplus>=0.3.1"
python3 -c "from huggingface_hub import snapshot_download as d; print(d('${MODEL}'))"
python3 -c "from evalplus.data import get_human_eval_plus as h; print(len(h()), 'HumanEval+ tasks')"
