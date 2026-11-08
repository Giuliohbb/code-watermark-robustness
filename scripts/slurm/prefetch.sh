#!/usr/bin/env bash
# Fallback for when jobs have no internet access: download the model and the
# EvalPlus data on the access machine, into the same folders the jobs read.
set -euo pipefail
export HF_HOME="${HOME}/dados/hf"
MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
python3 -m pip install --user --quiet "huggingface_hub>=0.23" "evalplus>=0.3.1"
python3 -c "from huggingface_hub import snapshot_download as d; print(d('${MODEL}'))"
python3 -c "from evalplus.data import get_human_eval_plus as h; print(len(h()), 'HumanEval+ tasks')"
