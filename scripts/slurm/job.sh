#!/usr/bin/env bash
# Runs one experiment inside the cluster PyTorch container. Safe to re-run: the
# environment is reused, downloads are skipped when cached, and every pipeline
# stage resumes from what is already on disk (the lab restarts jobs from scratch
# if a machine reboots).
#
# Usage (from a job):  bash scripts/slurm/job.sh <config-name>   e.g. pilot
set -euo pipefail

CONFIG_NAME="${1:?usage: job.sh <config-name>}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${REPO_DIR}/configs/${CONFIG_NAME}.yaml"
VENV="${HOME}/venvs/cwm"
OUTPUT_DIR="${HOME}/runs/cwm"

# Model weights are read constantly, so they live in ~/dados (see the lab manual).
export HF_HOME="${HOME}/dados/hf"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

[[ -f "${CONFIG}" ]] || { echo "config not found: ${CONFIG}"; exit 2; }

log "host=$(hostname) job=${SLURM_JOB_ID:-none} config=${CONFIG_NAME}"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

if python -c "import urllib.request as u; u.urlopen('https://pypi.org/simple/pip/', timeout=10)" \
    >/dev/null 2>&1; then
  ONLINE=1
else
  ONLINE=0
  log "this job has NO internet access"
fi

# 1. Environment: a venv on top of the container's PyTorch (never reinstalls torch).
if [[ ! -x "${VENV}/bin/python" ]]; then
  log "creating venv at ${VENV}"
  python -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
STAMP="${VENV}/.installed-$(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo dev)"
if [[ ! -f "${STAMP}" ]]; then
  if [[ "${ONLINE}" == 0 ]]; then
    echo "The project is not installed and pip cannot reach PyPI from this job."
    echo "Send the log to the group: jobs need internet once to install dependencies."
    exit 4
  fi
  log "installing the project"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -e "${REPO_DIR}[plots]"
  rm -f "${VENV}"/.installed-* && touch "${STAMP}"
fi

# 2. Downloads (skipped when cached). Fails early with a clear message if the
#    job has no internet access and nothing is cached yet.
log "checking model and dataset"
python - "${CONFIG}" <<'PY'
import sys
from cwm.config import load_config

config = load_config(sys.argv[1])
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(config.model.name, revision=config.model.revision)
    print("model ready:", path)
    if config.dataset_path is None:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        (get_human_eval_plus if config.dataset == "humaneval" else get_mbpp_plus)()
        print("dataset ready:", config.dataset)
except Exception as error:  # noqa: BLE001
    print(f"\nDownload failed: {error}\n"
          "If jobs have no internet access, run scripts/slurm/prefetch.sh on the "
          "access machine first (see scripts/slurm/README.md).")
    sys.exit(3)
PY
export HF_HUB_OFFLINE=1

# 3. The experiment itself.
log "running cwm on ${CONFIG_NAME}"
cwm run "${CONFIG}" --set "output_dir=${OUTPUT_DIR}"
log "finished; report at ${OUTPUT_DIR}/$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "${CONFIG}")/report/report.md"
