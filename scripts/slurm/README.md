# Running on a Slurm cluster (one 24 GB GPU)

The pipeline needs one GPU with 24 GB (an RTX 4090 is enough: the 7B model takes
about 15 GB in bf16, without quantization). `job.sh` creates a venv on top of an
existing CUDA PyTorch, installs the project, downloads the model and runs the
pipeline. Every step is idempotent, so a requeued job resumes where it stopped.

```bash
# on the login node
git clone https://github.com/Giuliohbb/code-watermark-robustness ~/code-watermark-robustness
cd ~ && mkdir -p runs
sbatch -p <partition> -A <account> code-watermark-robustness/scripts/slurm/run.sbatch

squeue -u $USER                    # PD = waiting, R = running
tail -f runs/cwm-<JOBID>.out
```

Results: `~/runs/cwm/pilot/report/report.md`.

Other configs use the same script, e.g. the full run:

```bash
CONFIG_NAME=humaneval_full sbatch -p <partition> -A <account> --time=12:00:00 \
  code-watermark-robustness/scripts/slurm/run.sbatch
```

## Cluster-specific settings

Anything specific to your cluster goes in `~/.cwm-slurm.env` (outside the repo),
which `run.sbatch` and `prefetch.sh` read if it exists and export to the job:

```bash
CONTAINER_IMAGE=/path/to/pytorch-image.sqsh   # optional: run inside a Pyxis/enroot container
CONTAINER_MOUNTS="$HOME:$HOME"                 # optional: extra mounts for the container
STEP_GRES=gpu:1                                # optional: GPU request repeated on the srun step
HF_HOME=$HOME/.cache/huggingface               # optional: model cache on fast storage
PATH=/usr/bin:/bin                             # optional: if the cluster requires a fixed PATH in jobs
```

If the log says the job has no internet access and the model is missing, run
`bash code-watermark-robustness/scripts/slurm/prefetch.sh` on the login node and
submit again.
