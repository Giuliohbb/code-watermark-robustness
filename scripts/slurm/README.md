# Running on a Slurm cluster (one 24 GB GPU)

One job at a time, up to 2 days, 1x RTX 4090. The job runs inside the lab's
PyTorch container; `job.sh` creates a venv on top of it, installs the project,
downloads the model into `~/dados/hf` and runs the pipeline. Everything is
idempotent, so a job restarted after a reboot resumes where it stopped.

```bash
# on the access machine
ssh <login>@<cluster>
git clone https://github.com/Giuliohbb/code-watermark-robustness ~/code-watermark-robustness
mkdir -p ~/runs ~/dados
sbatch ~/code-watermark-robustness/scripts/slurm/pilot.sbatch

squeue -u $USER                    # PD = waiting, R = running
tail -f ~/runs/cwm-pilot-<JOBID>.out
```

Results: `~/runs/cwm/pilot/report/report.md`.

Other configs use the same script, e.g. the full run:

```bash
sbatch --job-name=cwm-full --time=12:00:00 --export=ALL,CONFIG_NAME=humaneval_full \
  ~/code-watermark-robustness/scripts/slurm/pilot.sbatch
```

If the log says the job has no internet access and the model is missing, run
`bash scripts/slurm/prefetch.sh` on the access machine and submit again.
