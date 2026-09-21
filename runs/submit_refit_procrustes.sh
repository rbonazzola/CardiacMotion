#!/bin/bash
#SBATCH --job-name=refit_procrustes_full
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#
# Generalized Procrustes Analysis refit against the full (~96k subject)
# cohort, all 6 partitions -- see the script's own docstring for the
# algorithm. This interactive session's cgroup caps total memory at 12GB
# (confirmed via /sys/fs/cgroup/user.slice/user-<uid>.slice/memory.max),
# nowhere near enough for the full cohort (biventricle alone peaks around
# 34GB) -- hence running this as a real batch job with its own allocation
# instead of inline.
#
# Submit with:  sbatch runs/submit_refit_procrustes.sh

set -euo pipefail

REPO_ROOT="/net/scratch/t19767rb/src/CardiacMotion"
cd "$REPO_ROOT"
mkdir -p logs

source "/mnt/iusers01/fatpou01/compsci01/t19767rb/miniforge3/etc/profile.d/conda.sh"
conda activate cardio

python3 scripts/maintenance/refit_procrustes.py
