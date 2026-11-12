#!/bin/bash
#SBATCH --job-name=cross-market-ml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
# Activate the Python environment and supply allocation-specific partition/account/GPU flags.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch, not on a login node}"
cd "$SLURM_SUBMIT_DIR"
export RUN_MODE=full
export DATA_METHOD=local
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python - <<'PY'
from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient

job_id = os.environ["SLURM_JOB_ID"]
output = Path("results") / f"run_{job_id}.executed.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
notebook = nbformat.read('Cross_Market_Transfer_Learning.ipynb', as_version=4)
try:
    NotebookClient(
        notebook, timeout=None, kernel_name='cross-market-wrds',
        resources={'metadata': {'path': str(Path.cwd())}},
    ).execute()
finally:
    nbformat.write(notebook, output)
PY
