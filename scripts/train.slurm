#!/bin/bash
#SBATCH --job-name=MDFNet
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
# Activate a Python environment with requirements.txt before sbatch.
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"
python -u main.py "$@"
