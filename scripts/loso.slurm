#!/bin/bash
#SBATCH --job-name=MDFNet-LOSO
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-loso-%j.out

set -euo pipefail
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"
python -u run_loso.py "$@"
