#!/bin/bash
# Example only: submit on a Slurm-managed host with mounted dataset and environment.
#SBATCH --job-name=mdfnet-fig2
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=fig2-%j.out

set -euo pipefail
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"
python -u analysis/run_fig2_experiment.py \
  --dataset "${DATASET:?Set DATASET to KUL or DTU}" \
  --condition "${CONDITION:?Set CONDITION, e.g. All, region_F, or band_alpha}" \
  --fold "${FOLD:?Set FOLD from 0 through 3}" \
  --data-dir "${DATA_DIR:?Set DATA_DIR to the mounted data directory}" \
  --output-dir "${OUTPUT_DIR:-outputs_fig2}"
