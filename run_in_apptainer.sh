#!/bin/bash
#SBATCH -J 2BFT15500
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -t 48:00:00
#SBATCH -o all_log/slurm_%x_%j.out
#SBATCH -e all_log/slurm_%x_%j.err

set -euo pipefail

: "${SIF:?Set SIF to the Goal2Pixel Apptainer image path}"
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
APPTAINER_BIND="${APPTAINER_BIND:-$PROJ:$PROJ,/tmp:/tmp}"
CONDA_LIB_DIR="${CONDA_LIB_DIR:-/opt/conda/envs/goal2pixel/lib}"

PROJECT_NAME="${PROJECT_NAME:-goal2pixel}"
EXP_NAME="${EXP_NAME:-vlm_config_2B_Lora}"
CUDA_VISIBLE_DEVICES=0,1
GRADIENT_ACC=4
MAX_STEPS=15500
META_FILE_NAME=r2r_rxr/RGB_HisKFSingleColor_ALL_trajectoryMASK.json

mkdir -p "$PROJ/all_log"
cd "$PROJ"

echo "JobID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "Start: $(date)"

module load apptainer >/dev/null 2>&1 || true

apptainer exec --nv \
  -B "$APPTAINER_BIND" \
  --env LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CONDA_LIB_DIR" \
  "$SIF" bash -lc "cd \"$PROJ\" && PROJECT_NAME=\"$PROJECT_NAME\" EXP_NAME=\"$EXP_NAME\" CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GRADIENT_ACC=$GRADIENT_ACC MAX_STEPS=$MAX_STEPS META_FILE_NAME=\"$META_FILE_NAME\" ./run_vlm_finetune.sh"

echo "End: $(date)"
