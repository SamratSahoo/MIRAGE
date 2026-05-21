#!/bin/bash
#SBATCH --job-name=mirage-enc
#SBATCH --output=/scratch/users/%u/mirage/out/%x.%j.out
#SBATCH --error=/scratch/users/%u/mirage/err/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

INPUT_MODE=${1:-xy}
RUN_NAME=${2:-${INPUT_MODE}}

set -euo pipefail

source /home/users/asattira/miniconda3/etc/profile.d/conda.sh
conda activate mirage

export MINARI_DATASETS_PATH=/scratch/users/asattira/mirage/minari
cd /home/users/asattira/MIRAGE

echo "==== job $SLURM_JOB_ID on $(hostname) ===="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "input_mode=${INPUT_MODE}  run_name=${RUN_NAME}"

python -u -m mirage.train_encoder \
    --config configs/encoder_umaze.yaml \
    --input_mode "${INPUT_MODE}" \
    --run_name "${RUN_NAME}"
