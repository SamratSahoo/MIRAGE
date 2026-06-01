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
#SBATCH --time=04:00:00

set -euo pipefail

source /home/users/asattira/miniconda3/etc/profile.d/conda.sh
conda activate mirage

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

export MINARI_DATASETS_PATH="$PROJECT_DIR/data/minari"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DEFAULT_CONFIG="$PROJECT_DIR/config/encoder/dual_input_masked.yaml"
config="${1:-$DEFAULT_CONFIG}"
shift || true
extra_args="$@"

echo "==== job $SLURM_JOB_ID on $(hostname) ===="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "config = $config"
echo "extra args = $extra_args"

python -u train_encoder.py --config "$config" $extra_args
