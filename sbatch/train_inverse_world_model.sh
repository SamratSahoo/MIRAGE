#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad-lo
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a5000:1
#SBATCH --job-name="mirage-iwm"
#SBATCH --output=sbatch/%A.out
#SBATCH --error=sbatch/%A.err

echo "SLURM_JOBID        = $SLURM_JOBID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "submit directory   = $SLURM_SUBMIT_DIR"

echo "Setting up conda..."
__conda_setup="$('/iliad/u/samrat/anaconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/iliad/u/samrat/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/iliad/u/samrat/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/iliad/u/samrat/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate mirage
echo "conda env = $CONDA_DEFAULT_ENV   python = $(which python)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
echo "working directory = $(pwd)"

export MINARI_DATASETS_PATH="$PROJECT_DIR/data/minari"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

DEFAULT_CONFIG="$PROJECT_DIR/config/inverse_world_model/inverse.yaml"
config="${1:-$DEFAULT_CONFIG}"
shift || true
extra_args="$@"
echo "Using config: $config"
echo "extra args: $extra_args"

cmd="python -u train_inverse_world_model.py --config $config $extra_args"
echo "Running: $cmd"
$cmd

echo "DONE"
