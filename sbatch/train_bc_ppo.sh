#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad-lo
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a5000:1
#SBATCH --job-name="mirage-bcppo"
#SBATCH --output=sbatch/%A.out
#SBATCH --error=sbatch/%A.err

echo "SLURM_JOBID        = $SLURM_JOBID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"

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

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$PYTHONPATH"
export WARP_CACHE_PATH="/tmp/${USER}_warp_cache_${SLURM_JOB_ID}"
mkdir -p "$WARP_CACHE_PATH"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

BC_EXP="${1:-ab_latent_bc}"
PPO_CONFIG="${2:-config/ppo/ab_latent_bc.yaml}"
CKPT="runs/${BC_EXP}/${BC_EXP}.cleanrl_model"

# Run BC warm-start only if no checkpoint exists yet (so a preemption-requeue
# resumes the PPO run's own progress instead of overwriting it with a fresh BC).
if [ ! -f "$CKPT" ]; then
  echo "=== BC warm-start -> $CKPT ==="
  python -u train_bc_latent.py --exp_name "$BC_EXP" || { echo "BC FAILED"; exit 1; }
else
  echo "=== checkpoint exists ($CKPT); skipping BC, resuming PPO ==="
fi

echo "=== PPO finetune: $PPO_CONFIG ==="
python -u train_ppo.py --config "$PPO_CONFIG"
echo "DONE"
