#!/bin/bash
#SBATCH -J Transformer
#SBATCH -o Transformer.o%j
#SBATCH -e Transformer.e%j
#SBATCH -p gpu
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH -A exoweather
#SBATCH -t 48:00:00

cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting." >&2; exit 1;
fi

CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'." >&2; exit 1; }

# Try to load any available CUDA module
echo "Looking for available CUDA modules..."
module avail cuda 2>&1 | grep -i cuda || echo "No CUDA modules found via module system"

# Try different common CUDA versions
for cuda_version in cuda/12.7 cuda/12.6 cuda/12.5 cuda/12.4 cuda/12.0 cuda/11.8 cuda/11.7 cuda; do
    if module load "$cuda_version" 2>/dev/null; then
        echo "Successfully loaded $cuda_version"
        break
    fi
done

echo "------------------------------------------------"
echo "Starting Python application..."
python -u src/main.py --train 2>&1
echo "Job completed successfully."
