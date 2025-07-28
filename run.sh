#!/bin/bash
#SBATCH -J MYGPUJOB         # Job name
#SBATCH -o MYGPUJOB.o%j     # Name of job output file
#SBATCH -e MYGPUJOB.e%j     # Name of stderr error file
#SBATCH -p gpu              # Queue (partition) name for GPU nodes
#SBATCH -N 1                # Total # of nodes per instance
#SBATCH -n 4                # Total # of CPU cores (adjust as needed)
#SBATCH --clusters=edge     # *** CRITICAL: Directs the job to the edge cluster nodes (gn11-gn14) ***
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1            # Request 1 GPU (adjust if more needed)
#SBATCH --mem=79G           # Memory (RAM) requested for gpu-mig
#SBATCH -t 24:00:00         # Run time (hh:mm:ss) for gpu-mig (adjust if needed)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# ============================================================================
# USAGE:
#   sbatch run.sh train              # Normalize then train
#   sbatch run.sh tune               # Normalize then tune
#   sbatch run.sh tune-resume        # Normalize then resume tuning
# ============================================================================

# Get command from first argument (default to train)
COMMAND=${1:-train}

# Change to the directory from which the script was submitted
cd "$SLURM_SUBMIT_DIR"

# Print job information
echo "=========================================="
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Command: $COMMAND"
echo "Working directory: $PWD"
echo "=========================================="

# Dynamically locate Conda initialization script
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting."
    exit 1
fi

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate your Conda environment
conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'. Exiting."; exit 1; }

# Check if the 'module' command is available
if command -v module &> /dev/null; then
    # Initialize module system
    source /usr/share/Modules/init/bash 2>/dev/null || \
    source /etc/profile.d/modules.sh 2>/dev/null || \
    echo "Warning: Modules system not initialized, but proceeding."

    # Load CUDA module
    module load cuda11.8 2>/dev/null || echo "Warning: Failed to load CUDA module. Proceeding with system defaults."
else
    echo "Warning: 'module' command not found. Proceeding with system defaults."
fi

# Check for CUDA compatibility
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA environment or GPU not detected. Exiting."
    exit 1
fi

# Print GPU information
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Print Python and PyTorch information
echo "Python version:"
python --version
echo ""
echo "PyTorch version and CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo "=========================================="

# Function to check exit status
check_status() {
    if [ $1 -ne 0 ]; then
        echo "Error: $2 failed with exit code $1"
        exit $1
    fi
}

# ============================================================================
# STEP 1: NORMALIZE DATA
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 1: Normalizing data..."
echo "=========================================="
python src/main.py normalize
check_status $? "Data normalization"
echo "Data normalization completed successfully!"

# ============================================================================
# STEP 2: RUN MAIN COMMAND
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 2: Running $COMMAND..."
echo "=========================================="

case "$COMMAND" in
    train)
        echo "Starting model training..."
        python src/main.py train
        check_status $? "Model training"
        echo "Model training completed successfully!"
        ;;
    
    tune)
        echo "Starting hyperparameter tuning..."
        # You can adjust --num-trials as needed
        python src/main.py tune --num-trials 100
        check_status $? "Hyperparameter tuning"
        echo "Hyperparameter tuning completed successfully!"
        ;;
    
    tune-resume)
        echo "Resuming hyperparameter tuning..."
        python src/main.py tune --resume --num-trials 50
        check_status $? "Hyperparameter tuning (resume)"
        echo "Hyperparameter tuning (resume) completed successfully!"
        ;;
    
    train-profile)
        echo "Starting model training with profiling..."
        python src/main.py train --profile --profile-epochs 2
        check_status $? "Model training with profiling"
        echo "Model training with profiling completed successfully!"
        ;;
    
    train-best)
        echo "Training with best hyperparameters..."
        # This assumes you've already run tune and have best_config.json
        if [ -f "models/trained_model/hyperparam_search/best_config.json" ]; then
            python src/main.py train --config models/trained_model/hyperparam_search/best_config.json
            check_status $? "Model training with best config"
            echo "Model training with best config completed successfully!"
        else
            echo "Error: best_config.json not found. Run 'tune' first."
            exit 1
        fi
        ;;
    
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo "Valid commands: train, tune, tune-resume, train-profile, train-best"
        exit 1
        ;;
esac

# ============================================================================
# JOB COMPLETION
# ============================================================================
echo ""
echo "=========================================="
echo "Job completed successfully on $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="

# Optional: Copy important results to a safe location
# Uncomment and modify as needed:
# RESULTS_DIR="/path/to/safe/storage/${SLURM_JOB_ID}"
# mkdir -p "$RESULTS_DIR"
# cp -r models/trained_model "$RESULTS_DIR/"
# echo "Results copied to: $RESULTS_DIR"