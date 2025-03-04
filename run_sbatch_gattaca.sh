#!/bin/bash
#SBATCH -J MYJOB            # Job name
#SBATCH -o MYJOB.o%j        # Name of job output file
#SBATCH -e MYJOB.e%j        # Name of stderr error file
#SBATCH -p compute          # Queue (partition) name
#SBATCH -N 1                # Total # of nodes per instance
#SBATCH -n 1                # Total # of cores
#SBATCH --mem=6G            # Memory (RAM) requested
#SBATCH -t 48:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# Change to work directory
# NOTE: /scratch points to the scratch filesystem local
#       to the side of the cluster the job is running on.
#       In this case, /scratch points to the JPL scratch,
#       which is also accessible at /scratch-jpl.

# Dynamically locate Conda initialization script
CONDA_EXE=$(command -v conda)
CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate nn

# Run your Python script
python src/main.py --gen-profiles --normalize