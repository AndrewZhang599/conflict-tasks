#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J param_recov

# priority

#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/azhan378/hssm_estimates/slurm/slurm_param_recov_%A_%a.out 

# Request runtime, memory, cores
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH -c 12
#SBATCH -N 1


#SBATCH -p batch
#SBATCH --array=0-24

# --------------------------------------------------------------------------------------

# BASIC SETUP
source /users/azhan378/.bashrc  

conda deactivate
conda deactivate
conda activate hssm_oscar_bambi 
# Read in arguments:

python -u "/users/azhan378/hssm_estimates/script for parameter recovery copy.py" --run_index $SLURM_ARRAY_TASK_ID #can change to run gamma or SSP 
