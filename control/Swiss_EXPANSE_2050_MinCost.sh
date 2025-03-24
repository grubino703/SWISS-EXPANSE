#!/bin/bash

#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=170000 # in MB
#SBATCH -o run_Swiss_EXPANSE_2050_MinCost-%A_%a.out

# Activate environment
ml GCCcore/11.2.0
ml Gurobi/9.5.0
ml Python/3.9.6
. ~/baobab_python_env/bin/activate

# Send costoptimal run to cluster
echo "Running minimum cost scenario on node " $(hostname)
~/baobab_python_env/bin/python run_Swiss_EXPANSE_2050_MinCost.py