#!/bin/bash
#SBATCH -J6PMCMC
#SBATCH --account=def-jlmciver
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH -t 5:00:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rishavrste@gmail.com
#SBATCH -o Report/Report-%j.out


module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python MCMC.py
