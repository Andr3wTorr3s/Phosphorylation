#!/bin/bash
#SBATCH --job-name=tst
#SBATCH --partition=gpu 
#SBATCH --mail-type=NONE 
#SBATCH --mail-user=andrew.torres@du.edu
#SBATCH --mem=30gb 
#SBATCH --time=24:05:00
#SBATCH --output=out.log
#SBATCH --gres=gpu:v100:1   


# Load modules below
#module load compilers/anaconda-3.8-2020.11
#module load compilers/anaconda-2021.11 
#module load apps/python/3.10.6
module load compilers/anaconda-3.8-2020.11

module load cuda10.1/toolkit/10.1.243
module load libraries/cuDNN/7.6.5


conda activate ml
#module load gnu-parallel/2021.07.22
# Execute commands for application below


#python fluor.py# infer fluoresence
python test.py

