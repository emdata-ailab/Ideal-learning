#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -p 2080
#SBATCH -J sbatch-multiple-task
#SBATCH --ntasks 2
#SBATCH --gres gpu:1 
# set partition

#SBATCH -o sbatchs/exp_proOri_T90.out

# run the application
parallel -j 2 --keep-order {} < sbatchs/exp_proOri_T90
