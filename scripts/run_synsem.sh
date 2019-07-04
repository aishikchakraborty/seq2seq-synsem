#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --ntasks=1
#SBATCH --mem=30000M
#SBATCH --output=synsem.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
###########################
python -u main.py --cuda --num-epoch 8
