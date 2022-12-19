#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:10:00
#SBATCH --mem=16GB
#SBATCH --job-name=ssba
#SBATCH --output=./logs/ssba_%j.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate backdoor-attack-env

cd /scratch/am11533/stealthy-syntactical-backdoor-attack

python main.py --model BERT --data sst-2 --data_purity gpt3defend