#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:10:00
#SBATCH --mem=16GB
#SBATCH --job-name=ssba
#SBATCH --output=logs/bert/sst2/bert_sst2_textbugpoison_attack.log

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate backdoor-attack-env

cd /scratch/am11533/stealthy-syntactical-backdoor-attack


# python main.py --model BERT --data sst-2 --data_purity textbugpoison logs/bert/sst2/bert_sst2_textbugpoison_attack.log
# python main.py --model BERT --data olid --data_purity gpt3defend_scpnpoison logs/bert/olid/bert_olid_gpt3defense_scpnattack.log