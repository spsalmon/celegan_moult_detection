#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# TO BE CHANGED TO YOUR TRAINING AND VALIDATION SET DIRECTORIES

TESTING_DIR="/mnt/external.data/TowbinLab/spsalmon/moult_database/test_database/fluo/"
TESTING_LABELS="$TESTING_DIR""labels.csv"

REPORT_DIR="./test_reports/"

# CHANGE FOR THE PATH OF THE MODEL YOU WANT TO USE

MODEL_PATH="./checkpoints/CP_epoch26.pth"

python3 test.py -b 20 -i "$TESTING_DIR" -o "$REPORT_DIR" -m "$MODEL_PATH"