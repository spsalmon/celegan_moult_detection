#!/bin/bash
#SBATCH -J pred
#SBATCH -o pred.out
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 48:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# TO BE CHANGED TO YOUR TRAINING AND VALIDATION SET DIRECTORIES

EXPERIMENT_DIR="/mnt/external.data/TowbinLab/kstojanovski/20211206_Ti2_10x_wBT264_186_daf-16d_25C_20211206_170249_693/analysis/"
IMAGES_DIR="$EXPERIMENT_DIR""ch2/"
REPORT_DIR="$EXPERIMENT_DIR""report/"

# CHANGE FOR THE PATH OF THE MODEL YOU WANT TO USE

MODEL_PATH="./checkpoints/CP_epoch26.pth"

python3 predict_moults_test.py -b 15 -i "$IMAGES_DIR" -o "$REPORT_DIR" -m "$MODEL_PATH"