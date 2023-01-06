#!/bin/bash
#SBATCH -J train
#SBATCH -o train.out
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 72:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

# TO BE CHANGED TO YOUR TRAINING AND VALIDATION SET DIRECTORIES

DATABASE_DIR="/mnt/external.data/TowbinLab/spsalmon/moult_database/fluo/"
TRAINING_DIR="${DATABASE_DIR}training/"
VALIDATION_DIR="${DATABASE_DIR}validation/"

CHECKPOINT_DIR="./checkpoints/"

# YOU CAN CHANGE THE LEARNING RATE HERE. BASE VALUE IS 1e-4

LEARNING_RATE=0.00001

# CHOOSE THE NUMBER OF LAYERS YOU WANT TO USE. CAN ONLY BE 50, 101 OR 152

LAYERS=50

# REPLACE BY THE CLASSES YOU WANT TO CLASSIFY THE IMAGES INTO

CLASSES="[not_moulting, moulting]"

python3 train.py -b 6 -e 31 -t 1 -d 1184 --training-dir "$TRAINING_DIR" --validation-dir "$VALIDATION_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --learning-rate $LEARNING_RATE --layers $LAYERS --classes "$CLASSES"