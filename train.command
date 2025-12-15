#!/bin/zsh

# go to your project directory
cd "/Users/pratyushgupta/Documents/Pneumonia-Severity-Classification" || exit

# activate virtual environment
source tf_env/bin/activate

# run model training
python src/train_cnn.py

# keep terminal open when finished
echo "Training complete — press Enter to close."
read

