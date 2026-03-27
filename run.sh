#!/bin/bash
# 6INTELSY Final Project - One Command Reproduce Script
# Authors: Justin Errol L. Priniel & Jaycen John C. Carreon

echo "Setting up environment..."
pip install -r requirements.txt

echo "Running Data Pipeline..."
python data/get_data.py

echo "Creating necessary directories..."
mkdir -p experiments/results

echo "Running Training, Ablations, and Evaluation..."
python src/eval.py

echo "Pipeline complete! Check the experiments/results folder for plots."
