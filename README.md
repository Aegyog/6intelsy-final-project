Personalized Recommendation with Contextual Bandits (NLP + RL)

Course: 6INTELSY Final Project

Authors: Justin Errol L. Priniel & Jaycen John C. Carreon

Department: School of Computing, Holy Angel University

Project Overview

This project implements a hybrid recommendation architecture designed to solve the "cold-start" problem in digital news. By combining Deep Learning (1D Text-CNN) for semantic feature extraction and Reinforcement Learning (LinUCB Contextual Bandit) for dynamic policy updates, the system adapts to user preferences in real-time.

Key Features

NLP & CNN Backbone: Utilizes a 1D Text-CNN with parallel filter banks (sizes 2, 3, and 4) to capture semantic patterns in news titles.

Contextual Bandit Agent: Implements the LinUCB algorithm to balance exploration of new topics with the exploitation of known user interests.

Offline Simulation: Evaluated using a Replay Simulator on the Microsoft News Dataset (MIND-small).

Bias Mitigation: Uses bandit uncertainty bounds to prevent "filter bubbles" and ensure content diversity.

Repository Structure

src/: Core Python source code for models and evaluation.

notebooks/: Jupyter notebooks for Exploratory Data Analysis (EDA).

experiments/results/: Generated plots for Ablation studies and Slice analysis.

docs/: Final Report, Slide Deck, and Defense Script.

Quick Start

To reproduce the experiments and generate the result plots:

Install dependencies:

pip install -r requirements.txt


Run the evaluation pipeline:

bash run.sh


(Windows users can run python src/eval.py directly)

Results Summary

Our experiments confirm that:

Low Exploration ($\alpha=0.1$): Leads to faster convergence and higher terminal CTR in sparse news environments.

High-Dimensional Embeddings (128-dim): Provide superior semantic resolution compared to 32-dim versions, resulting in a 15% increase in reward accuracy.

Categorical Performance: The model excels in niche categories like Finance and Sports but requires more complex features for broad categories like Entertainment.