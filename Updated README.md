Personalized Recommendation with Contextual Bandits (NLP + RL)

Course: 6INTELSY Final Project
Authors: Justin Errol L. Priniel & Jaycen John C. Carreon

Overview
This project is an intelligent news recommendation system[cite: 10]. It ranks items by capturing semantic features from text and adapting to shifting user preferences[cite: 10, 12]. It combines NLP embeddings, a 1D Text CNN for feature extraction, and a Contextual Bandit reinforcement learning agent[cite: 12, 16].

v0.9 Progress (Week 2 Checkpoint)
* Data acquired and cleaned with splits finalized.
* EDA notebook completed with click-through rate analysis.
* CNN experiment is running and NLP component is prototyped.
* RL agent is stubbed with reward design and early learning curves.

Dataset and Ethics
We use the Microsoft News Dataset (MIND)[cite: 33]. The data consists of anonymized behavior logs with all personal information removed[cite: 38]. We use exploration strategies to mitigate filter bubbles[cite: 60].

Quick Start
1. Install dependencies: pip install -r requirements.txt
2. Open EDA: notebooks/EDA_MIND_Dataset.ipynb
3. Run Simulator: python src/evaluation/simulator.py
EOF
