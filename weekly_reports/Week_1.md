# Weekly Report - Week 1  
**Project:** RL-based Hyperparameter Scheduling for Distantly Supervised NER (Stage I)  
**Team Members:** Lei Ai, Zhihao Tang, Yang Jiao  
**Date:** May 5 – May 11, 2025  

---

### 0. Problem Scope Refinement
- We finalized the **focus of the project to be exclusively on Stage I** of the BOND framework (i.e., the BERT-assisted distant supervision phase).
- Stage II (self-training with pseudo-labeling) will not be modified or included in this project.

### 1. Baseline Setup and Replication
- Successfully **cloned and set up the BOND codebase**.
- Completed environment configuration and data preprocessing.
- **Replicated baseline results** for `conll03-distant` and `ontonotes5-distant` datasets using RoBERTa, achieving comparable F1 scores to the original paper.
- Confirmed the compatibility of our RL scheduler with the BOND Stage I structure.

### 2. Reinforcement Learning Algorithm Survey
- Conducted a literature and framework survey to identify **suitable RL algorithms** for dynamic hyperparameter optimization in noisy NER settings.
- Shortlisted candidate algorithms:
  - ε-greedy (discrete action selection with exploration-exploitation trade-off)
  - Gaussian Thompson Sampling (Bayesian sampling for performance-driven selection)
  - DQN (Deep Q-Learning for discrete control)
  - PPO (Policy gradient-based optimization with stability guarantees)
  - Soft Actor-Critic (Entropy-regularized RL for robustness and exploration)

---

# Next Week Schedule - Week 2

### Initial RL Experiments
- Implement and test two **lightweight RL strategies** in Stage I training:
  - **ε-greedy** for learning rate selection.
  - **Gaussian Thompson Sampling** for weight decay and ADAM parameters.
- Focus on improved **training stability** and **reduction in loss variance** during the first 10 epochs on `wikigold-distant`.

