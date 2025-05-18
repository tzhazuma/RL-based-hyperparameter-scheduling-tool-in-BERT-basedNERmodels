# Weekly Report - Week 2

**Project:** RL-based Hyperparameter Scheduling for Distantly Supervised NER (Stage I)
**Team Members:** Lei Ai, Zhihao Tang, Yang Jiao
**Date:** May 12 – May 18, 2025

---

## 0. Goal of the Week

- Implement, integrate, and evaluate two RL-based hyperparameter optimization algorithms:
  - **Gaussian Thompson Sampling (GTS)**
  - **Deep Q-Network (DQN)**
- Apply them to `conll03-distant` and `webpage-distant` datasets in the BOND Stage I pipeline.
- Focus on evaluating early-stage training F1 scores and performance stability.

---

## 1. Gaussian Thompson Sampling (GTS) Implementation

**Algorithm Summary:**

- Used Gaussian Thompson Sampling to select weight decay and ADAM ε hyperparameters.
- Set initial trials as random exploration, followed by posterior-guided exploitation.
- Reward is based on dev set F1 score after short training.

### conll03-distant

- **Total Trials:** 26
- **Random Trials:** First 10
- **GTS Trials:** Trial 11–25
- **Best F1:** 73.76 (Trial 11, GTS)
- **Original Best (baseline):** 75.61
- **Observation:** GTS achieved best performance shortly after transition (Trial 11), and maintained stable F1 above 71 in later trials.

### webpage-distant

- **Total Trials:** 100
- **Random Trials:** First 20
- **GTS Trials:** Trial 21–100
- **Best F1:** 62.75 (Trial 4, random)
- **Original Best (baseline):** 59.11
- **Observation:** Despite best trial being in random phase, GTS maintained consistent 0.5–0.6 F1 band after Trial 20, outperforming baseline in overall stability.

---

## 2. Deep Q-Network (DQN) Integration

**Algorithm Summary:**

- Discretized hyperparameter space (e.g., learning rate, weight decay).
- States: training step, previous F1 score, loss delta.
- Reward: improvement in dev F1, penalized by loss.
- 30 trials conducted, each representing a single training instance.

### Key DQN Results (wikigold-distant)

- **Best F1 Score:** 15th trial: **0.05714**
- **Worst F1 Score:** several trials < 0.01
- **Mean F1 Score:** ~0.028 (across 30 trials)
- **Loss Range:** 0.65 – 0.78
- **Observation:** Current DQN controller has not learned an effective policy yet. Reward sparsity and unstable value estimates likely caused learning difficulty. Additional tuning or advanced variants (e.g., Double DQN, reward shaping) may help.

---

## 3. Comparative Analysis

| Method | Dataset          | Best F1 | Mean F1 (Late Trials) | Stability   | Notes                                          |
| ------ | ---------------- | ------- | --------------------- | ----------- | ---------------------------------------------- |
| GTS    | conll03-distant  | 73.76   | ~72.4                 | High        | Stable after transition to TS, near baseline   |
| GTS    | webpage-distant  | 62.75   | ~55.0                 | Medium-High | Best found early, GTS maintains good stability |
| DQN    | wikigold-distant | 0.057   | ~0.028                | Low         | Unstable policy, ineffective reward shaping    |

---

## 4. Challenges Encountered

- GTS sometimes achieves best performance early, making later updates less effective.
- DQN’s state and reward design needs redesign to improve reward signal clarity.
- DQN trials suffer from severe underfitting due to shallow reward or insufficient training steps.

---

## Next Week Schedule - Week 3

### 1. Refine DQN Setup

- Introduce reward normalization / advantage shaping.
- Try Double DQN or Dueling DQN to reduce Q bias.
- Increase training steps per trial.

### 2. Add Baseline Algorithms

- Implement **ε-greedy** and **Bayesian UCB** as baseline RL strategies.
- Compare with static grid and random search.
