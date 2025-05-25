# Weekly Report - Week 3

**Project:** RL-based Hyperparameter Scheduling for Distantly Supervised NER (Stage I)  
**Team Members:** Lei Ai, Zhihao Tang, Yang Jiao  
**Date:** May 19 – May 25, 2025

---

## 0. Goal of the Week

- Implement and evaluate two RL-based optimization algorithms:
  - **Proximal Policy Optimization (PPO)**
  - **ε-Greedy Strategy**
- Benchmark performance on `wikigold-distant` dataset.
- Compare against previous GTS and DQN results.
- Investigate effect of training steps on stability and generalization.

---

## 1. PPO Integration and Results

**Summary:**

- Implemented PPO-based hyperparameter controller with policy gradient methods.
- Designed state as a combination of prior F1, dev loss, and step count.
- Policy selects from discretized hyperparameter vector (5D space).
- Total 50 trials conducted.

**Results (wikigold-distant):**

- **Best F1 Score:** **0.6032** (Trial 12) — *exceeds original BOND Stage I performance*.
- **Stability:** Moderate — performance shows fluctuations after peak, but remains above 0.4 in most successful trials.
- **Observation:** PPO outperforms prior DQN significantly. However, best result appears early in training, indicating PPO might require longer learning horizon.

> **Note:** Training steps per PPO agent were limited (~100), which may not be sufficient for policy convergence over a 5D action space.

---

## 2. ε-Greedy Evaluation

**Summary:**

- Discretized hyperparameter vectors into fixed bins.
- Used ε-greedy strategy to explore-exploit parameter combinations.
- Fixed ε schedule: decay from 0.9 → 0.1 over trials.

**Results:**

- **Overall Performance:** Unsatisfactory — no consistent learning observed.
- **Best F1:** Below 0.4; majority of trials under baseline.
- **Conclusion:** ε-greedy is not effective for high-dimensional, noisy NER environments and will not be pursued further.

---

## 3. Key Challenges

- **Early performance peaks (PPO, GTS):** Suggests insufficient exploration or early policy overfitting.
- **Limited training budget:** 100 training steps may be too short for effective RL agent convergence.
- **High-dimensional action space (5D):** Increases difficulty for all controllers, especially non-gradient methods like ε-greedy.

---

## 4. Next Week Schedule - Week 4

### 1. PPO Improvement

- Increase training steps per trial to 300+.
- Add reward normalization / baseline advantage computation.
- Evaluate PPO on other datasets (e.g., conll03-distant, webpage-distant).

### 2. Focused Experimentation

- Focus on **PPO** and **GTS**, both showing promising results.
- Compare policies' generalization across datasets.

### 3. Visualization and Reporting

- Prepare intermediate performance heatmaps.
- Document policy behavior and hyperparameter distributions over time.

---

