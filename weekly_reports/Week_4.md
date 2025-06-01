# Weekly Report - Week 4

**Project:** RL-based Hyperparameter Scheduling for Distantly Supervised NER (Stage I)
**Team Members:** Lei Ai, Zhihao Tang, Yang Jiao
**Date:** May 26 – June 1, 2025

---

## 0. Goal of the Week

- Implement and evaluate **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm.
- Apply TD3 to multiple datasets under BOND Stage I distant supervision setting.
- Investigate whether **increasing training trials** and **initial random exploration** improves learning quality.

---

## 1. TD3 Implementation and Results

### Algorithm Setup:

- TD3 controller operates over a **5-dimensional hyperparameter vector** (e.g., learning rate, weight decay, Adam ε, etc).
- Trials: **500 total**, with early **random exploration (ε=1.0)**:
  - `webpage-distant`: first 100 trials random
  - `wikigold-distant`: first 15 trials random

---

### Results: `webpage-distant`

- **Best F1:** **0.6403** at **Trial 101**
- **Observation:** TD3 quickly converges post-exploration and maintains stable F1 around 0.55–0.60
- Training curve shows dense learning after transition from random trials.

![f8c72781ed4e238e61c5fb1bedaf651](https://github.com/user-attachments/assets/642e09f9-0de5-4e88-a659-a399138eede8)


---

### Results: `wikigold-distant`

- **Best F1:** **0.5399** at **Trial 227**
- **Observation:** F1 gradually improves after random trials, reaching stable region only after ~200 trials.

![5b6296109523862613f7e55a3214042](https://github.com/user-attachments/assets/b18b7d16-820c-4122-b6e2-dd3267fa7366)


---

## 2. Key Observations

| Dataset              | Random Trials | Total Trials | Best F1 | Stability | Notes                                                |
| -------------------- | ------------- | ------------ | ------- | --------- | ---------------------------------------------------- |
| `webpage-distant`  | 100           | 500          | 0.6403  | High      | Strong convergence after exploration; stable plateau |
| `wikigold-distant` | 15            | 500          | 0.5399  | Moderate  | Needs longer training to stabilize; later peak       |

- **Increasing training length helps**: Compared to previous weeks (100–200 trials), the 500-trial setup yields clearly better performance.
- **Longer random warm-up phase** seems beneficial in complex datasets like `webpage`.

---

## 3. Technical Adjustments

- Raised **total trial number to 500**
- Increased **random exploration trials** from 10–20 (previous) to up to **100**
- Tuned TD3 learning rate and actor/critic update intervals to stabilize early training

---

## 4. Challenges

- Training is **computationally intensive**, especially with deep actor-critic setups.
- Later-stage learning sometimes plateaus; considering value clipping or entropy regularization.
- Hyperparameter space is **continuous and multi-modal**, which increases exploration burden.

---
