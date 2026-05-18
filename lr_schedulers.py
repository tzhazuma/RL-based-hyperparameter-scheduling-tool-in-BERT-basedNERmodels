#!/usr/bin/env python
# coding: utf-8
"""
Learning rate scheduler baselines compatible with BOND NER training pipeline.

Provides three standard LR schedulers:
- StepDecayScheduler:  Reduce LR by factor gamma every step_size epochs
- CosineAnnealingScheduler:  Cosine annealing (Loshchilov & Hutter, 2017)
- ReduceLROnPlateauScheduler:  Reduce LR when validation loss stops improving

All schedulers share an LRSchedulerBase interface:
    step(epoch, metric=None) -> float  (returns new learning rate)
    get_lr() -> float
"""

import math


class LRSchedulerBase:
    """Base class for LR schedulers.

    Args:
        initial_lr: Starting learning rate.
    """
    def __init__(self, initial_lr=1e-5):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    def step(self, epoch, metric=None):
        """Update LR and return new value.

        Args:
            epoch: Current epoch number (0-indexed).
            metric: Validation loss or other metric (used by plateau scheduler).

        Returns:
            Updated learning rate.
        """
        raise NotImplementedError

    def get_lr(self):
        return self.current_lr


class StepDecayScheduler(LRSchedulerBase):
    """Reduce LR by factor gamma every step_size epochs.

    lr = initial_lr * gamma^(floor(epoch / step_size))

    Args:
        initial_lr: Starting learning rate.
        step_size: Number of epochs between LR drops.
        gamma: Multiplicative factor applied each step_size epochs.
    """
    def __init__(self, initial_lr=1e-5, step_size=10, gamma=0.5):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, epoch, metric=None):
        self.current_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        return self.current_lr


class CosineAnnealingScheduler(LRSchedulerBase):
    """Cosine annealing schedule (Loshchilov & Hutter, 2017).

    lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max))

    Args:
        initial_lr: Starting learning rate.
        T_max: Number of epochs for one full cosine cycle.
        eta_min: Minimum learning rate (floor).
    """
    def __init__(self, initial_lr=1e-5, T_max=50, eta_min=1e-7):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, epoch, metric=None):
        self.current_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max)
        )
        return self.current_lr


class ReduceLROnPlateauScheduler(LRSchedulerBase):
    """Reduce LR when validation metric (e.g., loss) stops improving.

    Tracks the best metric (lower is better) and counts epochs without
    improvement. After `patience` bad epochs, multiplies LR by `factor`.

    Args:
        initial_lr: Starting learning rate.
        factor: Multiplicative factor applied on plateau.
        patience: Number of epochs with no improvement before reducing LR.
        min_lr: Floor for learning rate.
    """
    def __init__(self, initial_lr=1e-5, factor=0.5, patience=5, min_lr=1e-7):
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = None
        self.num_bad_epochs = 0

    def step(self, epoch, metric=None):
        if metric is None:
            return self.current_lr

        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.num_bad_epochs = 0
        return self.current_lr


# ---------------------------------------------------------------------------
# Quick unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # StepDecay: 1e-4 halved every 10 epochs -> epoch 49 -> 1e-4 * (0.5^4) = 6.25e-6
    s = StepDecayScheduler(1e-4, step_size=10, gamma=0.5)
    for e in range(50):
        lr = s.step(e)
    expected_s = 1e-4 * (0.5 ** (49 // 10))  # 0.5^4 = 6.25e-6
    assert abs(lr - expected_s) < 1e-12, f"StepDecay: expected {expected_s}, got {lr}"
    print(f"[PASS] StepDecay final LR: {lr:.10e}")

    # Cosine: starts at 1e-4, ends near eta_min (1e-7) at epoch 50
    c = CosineAnnealingScheduler(1e-4, T_max=50)
    for e in range(50):
        lr = c.step(e)
    expected_c = 1e-7 + 0.5 * (1e-4 - 1e-7) * (1 + math.cos(math.pi * 49 / 50))
    assert abs(lr - expected_c) < 1e-12, f"Cosine: expected {expected_c}, got {lr}"
    print(f"[PASS] Cosine final LR: {lr:.10e}")

    # Plateau: best loss=0.44 at epoch 2, then num_bad tracks:
    #   epochs 3-4 (num_bad=2 -> first drop to 5e-5)
    #   epochs 5-6 (num_bad=2 -> second drop to 2.5e-5)
    #   epochs 7-8 (num_bad=2 -> third drop to 1.25e-5)
    p = ReduceLROnPlateauScheduler(1e-4, patience=2, factor=0.5, min_lr=1e-7)
    losses = [0.5, 0.45, 0.44, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52]
    lrs = []
    for e, loss in enumerate(losses):
        lr = p.step(e, metric=loss)
        lrs.append(lr)
    expected_p = 1.25e-5
    assert abs(lrs[-1] - expected_p) < 1e-12, f"Plateau: expected {expected_p}, got {lrs[-1]}"
    print(f"[PASS] Plateau final LR: {lrs[-1]:.10e}")

    # Edge: Plateau without metric returns current LR unchanged
    p2 = ReduceLROnPlateauScheduler(1e-4)
    lr_none = p2.step(0, metric=None)
    assert lr_none == 1e-4
    print(f"[PASS] Plateau None-metric: {lr_none}")

    print("\nAll tests passed.")
