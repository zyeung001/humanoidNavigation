# src/training/schedules.py
"""Linear decay schedules for SB3 hyperparameters (learning rate, clip range)."""


def lr_schedule(initial_lr: float, final_lr: float, total_steps: int = 0):
    """Linear decay from initial_lr to final_lr.

    Uses SB3's progress_remaining which goes from 1.0 → 0.0 over training.
    The total_steps parameter is accepted for API compatibility but unused.
    """
    def schedule(progress_remaining: float):
        return initial_lr * progress_remaining + final_lr * (1.0 - progress_remaining)
    return schedule


def clip_schedule(initial: float, final: float, total_steps: int = 0):
    """Linear decay from initial to final clip range.

    Uses SB3's progress_remaining which goes from 1.0 → 0.0 over training.
    The total_steps parameter is accepted for API compatibility but unused.
    """
    def schedule(progress_remaining: float):
        return initial * progress_remaining + final * (1.0 - progress_remaining)
    return schedule
