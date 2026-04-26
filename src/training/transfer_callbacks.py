"""
Transfer-training callbacks shared across training scripts.

Extracted from scripts/train_walking.py (verbatim) so other transfer scripts
(train_nav.py) can reuse the same VF warmup / log_std safety logic without
duplicating ~200 lines.

Behavior is identical to the train_walking inline versions; do not diverge.
"""

import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback


class ValueFunctionWarmupCallback(BaseCallback):
    """
    Three-phase warmup for transfer between tasks (e.g. standing → walking,
    walking → nav), with extra VF training.

    Phase 1 (0 to warmup_steps): Policy frozen (requires_grad=False).
        Only value function trains. KL=0 since policy doesn't change.

    Phase 2 (warmup_steps to warmup_steps + rampup_steps): Gradual unfreeze.
        Policy params are unfrozen but updates are scaled by a factor that
        ramps from 0 to max_scale.

    Phase 3 (warmup_steps + rampup_steps onward): Permanent scaling.
        Policy updates are permanently scaled by max_scale. With 17 action
        dims, full-speed PPO updates overshoot the clip boundary at ANY
        learning rate (tested 3e-4 to 5e-6, all produce >95% clip fraction).
        Permanent scaling keeps applied updates conservative while letting
        Adam compute correct gradients and momentum.

    Extra VF Training (all phases):
        After each model.train(), run extra_vf_epochs additional VF-only
        gradient steps using the same rollout buffer. SB3's PPO clipping
        only affects policy loss, not VF loss, so the VF benefits from more
        gradient steps even when clip_fraction is high.

    The value function always trains at full speed (unaffected by scaling).
    log_std is excluded from interpolation (managed by LogStdClampCallback).
    """

    def __init__(
        self,
        warmup_steps: int = 250_000,
        rampup_steps: int = 500_000,
        max_scale: float = 0.3,
        extra_vf_epochs: int = 7,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.rampup_steps = rampup_steps
        self.max_scale = max_scale
        self.extra_vf_epochs = extra_vf_epochs
        self._phase = "init"  # init -> frozen -> ramping -> steady
        self._policy_param_names: set = set()
        self._saved_state = None

    def _on_training_start(self) -> None:
        for name, param in self.model.policy.named_parameters():
            is_value = "vf" in name or "value" in name
            if not is_value:
                self._policy_param_names.add(name)
                param.requires_grad = False
        self._phase = "frozen"
        n_total = sum(1 for _ in self.model.policy.parameters())
        print(f"  [VFWarmup] Frozen {len(self._policy_param_names)}/{n_total} "
              f"params (policy network)")
        print(f"  [VFWarmup] Phase 1: Value-only training for "
              f"{self.warmup_steps:,} steps")
        print(f"  [VFWarmup] Phase 2: Policy ramp 0->{self.max_scale:.1f} over "
              f"{self.rampup_steps:,} steps")
        print(f"  [VFWarmup] Phase 3: Permanent scaling at {self.max_scale:.1f}")

        if self.extra_vf_epochs > 0 and not getattr(self.model, "_extra_vf_patched", False):
            _ppo_train = type(self.model).train
            _model = self.model
            _callback = self

            def _train_with_extra_vf():
                _ppo_train(_model)
                _callback._extra_vf_training()

            self.model.train = _train_with_extra_vf
            self.model._extra_vf_patched = True
            print(f"  [VFWarmup] Extra VF training: {self.extra_vf_epochs} "
                  f"epochs after each train()")

    def _extra_vf_training(self):
        model = self.model
        buffer = model.rollout_buffer
        if not buffer.full:
            return

        grad_state = {}
        for name, param in model.policy.named_parameters():
            grad_state[name] = param.requires_grad
            is_value = "vf" in name or "value" in name
            param.requires_grad = is_value

        model.policy.set_training_mode(True)

        total_loss = 0.0
        for _epoch in range(self.extra_vf_epochs):
            for data in buffer.get(batch_size=None):
                values = model.policy.predict_values(data.observations)
                vf_loss = F.mse_loss(data.returns, values.flatten())

                model.policy.optimizer.zero_grad()
                (model.vf_coef * vf_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.policy.parameters(), model.max_grad_norm
                )
                model.policy.optimizer.step()

                total_loss += vf_loss.item()

        for name, param in model.policy.named_parameters():
            param.requires_grad = grad_state[name]

        model.policy.set_training_mode(False)

        avg_loss = total_loss / max(self.extra_vf_epochs, 1)
        if self.verbose and self.num_timesteps % 100_000 < 25_000:
            print(f"  [VFWarmup] Extra VF: {self.extra_vf_epochs} epochs, "
                  f"avg_loss={avg_loss:.6f}")

    def _on_rollout_start(self) -> None:
        if self._phase not in ("ramping", "steady") or self._saved_state is None:
            return

        if self._phase == "ramping":
            elapsed = self.num_timesteps - self.warmup_steps
            progress = min(elapsed / max(self.rampup_steps, 1), 1.0)
            scale = progress * self.max_scale

            if progress >= 1.0:
                self._phase = "steady"
                scale = self.max_scale
                print(f"\n  [VFWarmup] Phase 3: Steady scale={self.max_scale:.2f} "
                      f"at step {self.num_timesteps:,}")
                self._reset_policy_optimizer()
        else:
            scale = self.max_scale

        with torch.no_grad():
            for name, param in self.model.policy.named_parameters():
                if name in self._saved_state:
                    old = self._saved_state[name]
                    param.data.copy_(old + scale * (param.data - old))
                    self._saved_state[name] = param.data.clone()

        if self.verbose and self.num_timesteps % 100_000 < 25_000:
            print(f"  [VFWarmup] scale={scale:.3f} step={self.num_timesteps:,} "
                  f"({self._phase})")

    def _reset_policy_optimizer(self):
        optimizer = self.model.policy.optimizer
        reset_count = 0
        for name, param in self.model.policy.named_parameters():
            if name in self._saved_state and param in optimizer.state:
                del optimizer.state[param]
                reset_count += 1
        print(f"  [VFWarmup] Reset optimizer state for {reset_count} policy params")

    def _on_step(self) -> bool:
        if self._phase == "frozen" and self.num_timesteps >= self.warmup_steps:
            self._saved_state = {}
            for name, param in self.model.policy.named_parameters():
                if name in self._policy_param_names:
                    param.requires_grad = True
                    if "log_std" not in name:
                        self._saved_state[name] = param.data.clone()
            self._phase = "ramping"
            print(f"\n  [VFWarmup] Phase 2: Ramp starting at step "
                  f"{self.num_timesteps:,}")
        return True


class LogStdClampCallback(BaseCallback):
    """
    Safety net: clamps policy log_std to prevent action distribution explosion.

    When entropy coefficient is too high or value function is unstable,
    log_std can grow exponentially, making all actions pure noise. Clamp
    infrequently (every 2000 steps) to let the policy explore between
    safety checks.
    """

    def __init__(
        self,
        log_std_min: float = -2.0,
        log_std_max: float = 1.0,
        clamp_freq: int = 2000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clamp_freq = clamp_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.clamp_freq == 0:
            if hasattr(self.model.policy, "log_std"):
                with torch.no_grad():
                    log_std = self.model.policy.log_std
                    old_max = log_std.max().item()
                    log_std.clamp_(self.log_std_min, self.log_std_max)
                    new_max = log_std.max().item()
                    if self.verbose and old_max > self.log_std_max:
                        print(f"  [LogStdClamp] Clamped log_std: "
                              f"{old_max:.3f} -> {new_max:.3f} at step "
                              f"{self.num_timesteps:,}")
        return True
