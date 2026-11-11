"""PPO whose step size is set by the KL it actually produces, the way rsl_rl / Isaac Lab do it.

WHY. flex_plant-1025-0011 ran approx_kl 0.2-0.8 with 40-55% of samples clipped for 60M steps:
every update landed far outside the trust region, and the run never beat the law it trained
over. flex_plant_v2 tried SB3's own fix, target_kl, which only stops an update's EPOCHS early
once KL is crossed -- the step that crossed it is still taken, and the LR that caused it stays.

WHAT. After every minibatch's forward pass, before its optimizer step, the KL between the
rollout policy and the current one on that minibatch decides the LR for the step
(rsl_rl/algorithms/ppo.py, "adaptive" schedule):

    kl > 2 * desired_kl                -> lr / 1.5
    0 < kl < desired_kl / 2            -> lr * 1.5      (kl == 0: actor frozen, leave it)
    lr clamped to [lr_min, lr_max]

No LR schedule and no clip decay: the KL is the schedule. The LR is model state, so a resumed
run continues from the LR it had, not from the config's starting value.

KL estimator: SB3's (r - 1) - log r on the minibatch's stored log-probs (Schulman's k3), not
rsl_rl's closed-form Gaussian KL: SB3's buffer does not keep the rollout's mu/sigma, and k3 is
an unbiased estimate of the same KL. train/approx_kl is logged over ALL minibatches of the
update (SB3 logs only the last epoch's), next to train/learning_rate.

SMOOTHNESS (lcp_lambda > 0; LCP, "Lipschitz-constrained policies"). Adds lcp_lambda * the mean
squared Frobenius norm of d mu / d obs_norm to the loss: a policy whose ACTION MEAN changes little
when its normalized input changes little cannot turn sensor noise into twitch. It is mu that is
penalized, not log pi as in the paper: with our 0.05 rad std, the log-pi gradient is 2-500x
larger and would swamp the term. Estimated with one Hutchinson probe per sample
(E_v ||d(mu . v)/d obs||^2 = ||J||_F^2 for v ~ N(0, I)): one extra backward, not 17.
lcp_batch > 0 estimates it on that many random samples of each minibatch instead of all of
them: the same expected penalty for a fraction of the double backward, which on the full
6144-sample minibatch more than doubled the update's cost.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


def lcp_penalty(policy, obs: th.Tensor) -> th.Tensor:
    """Mean over the batch of ||d mu / d obs||_F^2 (one Hutchinson probe), differentiable."""
    obs = obs.detach().requires_grad_(True)
    mu = policy.get_distribution(obs).distribution.mean
    (grad,) = th.autograd.grad((mu * th.randn_like(mu)).sum(), obs, create_graph=True)
    return grad.pow(2).sum(dim=-1).mean()


class AdaptiveKLPPO(PPO):

    def __init__(self, *args, desired_kl: float = 0.01, lr_min: float = 1e-5, lr_max: float = 1e-2,
                 lcp_lambda: float = 0.0, lcp_batch: int = 0, **kwargs):
        kwargs.setdefault("target_kl", None)
        super().__init__(*args, **kwargs)
        if self.target_kl is not None:
            raise ValueError("AdaptiveKLPPO sets the LR from the KL; target_kl must be None")
        self.desired_kl, self.lr_min, self.lr_max = float(desired_kl), float(lr_min), float(lr_max)
        self.lcp_lambda, self.lcp_batch = float(lcp_lambda), int(lcp_batch)
        # The starting LR. PPO.load() builds the object without a model first, then restores the
        # saved adaptive_lr over this, so a resumed run continues from where its LR had got to.
        lr0 = self.learning_rate(1.0) if callable(self.learning_rate) else float(self.learning_rate)
        self.adaptive_lr = float(np.clip(lr0, self.lr_min, self.lr_max))

    def adapt_lr(self, kl: float) -> float:
        """The rsl_rl rule for one minibatch; returns (and stores) the LR to step with."""
        if kl > 2.0 * self.desired_kl:
            self.adaptive_lr = max(self.lr_min, self.adaptive_lr / 1.5)
        elif 0.0 < kl < 0.5 * self.desired_kl:
            self.adaptive_lr = min(self.lr_max, self.adaptive_lr * 1.5)
        return self.adaptive_lr

    def _set_lr(self, lr: float) -> None:
        for group in self.policy.optimizer.param_groups:
            group["lr"] = lr

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._set_lr(self.adaptive_lr)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses, pg_losses, value_losses, clip_fractions, kls, lrs = [], [], [], [], [], []
        lcps = []
        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_ratio = log_prob - rollout_data.old_log_prob
                ratio = th.exp(log_ratio)
                with th.no_grad():
                    kl = float(th.mean((ratio - 1) - log_ratio).cpu())
                kls.append(kl)
                self._set_lr(self.adapt_lr(kl))
                lrs.append(self.adaptive_lr)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if self.lcp_lambda > 0:
                    obs = rollout_data.observations
                    if 0 < self.lcp_batch < len(obs):
                        obs = obs[th.randperm(len(obs), device=obs.device)[:self.lcp_batch]]
                    lcp = lcp_penalty(self.policy, obs)
                    lcps.append(lcp.item())
                    loss = loss + self.lcp_lambda * lcp
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", float(np.mean(kls)))
        self.logger.record("train/approx_kl_max", float(np.max(kls)))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/learning_rate", self.adaptive_lr)
        self.logger.record("train/learning_rate_mean", float(np.mean(lrs)))
        if lcps:
            self.logger.record("train/lcp_penalty", float(np.mean(lcps)))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
