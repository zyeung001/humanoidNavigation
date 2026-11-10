"""Actor-critic whose critic sees the true sim state and whose actor never does.

WHY. The value function only has to be right in simulation: it is thrown away at export, so it
may read what no robot sensor gives -- the episode's servo lag and kp scale, the push being
applied, the base velocity, clean gravity (StandingEnv privileged_critic). A critic that knows
it is being pushed can tell a bad action from a bad moment, which is what the 228 noisy actor
dims cannot (rsl_rl / Isaac Lab legged-robot recipes all do this).

HOW. StandingEnv appends the privileged block AFTER the actor's observation, so the policy's
input is [actor_obs (actor_obs_dim) | privileged]. VecNormalize normalizes both. The actor MLP
is built on, and only ever fed, features[..., :actor_obs_dim]; the critic MLP gets everything.
So the exported actor (scripts/deploy/export_policy.py) is the same 228-input network as
before, and its obs_mean / obs_var are the first actor_obs_dim entries of the normalizer.
"""

from __future__ import annotations

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


class AsymmetricMlpExtractor(nn.Module):
    """SB3's MlpExtractor with the actor cut down to the first actor_dim features."""

    def __init__(self, feature_dim: int, actor_dim: int, net_arch, activation_fn, device):
        super().__init__()
        if not 0 < actor_dim < feature_dim:
            raise ValueError(f"actor_dim {actor_dim} must be inside the {feature_dim} features")
        self.actor_dim = int(actor_dim)
        actor = MlpExtractor(self.actor_dim, net_arch, activation_fn, device)
        critic = MlpExtractor(feature_dim, net_arch, activation_fn, device)
        self.policy_net, self.latent_dim_pi = actor.policy_net, actor.latent_dim_pi
        self.value_net, self.latent_dim_vf = critic.value_net, critic.latent_dim_vf

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_dim])

    def forward_critic(self, features):
        return self.value_net(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, *args, actor_obs_dim: int, **kwargs):
        self.actor_obs_dim = int(actor_obs_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            self.features_dim, self.actor_obs_dim, self.net_arch, self.activation_fn, self.device)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data["actor_obs_dim"] = self.actor_obs_dim
        return data
