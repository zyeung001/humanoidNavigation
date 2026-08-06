# real_turning_env.py
"""Real-robot turning environment for the hardware WTR (waist-twist) validation.

Built on the StandingEnv substrate (humanoid_real_v2.xml, 228-dim proprioceptive obs,
position actuators, action_smoothing_tau) so the body, sensing, and action path are the
SAME as the deployed standing policy. Adds, for the turning task:

  * an 11-dim command block appended to the obs (same layout as the Humanoid-v5 walking env)
  * a yaw-rate-tracking reward measured through the SHARED src.core.heading.HeadingYaw helper
    -- the heading_source toggle (torso=hack vs pelvis=fix) is the literally identical code
    path used by the sim ablation (see [[real-turning-env]] / project memory).
  * a slow turn-in-place curriculum matched to the no-ankle hip-yaw morphology.

CRITICAL morphology note: humanoid_real_v2's freejoint root is the PELVIS (base_link), the
opposite of Humanoid-v5 (root = torso). So HeadingYaw is built with root_is_torso=False,
torso_body='0003_8' (the chest, above the 3-joint waist chain), pelvis_body='base_link'.
qvel[5] is therefore the PELVIS yaw here; torso yaw comes from mj_objectVelocity('0003_8').
"""
from typing import Optional

import numpy as np
from gymnasium.spaces import Box

from src.core.heading import HeadingYaw

from .standing_env import StandingEnv

# Command-block layout (11 dims, appended once, matching walking_env / config CLAUDE.md):
#   [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, yaw_actual,
#    err_vx, err_vy, err_speed, err_angle, err_yaw]
COMMAND_BLOCK_DIM = 11


class RealTurningEnv(StandingEnv):
    def __init__(self, render_mode: Optional[str] = None, config=None):
        super().__init__(render_mode=render_mode, config=config)
        cfg = self.cfg

        # --- heading source (the experimental variable), through the shared helper ---
        self.heading_source = str(cfg.get('heading_source', 'torso')).lower()
        if self.heading_source not in ('torso', 'pelvis'):
            print(f"WARNING: heading_source={self.heading_source!r} invalid; using 'torso'")
            self.heading_source = 'torso'
        self.torso_body = str(cfg.get('torso_body', '0003_8'))
        self.pelvis_body = str(cfg.get('pelvis_body', 'base_link'))
        self._heading = HeadingYaw(self.heading_source, torso_body=self.torso_body,
                                   pelvis_body=self.pelvis_body, root_is_torso=False)

        # --- command / reward params ---
        self.max_yaw_rate = float(cfg.get('max_yaw_rate', 0.5))      # rad/s ceiling for commands
        self.max_lin_speed = float(cfg.get('max_lin_speed', 0.0))   # turn-in-place: 0 forward by default
        self.reward_yaw_weight = float(cfg.get('reward_yaw_weight', 10.0))
        self.reward_yaw_bandwidth = float(cfg.get('reward_yaw_bandwidth', 10.0))
        self.reward_upright_weight = float(cfg.get('reward_upright_weight', 4.0))
        self.reward_height_weight = float(cfg.get('reward_height_weight', 2.0))
        self.tilt_rate_penalty = float(cfg.get('tilt_rate_penalty', 0.5))

        # --- turn-enabling terms ported from walking_env (the recipe that made v5 turn) ---
        # 1) feet air-time: a planted-foot robot CANNOT rotate in place (friction); reward
        #    foot-lifting so the policy discovers the pivot. (walking_env: "must lift feet to
        #    rotate".) Continuous per-step bonus per airborne foot past min_air_time.
        self.feet_air_time_weight = float(cfg.get('feet_air_time_weight', 1.0))
        self.min_air_time = float(cfg.get('min_air_time', 0.15))
        self.contact_force_threshold = float(cfg.get('contact_force_threshold', 5.0))
        self.foot_body_ids = (self.model_spec.foot_body_ids if self._is_custom else [6, 9])
        self.feet_air_time = np.zeros(len(self.foot_body_ids))
        # 2) turn-gated survival: scale the upright+height floor by how much we ACTUALLY turn
        #    toward the command, so standing still no longer banks the floor (the exploit that
        #    sank the first run). floor stays 1.0 when no turn is commanded (standing is correct).
        self.turn_survival_floor = float(cfg.get('turn_survival_floor', 0.3))
        # 3) wrong-direction penalty: pure-Gaussian yaw reward is flat past err=0.5 rad/s, so it
        #    cannot tell wrong-way from no-turn. Penalize yaw rate opposing the command.
        self.yaw_wrong_dir_penalty = float(cfg.get('yaw_wrong_dir_penalty', 1.5))

        # command state (vx, vy, yaw_rate). fixed_command overrides the generator (eval/replay).
        self.fixed_command = cfg.get('fixed_command', None)
        self.commanded_vx = 0.0
        self.commanded_vy = 0.0
        self.commanded_yaw_rate = 0.0

        # --- extend the obs space by the command block (appended once, after history stack) ---
        base_dim = int(self.observation_space.shape[0])
        self.turning_obs_dim = base_dim + COMMAND_BLOCK_DIM
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.turning_obs_dim,), dtype=np.float32)
        print(f"  RealTurningEnv: heading_source={self.heading_source} "
              f"(torso={self.torso_body}, pelvis={self.pelvis_body}); "
              f"obs {base_dim} + cmd {COMMAND_BLOCK_DIM} = {self.turning_obs_dim}")

    # ----- command generation -----
    def _sample_command(self):
        """Turn-in-place command. fixed_command (eval) wins; else random yaw within ceiling."""
        if self.fixed_command is not None:
            vx, vy, yr = self.fixed_command
        else:
            yr = float(np.random.uniform(-self.max_yaw_rate, self.max_yaw_rate))
            vx = float(np.random.uniform(0.0, self.max_lin_speed)) if self.max_lin_speed > 0 else 0.0
            vy = 0.0
        self.commanded_vx, self.commanded_vy, self.commanded_yaw_rate = vx, vy, yr

    # ----- yaw-rate of the configured heading source (the experimental signal) -----
    def _get_actual_yaw_rate(self) -> float:
        return self._heading.actual_yaw_rate(self.env.unwrapped.model, self.env.unwrapped.data)

    def _actual_planar_vel(self):
        """Root planar linear velocity (world frame x,y)."""
        v = self.env.unwrapped.data.qvel[0:2]
        return float(v[0]), float(v[1])

    # ----- obs: stacked proprioceptive frames + command block -----
    def _command_block(self) -> np.ndarray:
        vx_a, vy_a = self._actual_planar_vel()
        yaw_a = self._get_actual_yaw_rate()
        err_vx = self.commanded_vx - vx_a
        err_vy = self.commanded_vy - vy_a
        cmd_speed = float(np.hypot(self.commanded_vx, self.commanded_vy))
        act_speed = float(np.hypot(vx_a, vy_a))
        err_speed = cmd_speed - act_speed
        err_angle = float(np.arctan2(self.commanded_vy, self.commanded_vx + 1e-9)
                          - np.arctan2(vy_a, vx_a + 1e-9))
        err_angle = float(np.arctan2(np.sin(err_angle), np.cos(err_angle)))
        err_yaw = self.commanded_yaw_rate - yaw_a
        return np.array([self.commanded_vx, self.commanded_vy, self.commanded_yaw_rate,
                         vx_a, vy_a, yaw_a, err_vx, err_vy, err_speed, err_angle, err_yaw],
                        dtype=np.float32)

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        base = super()._process_observation(obs)
        return np.concatenate([base, self._command_block()]).astype(np.float32)

    # ----- turning reward (stay up + track commanded yaw; do NOT penalize commanded yaw) -----
    def _compute_task_reward(self, obs, base_reward, info, action):
        data = self.env.unwrapped.data
        height = self._get_height()
        proj_grav = self._projected_gravity(data.qpos[3:7])
        upright_cos = -float(proj_grav[2])
        standing_up = height >= 1.2 and upright_cos > 0.6

        yaw_actual = self._get_actual_yaw_rate()
        yaw_err = yaw_actual - self.commanded_yaw_rate
        yaw_cmd = self.commanded_yaw_rate

        # --- turn-gated survival factor (kills the stand-still exploit) ---
        # direction_ratio = how much of the commanded yaw we actually achieve (toward the
        # command). When a turn is commanded, the upright+height floor is scaled by this so a
        # non-turning agent can't bank it. No turn commanded -> factor 1.0 (standing is correct).
        if abs(yaw_cmd) > 1e-3:
            direction_ratio = float(np.clip((yaw_actual * np.sign(yaw_cmd)) / abs(yaw_cmd), 0.0, 1.0))
            survival_factor = self.turn_survival_floor + (1.0 - self.turn_survival_floor) * direction_ratio
        else:
            survival_factor = 1.0

        # stay upright at height (now gated by survival_factor so it's not free while standing)
        upright_reward = self.reward_upright_weight * np.exp(-8.0 * (1.0 - upright_cos) ** 2)
        height_err = abs(height - self.base_target_height)
        height_reward = self.reward_height_weight * np.exp(-10.0 * height_err ** 2)
        floor_reward = (upright_reward + height_reward) * survival_factor

        # yaw-rate tracking on the configured heading source (THE experimental term), also gated
        yaw_reward = 0.0
        if standing_up:
            yaw_reward = (self.reward_yaw_weight
                          * np.exp(-self.reward_yaw_bandwidth * yaw_err ** 2) * survival_factor)

        # wrong-direction penalty: yaw rate opposing the command (gives gradient past err=0.5)
        wrong_dir_pen = 0.0
        if abs(yaw_cmd) > 1e-3 and (yaw_actual * np.sign(yaw_cmd)) < 0.0:
            wrong_dir_pen = -self.yaw_wrong_dir_penalty * min(abs(yaw_actual), abs(yaw_cmd))

        # feet air-time: reward foot-lifting so the policy can pivot (planted feet can't rotate)
        feet_air_reward = 0.0
        if self.feet_air_time_weight > 0:
            dt = float(self.env.unwrapped.dt)
            for i, body_id in enumerate(self.foot_body_ids):
                contact_force = float(np.linalg.norm(data.cfrc_ext[body_id]))
                if contact_force > self.contact_force_threshold:
                    self.feet_air_time[i] = 0.0
                else:
                    self.feet_air_time[i] += dt
                    if self.feet_air_time[i] > self.min_air_time:
                        air_bonus = min(self.feet_air_time[i] - self.min_air_time, 0.3) / 0.3
                        feet_air_reward += self.feet_air_time_weight * air_bonus

        # penalize OFF-axis tilt rate (roll/pitch wobble) but NOT the commanded yaw (z) rate
        ang = data.qvel[3:6]
        tilt_rate_pen = -self.tilt_rate_penalty * float(ang[0] ** 2 + ang[1] ** 2)
        control_cost = -0.005 * float(np.sum(np.square(action)))
        rate_pen = -float(self.action_rate_penalty) * float(np.sum(np.square(self._last_action_rate))) \
            if getattr(self, 'action_rate_penalty', 0.0) else 0.0

        reward = (floor_reward + yaw_reward + feet_air_reward + wrong_dir_pen
                  + tilt_rate_pen + control_cost + rate_pen)

        # fall termination (same spirit as standing)
        terminated = bool(height < 0.75 or upright_cos < 0.3)

        info.update({
            'yaw_cmd': yaw_cmd, 'yaw_actual': yaw_actual, 'yaw_err': yaw_err,
            'height': height, 'upright_cos': upright_cos,
            'r_yaw': yaw_reward, 'r_upright': upright_reward, 'r_floor': floor_reward,
            'r_feet_air': feet_air_reward, 'r_wrong_dir': wrong_dir_pen,
            'survival_factor': survival_factor, 'heading_source': self.heading_source,
        })
        return reward, terminated

    # ----- step / reset -----
    def step(self, action):
        # keep the live command in sync (fixed_command may be mutated externally for eval/replay)
        if self.fixed_command is not None:
            self.commanded_vx, self.commanded_vy, self.commanded_yaw_rate = self.fixed_command
        return super().step(action)

    def reset(self, seed: Optional[int] = None):
        self._sample_command()
        self.feet_air_time[:] = 0.0
        return super().reset(seed=seed)


def make_real_turning_env(render_mode=None, config=None):
    return RealTurningEnv(render_mode=render_mode, config=config)
