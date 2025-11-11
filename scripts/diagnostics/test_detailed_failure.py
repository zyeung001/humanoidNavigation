"""
Detailed failure analysis - log everything to understand WHY agent falls
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.standing_curriculum import make_standing_curriculum_env

print("="*60)
print("DETAILED FAILURE ANALYSIS")
print("="*60)
print()

config = {
    'obs_history': 4,
    'obs_include_com': True,
    'obs_feature_norm': True,
    'action_smoothing': True,
    'action_smoothing_tau': 0.2,
    'curriculum_start_stage': 3,
    'curriculum_max_stage': 3,
}

# Create environment with VecNormalize
env = DummyVecEnv([lambda: make_standing_curriculum_env(render_mode=None, config=config)])
env = VecNormalize.load("models/saved_models/vecnorm_standing.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/saved_models/best_standing_model.zip", env=env)

print("Running detailed episode analysis...\n")

obs = env.reset()
steps = 0
max_steps = 200

# Track metrics
heights = []
actions_mag = []
velocities = []
quaternions = []

while steps < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    steps += 1
    
    # Extract detailed info
    height = info[0].get('height', 0)
    x_vel = info[0].get('x_velocity', 0)
    y_vel = info[0].get('y_velocity', 0)
    z_vel = info[0].get('z_velocity', 0)
    quat_w = info[0].get('quaternion_w', 1.0)
    
    heights.append(height)
    actions_mag.append(np.abs(action).mean())
    velocities.append(np.sqrt(x_vel**2 + y_vel**2 + z_vel**2))
    quaternions.append(quat_w)
    
    if steps % 10 == 0:
        print(f"Step {steps:3d}: h={height:.3f}, vel={velocities[-1]:.3f}, "
              f"quat_w={quat_w:.3f}, action_mag={actions_mag[-1]:.3f}")
    
    if done[0]:
        print(f"\n❌ TERMINATED at step {steps}")
        print(f"Final height: {height:.3f}m")
        print(f"Final orientation (quat_w): {quat_w:.3f}")
        print(f"Final velocity: {velocities[-1]:.3f}")
        break

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Analyze trends
print(f"\nHeight progression:")
print(f"  Start: {heights[0]:.3f}m")
print(f"  Peak: {max(heights):.3f}m at step {heights.index(max(heights))}")
print(f"  End: {heights[-1]:.3f}m")
print(f"  Trend: {'Declining' if heights[-1] < heights[0] - 0.1 else 'Stable' if abs(heights[-1] - heights[0]) < 0.1 else 'Increasing'}")

print(f"\nVelocity progression:")
print(f"  Start: {velocities[0]:.3f}")
print(f"  Max: {max(velocities):.3f} at step {velocities.index(max(velocities))}")
print(f"  End: {velocities[-1]:.3f}")
print(f"  Trend: {'Accelerating' if velocities[-1] > velocities[0] + 0.5 else 'Stable'}")

print(f"\nOrientation (upright = 1.0):")
print(f"  Start: {quaternions[0]:.3f}")
print(f"  Min: {min(quaternions):.3f} at step {quaternions.index(min(quaternions))}")
print(f"  End: {quaternions[-1]:.3f}")
print(f"  Trend: {'Tipping over' if quaternions[-1] < 0.8 else 'Stable'}")

print(f"\nAction magnitude:")
print(f"  Mean: {np.mean(actions_mag):.3f}")
print(f"  Max: {max(actions_mag):.3f}")
print(f"  Trend: {'Increasing' if actions_mag[-1] > actions_mag[0] + 0.1 else 'Stable'}")

# Determine failure mode
print(f"\n" + "="*60)
print("FAILURE MODE DIAGNOSIS")
print("="*60)

if heights[-1] < 0.8:
    print("❌ FAILURE MODE: Height collapse")
    print("   Agent lost height and fell below termination threshold")
elif quaternions[-1] < 0.7:
    print("❌ FAILURE MODE: Orientation failure (tipped over)")
    print("   Agent's upright orientation was lost")
elif max(velocities) > 3.0:
    print("❌ FAILURE MODE: Instability (excessive velocity)")
    print("   Agent developed runaway velocity")
else:
    print("❓ FAILURE MODE: Unknown")
    print("   Agent terminated but metrics look reasonable")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if heights[-1] < heights[0] - 0.2:
    print("• Agent is losing height over time")
    print("  → Problem: Gravity compensation insufficient")
    print("  → Possible cause: Policy learned to minimize control cost too much")
elif quaternions[-1] < quaternions[0] - 0.1:
    print("• Agent is tipping over")
    print("  → Problem: Balance control failing")
    print("  → Possible cause: Policy doesn't respond to orientation changes")
elif max(velocities) > 2.0:
    print("• Agent develops drift/velocity")
    print("  → Problem: Position control failing")
    print("  → Possible cause: Reward doesn't penalize movement enough")
else:
    print("• Metrics look reasonable but agent still fails")
    print("  → Problem: Termination conditions too strict")
    print("  → OR: Multiple small issues compounding")

env.close()

