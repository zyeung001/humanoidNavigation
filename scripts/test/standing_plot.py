import sys
import os

# Add project root to path
project_root = '/content/humanoidNavigation'  # Adjust if needed
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from src.environments.standing_env import make_standing_env

# Load model - need to handle VecNormalize properly
config = {'target_height': 1.3, 'max_episode_steps': 2000}

# Create base environment
base_env = DummyVecEnv([lambda: make_standing_env(render_mode=None, config=config)])

# Load VecNormalize stats if they exist
try:
    env = VecNormalize.load("models/saved_models/vecnorm_standing.pkl", base_env)
    env.training = False
    env.norm_reward = False
    print("✓ Loaded VecNormalize")
except:
    env = base_env
    print("⚠ No VecNormalize found")

# Load model
model = PPO.load("models/saved_models/best_standing_model.zip", env=env)

# Run evaluation
obs = env.reset()
heights = []
actions_taken = []

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Extract height from vectorized env
    heights.append(env.envs[0].unwrapped.data.qpos[2])
    actions_taken.append(np.abs(action).mean())
    
    if done[0]:  # VecEnv returns array
        print(f"Episode ended at step {step}")
        break

# Plot height over time
plt.figure(figsize=(10, 6))
plt.plot(heights, linewidth=2)
plt.axhline(y=1.30, color='r', linestyle='--', label='Target', linewidth=2)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.title('Height trajectory of trained model', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('height_trajectory.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to height_trajectory.png")

print(f"\nMean action magnitude: {np.mean(actions_taken):.3f}")
print(f"Std action magnitude: {np.std(actions_taken):.3f}")
print(f"Mean height: {np.mean(heights):.3f}m")
print(f"Std height: {np.std(heights):.3f}m")
print(f"Episode length: {len(heights)} steps")

env.close()