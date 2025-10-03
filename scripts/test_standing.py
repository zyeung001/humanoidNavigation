# test_standing.py
import sys
import os
import numpy as np  # For std calc

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.humanoid_env import make_humanoid_env

def test_standing(model_path, vecnorm_path, n_steps=2000):  # Longer for "forever"
    print("Creating environment...")
    env = make_humanoid_env(task_type="standing", render_mode=None)
    vec_env = DummyVecEnv([lambda: env])
    
    print(f"Loading VecNormalize from {vecnorm_path}...")
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=vec_env, device='cpu')  # CPU for efficiency
    
    print(f"Testing for {n_steps} steps...")
    obs = vec_env.reset()
    heights, distances, rewards = [], [], []
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if len(info) > 0:
            h = info[0].get('height', 0)
            dist = info[0].get('distance_from_origin', 0)  # Add if not in info
            heights.append(h)
            distances.append(dist)
            rewards.append(reward[0])
            
            if step % 100 == 0:
                print(f"Step {step}: height={h:.3f}, dist={dist:.3f}, reward={reward[0]:.2f}")
        
        if done[0]:
            print(f"FAILED at step {step}!")
            print(f"  Final height: {h:.3f}, dist: {dist:.3f}")
            print(f"  Mean height: {np.mean(heights):.3f}, std: {np.std(heights):.3f}")
            print(f"  Max dist: {np.max(distances):.3f}")
            break
    else:
        print(f"SUCCESS! Stood for {n_steps} steps")
        print(f"  Mean height: {np.mean(heights):.3f}, std: {np.std(heights):.3f}")
        print(f"  Mean dist: {np.mean(distances):.3f}, max: {np.max(distances):.3f}")
        print(f"  Mean reward/step: {np.mean(rewards):.2f}")
    
    vec_env.close()

if __name__ == "__main__":
    test_standing(
        "models/saved_models/final_standing_model.zip",  # Use final
        "models/saved_models/vecnorm_standing.pkl",
        n_steps=2000
    )