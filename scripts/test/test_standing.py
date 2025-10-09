# test_standing.py
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

# imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.standing_env import make_standing_env

def test_standing(model_path, vecnorm_path, n_steps=2000):
    """Test a trained standing model"""
    print(f"Project root: {project_root}")
    print(f"Looking for model at: {model_path}")
    print(f"Looking for vecnorm at: {vecnorm_path}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    if not os.path.exists(vecnorm_path):
        print(f"ERROR: VecNormalize not found at {vecnorm_path}")
        return
    
    print("Creating environment...")

    test_config = {
        'max_episode_steps': 5000,
        'domain_rand': False,  # Turn OFF for testing to see true performance
        'rand_mass_range': [0.9, 1.1],
        'rand_friction_range': [0.85, 1.15],
        'target_height': 1.3
    }
    env = make_standing_env(render_mode=None)
    vec_env = DummyVecEnv([lambda: env])
    
    print(f"Loading VecNormalize...")
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    print(f"Loading model...")
    model = PPO.load(model_path, env=vec_env, device='cpu')
    
    print(f"Testing for {n_steps} steps...")
    obs = vec_env.reset()
    heights, distances, rewards = [], [], []
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if len(info) > 0:
            h = info[0].get('height', 0)
            dist = info[0].get('distance_from_origin', 0)
            heights.append(h)
            distances.append(dist)
            rewards.append(reward[0])
            
            if step % 100 == 0:
                print(f"Step {step:4d}: height={h:.3f}, dist={dist:.3f}, reward={reward[0]:.2f}")
        
        if done[0]:
            print(f"\n❌ FAILED at step {step}!")
            print(f"  Final height: {h:.3f}, dist: {dist:.3f}")
            print(f"  Mean height: {np.mean(heights):.3f} ± {np.std(heights):.3f}")
            print(f"  Max distance: {np.max(distances):.3f}")
            break
    else:
        print(f"\n✅ SUCCESS! Stood for {n_steps} steps")
        print(f"  Mean height: {np.mean(heights):.3f} ± {np.std(heights):.3f}")
        print(f"  Height error: {abs(np.mean(heights) - 1.3):.3f}")
        print(f"  Mean dist: {np.mean(distances):.3f}, max: {np.max(distances):.3f}")
        print(f"  Mean reward/step: {np.mean(rewards):.2f}")
    
    vec_env.close()

if __name__ == "__main__":
    # Use absolute paths for Google Colab
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                       default="/content/humanoidNavigation/models/saved_models/final_standing_model.zip")
    parser.add_argument("--vecnorm", type=str,
                       default="/content/humanoidNavigation/models/saved_models/vecnorm_standing.pkl")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    
    test_standing(args.model, args.vecnorm, args.steps)