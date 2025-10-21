"""
Quick script to verify a trained model can actually stand
test_standing.py
"""

import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.environments.standing_env import make_standing_env
from stable_baselines3 import PPO

def verify_standing(model_path="models/saved_models/best_standing_model.zip"):
    """Test if model can stand for extended period"""
    
    print("Loading model...")
    model = PPO.load(model_path)
    
    config = {'target_height': 1.25, 'max_episode_steps': 10000}
    env = make_standing_env(render_mode=None, config=config)
    
    print("Testing standing duration...")
    obs, _ = env.reset()
    
    step_count = 0
    height_errors = []
    
    for step in range(10000):  # Test for 10k steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        height_error = abs(info['height'] - 1.25)
        height_errors.append(height_error)
        step_count += 1
        
        if step % 500 == 0:
            recent_errors = height_errors[-500:] if len(height_errors) >= 500 else height_errors
            print(f"Step {step}: height={info['height']:.3f}, "
                  f"avg_error={np.mean(recent_errors):.3f}, "
                  f"reward={reward:.1f}")
        
        if terminated:
            print(f"FELL at step {step}!")
            break
    
    env.close()
    
    print(f"\n{'='*60}")
    if step_count >= 5000:
        print("✅ SUCCESS: Model can stand for 5000+ steps!")
    elif step_count >= 1000:
        print("⚠️ PARTIAL: Model stands for 1000+ steps but needs more training")
    else:
        print("❌ FAILURE: Model falls too quickly, needs different approach")
    print(f"Total steps stood: {step_count}")
    print(f"Average height error: {np.mean(height_errors):.3f}")
    print(f"Height stability (std): {np.std(height_errors):.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                       default="models/saved_models/best_standing_model.zip")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        verify_standing(args.model)
    else:
        print(f"Model not found: {args.model}")
        print("Train a model first with: python scripts/train_standing.py")