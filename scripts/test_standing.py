import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.humanoid_env import make_humanoid_env

def test_standing(model_path, vecnorm_path, n_steps=1000):
    """Test standing without video"""
    print("Creating environment...")
    model_path = "models/saved_models/final_standing_model.zip"
    env = make_humanoid_env(task_type="standing", render_mode=None)
    vec_env = DummyVecEnv([lambda: env])
    
    print(f"Loading VecNormalize from {vecnorm_path}...")
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=vec_env)
    
    print(f"Testing for {n_steps} steps...")
    obs = vec_env.reset()
    heights = []
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if len(info) > 0 and 'height' in info[0]:
            height = info[0]['height']
            heights.append(height)
            
            if step % 100 == 0:
                print(f"Step {step}: height={height:.3f}, reward={reward[0]:.2f}")
        
        if done[0]:
            print(f"FAILED at step {step}!")
            print(f"  Final height: {height:.3f}")
            print(f"  Info: {info[0]}")
            break
    else:
        print(f"SUCCESS! Stood for {n_steps} steps")
        print(f"  Mean height: {sum(heights)/len(heights):.3f}")
        print(f"  Height std: {(sum((h-sum(heights)/len(heights))**2 for h in heights)/len(heights))**0.5:.3f}")
    
    vec_env.close()

if __name__ == "__main__":
    test_standing(
        "models/saved_models/best_standing_model.zip",
        "models/saved_models/vecnorm_standing.pkl",
        n_steps=1000
    )