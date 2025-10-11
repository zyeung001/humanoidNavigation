"""
Diagnostic script to understand what your trained model is actually doing
Run this BEFORE training more to see the problem
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from src.environments.standing_env import make_standing_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

def diagnose_model(model_path="models/saved_models/best_standing_model.zip",
                   vecnorm_path="models/saved_models/vecnorm_standing.pkl"):
    """
    Diagnose what your model is doing wrong
    """
    print("=== DIAGNOSTIC TEST ===\n")
    
    # Load model and environment
    config = {'target_height': 1.4, 'max_episode_steps': 2000}
    base_env = DummyVecEnv([lambda: make_standing_env(render_mode=None, config=config)])
    
    # Load VecNormalize if it exists
    try:
        env = VecNormalize.load(vecnorm_path, base_env)
        env.training = False
        env.norm_reward = False
        print("✓ Loaded VecNormalize stats")
    except:
        env = base_env
        print("⚠ No VecNormalize stats found")
    
    model = PPO.load(model_path, env=env)
    print(f"✓ Loaded model from {model_path}\n")
    
    # Test 1: Initial standing ability
    print("TEST 1: Can the robot stand with zero actions?")
    print("-" * 50)
    obs = env.reset()
    initial_height = env.envs[0].unwrapped.data.qpos[2]
    print(f"Initial height: {initial_height:.3f}m")
    
    heights_zero = []
    for i in range(1000):
        obs, reward, done, info = env.step(np.array([[0.0] * env.action_space.shape[1]]))
        h = env.envs[0].unwrapped.data.qpos[2]
        heights_zero.append(h)
        if i % 100 == 0:
            print(f"Step {i:4d}: height = {h:.3f}m")
        if done[0]:
            print(f"⚠ Episode ended at step {i}")
            break
    
    # Test 2: What does your trained model do?
    print("\nTEST 2: What does your trained model do?")
    print("-" * 50)
    obs = env.reset()
    
    heights = []
    actions_taken = []
    rewards_received = []
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        h = env.envs[0].unwrapped.data.qpos[2]
        heights.append(h)
        actions_taken.append(np.abs(action[0]).mean())
        rewards_received.append(reward[0])
        
        if step % 50 == 0:
            print(f"Step {step:3d}: height={h:.3f}m, action_mag={actions_taken[-1]:.3f}, reward={reward[0]:.1f}")
        
        if done[0]:
            print(f"⚠ Episode ended at step {step}")
            break
    
    # Test 3: Physics check - what's the natural standing height?
    print("\nTEST 3: Natural standing height")
    print("-" * 50)
    env_test = gym.make('Humanoid-v5')
    obs, _ = env_test.reset()
    natural_height = env_test.unwrapped.data.qpos[2]
    print(f"Natural initial height (no actions): {natural_height:.3f}m")
    print(f"Target height: 1.40m")
    print(f"Difference: {abs(natural_height - 1.40):.3f}m")
    env_test.close()
    
    # Analysis and plotting
    print("\nANALYSIS")
    print("=" * 50)
    print(f"Mean height with trained model: {np.mean(heights):.3f} ± {np.std(heights):.3f}m")
    print(f"Mean action magnitude: {np.mean(actions_taken):.3f}")
    print(f"Std action magnitude: {np.std(actions_taken):.3f}")
    print(f"Mean reward: {np.mean(rewards_received):.2f}")
    print(f"Episode length: {len(heights)} steps")
    
    # Check for problems
    print("\nDIAGNOSTIC FINDINGS:")
    if np.mean(heights) < 1.2:
        print("❌ PROBLEM: Robot is too short! Mean height {:.3f}m < 1.2m".format(np.mean(heights)))
        print("   Likely cause: Reward function not pushing height up enough")
    
    if np.std(heights) > 0.15:
        print("❌ PROBLEM: Very unstable! Height std {:.3f} > 0.15".format(np.std(heights)))
        print("   Likely cause: Z-velocity penalty too weak or conflicting rewards")
    
    if np.mean(actions_taken) > 0.5:
        print("⚠ WARNING: High action magnitudes ({:.3f}) - might be fighting".format(np.mean(actions_taken)))
    
    if len(heights) < 200:
        print("❌ PROBLEM: Episode too short! Only {} steps".format(len(heights)))
        print("   Likely cause: Early termination or falls")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Height over time
    axes[0, 0].plot(heights, label='Trained model', linewidth=2)
    axes[0, 0].plot(heights_zero[:len(heights)], label='Zero actions', alpha=0.6)
    axes[0, 0].axhline(y=1.40, color='r', linestyle='--', label='Target (1.40m)')
    axes[0, 0].axhline(y=1.30, color='g', linestyle='--', label='Acceptable (1.30m)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].set_title('Height Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Action magnitudes
    axes[0, 1].plot(actions_taken)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Mean |action|')
    axes[0, 1].set_title('Action Magnitudes Over Time')
    axes[0, 1].grid(True)
    
    # Plot 3: Reward over time
    axes[1, 0].plot(rewards_received)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Rewards Over Time')
    axes[1, 0].grid(True)
    
    # Plot 4: Height distribution
    axes[1, 1].hist(heights, bins=30, alpha=0.7)
    axes[1, 1].axvline(x=1.40, color='r', linestyle='--', label='Target')
    axes[1, 1].set_xlabel('Height (m)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Height Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('diagnostic_results.png', dpi=150)
    print("\n✓ Diagnostic plots saved to diagnostic_results.png")
    
    env.close()
    
    return {
        'mean_height': np.mean(heights),
        'std_height': np.std(heights),
        'mean_action': np.mean(actions_taken),
        'episode_length': len(heights),
        'heights': heights
    }

if __name__ == "__main__":
    results = diagnose_model()