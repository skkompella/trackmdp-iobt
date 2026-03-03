#!/usr/bin/env python3
"""
IoBT-MAX Advanced Training Script

This script trains the PPO agent on the K-10 fully connected graph environment.
It features:
- Research-grade parameters for guaranteed convergence
- Periodic true-accuracy evaluation (bypassing negative reward bias)
- Automatic checkpoint saving
"""

import os
import sys
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import pdb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the IoBT classes we built in custom_environment.py
# (Ensure these match the names in your actual custom_environment.py file)
from src.core.iobt_environment import learning_iobt_sarsa
from src.core.gym_wrapper import iobt_gym_wrapper




def run_advanced_iobt_training():
    print("="*70)
    print("TRACK-MDP : IOBT-MAX ADVANCED TRAINING")
    print("="*70)

    # 1. Initialize the Environment with Research-Grade Parameters
    # 10% chance to exit, limits action space to 55 choices for fast convergence
    iobt_transitions = [0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90]
    
    qobj = learning_iobt_sarsa(
        run_number=2026,
        num_nodes=10,
        state_trans_cum_prob=iobt_transitions,
        max_sensors=10,       
        max_sensors_null=10, 
        time_limit=4,
        time_limit_max=4
    )
    
    env_config = {
        "qobj": qobj,
        "time_limit_max": qobj.time_limit_max
    }

    # 2. Setup Ray and Register Environment
    ray.init(ignore_reinit_error=True)
    register_env("iobt_env_graph", lambda config: iobt_gym_wrapper(config))

    # 3. Configure PPO Algorithm
    print("\nInitializing PPO Algorithm...")
    config = PPOConfig()
    config = config.environment("iobt_env_graph", env_config=env_config)
    config = config.training(
        lr=0.0005,           # Slightly lower learning rate for stable graph mapping
        grad_clip=30.0,
        train_batch_size=4000
    )
    config = config.resources(num_gpus=0)
    config = config.env_runners(num_env_runners=0) 
    # config = config.rollouts(num_rollout_workers=0, horizon=30) # Prevent nan errors
    config = config.api_stack(
        enable_rl_module_and_learner=False, 
        enable_env_runner_and_connector_v2=False
    )
    algo = config.build()

    # 4. Training Parameters
    TOTAL_ITERATIONS = 100
    EVAL_INTERVAL = 10       # Run evaluation every 10 iterations
    CHECKPOINT_DIR = os.path.join(project_root, f"agent_run{qobj.run_number}_iobt")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"\nTraining for {TOTAL_ITERATIONS} iterations.")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    print("-" * 70)

    # 5. The Main Training Loop
    try:
        for i in range(1, TOTAL_ITERATIONS + 1):
            result = algo.train()
            
            # Extract reward
            if 'episode_reward_mean' in result:
                reward_mean = result['episode_reward_mean']
                len_mean = result['episode_len_mean']
            else:
                reward_mean = result.get('env_runners', {}).get('episode_reward_mean', 0.0)
                len_mean = result.get('env_runners', {}).get('episode_len_mean', 0.0)
                
            print(f"Iteration {i:3d}: reward_mean = {reward_mean:8.4f}, episode_len_mean = {len_mean:6.2f}")

            # 6. Periodic True-Accuracy Evaluation & Checkpointing
            if i % EVAL_INTERVAL == 0:
                print(f"\n--- Running Evaluation (Iteration {i}) ---")
                accuracy, avg_sensors = evaluate_iobt_policy(algo, env_config, num_episodes=20)
                print(f"  True Tracking Accuracy : {accuracy*100:.2f}%")
                print(f"  Avg Sensors per Step   : {avg_sensors:.2f}")
                print("-" * 40)
                
                # Save Checkpoint
                checkpoint_path = algo.save(checkpoint_dir=CHECKPOINT_DIR)
                # print(f"  [Checkpoint Saved: {os.path.basename(checkpoint_path)}]\n")
                print(f"  [Checkpoint Saved: {CHECKPOINT_DIR}]")

        print("\n✅ Advanced Training Completed Successfully!")

    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user. Latest checkpoint preserved.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()


def evaluate_iobt_policy(algo, env_config, num_episodes=20):
    """
    Evaluates tracking accuracy by comparing action choices directly to ground truth.
    Bypasses negative reward biases caused by high sensor energy costs.
    """
    eval_env = iobt_gym_wrapper(env_config)
    qobj = env_config["qobj"]
    
    success_rates = []
    sensors_per_step = []

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        objects_found = 0
        total_steps = 0
        total_sensors_used = 0

        while not done and total_steps < 50:
            # Get action
            action = algo.compute_single_action(obs, explore=False)
            action_idx = int(action)
            
            # Decode action to find active sensors
            num_sensors_active = 0
            active_sensor_indices = []
            
            q_state = qobj.current_state
            comb_dict = qobj.grid_env.combination_dict_null if q_state == qobj.missing_state else qobj.grid_env.combination_dict[qobj.time_delay]
            
            for num_sensors in sorted(comb_dict.keys()):
                combs = comb_dict[num_sensors]
                if action_idx < len(combs):
                    num_sensors_active = num_sensors
                    active_sensor_indices = combs[action_idx]
                    break
                action_idx -= len(combs)

            # Ground truth check
            current_target_pos = eval_env.unwrapped.qobj.grid_env.object_pos
            if current_target_pos < qobj.num_nodes and current_target_pos in active_sensor_indices:
                objects_found += 1
                
            total_sensors_used += num_sensors_active
            total_steps += 1

            # Step environment
            obs, reward, done, truncated, info = eval_env.step(action)

        if total_steps > 0:
            success_rates.append(objects_found / total_steps)
            sensors_per_step.append(total_sensors_used / total_steps)

    accuracy = np.mean(success_rates) if success_rates else 0.0
    sensors = np.mean(sensors_per_step) if sensors_per_step else 0.0
    
    return accuracy, sensors


if __name__ == "__main__":
    run_advanced_iobt_training()