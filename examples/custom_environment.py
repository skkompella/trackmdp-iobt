#!/usr/bin/env python3
"""
Example 3: Custom Environment Configuration

This example shows how to create and test custom environment configurations
for different tracking scenarios.

Usage:
    python examples/custom_environment.py
"""

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.environment import learning_grid_sarsa_0
from src.core.gym_wrapper import grid_environment
from src.evaluation.evaluator import evaluate_policy
import ray
from ray.rllib.algorithms.ppo import PPOConfig


def small_grid_example():
    """
    Example of a small 5x5 grid for quick testing.
    """
    print("="*60)
    print("SMALL GRID ENVIRONMENT (5x5)")
    print("="*60)
    
    # Create a 5x5 environment
    qobj = learning_grid_sarsa_0(
        run_number=1001,
        N=5,                            # Small 5x5 grid
        num_trans=3,                    # Fewer transitions
        state_trans_cum_prob=[0.4, 0.7, 1.0],  # Simple probabilities
        max_sensors=4,                  # Moderate sensor count
        max_sensors_null=4,
        time_limit=1,
        time_limit_max=1
    )
    
    print(f"Environment Configuration:")
    print(f"  Grid Size: {qobj.N}x{qobj.N}")
    print(f"  Total States: {qobj.N * qobj.N}")
    print(f"  Missing State: {qobj.missing_state}")
    print(f"  Transitions: {qobj.num_trans}")
    print(f"  Max Sensors: {qobj.grid_env.max_sensors}")
    
    # Test object movement
    print(f"\nTesting Object Movement:")
    print("-" * 25)
    
    qobj.grid_env.reset_object_state()
    positions = [qobj.grid_env.object_pos]
    
    for i in range(10):
        if qobj.grid_env.object_pos == qobj.N * qobj.N:
            print(f"  Object reached terminal state at step {i}")
            break
        qobj.grid_env.object_move()
        positions.append(qobj.grid_env.object_pos)
    
    print(f"  Movement sequence: {positions}")
    
    # Visualize final grid state
    if qobj.grid_env.object_pos < qobj.N * qobj.N:
        print(f"\nGrid visualization (object at position {qobj.grid_env.object_pos}):")
        grid = np.zeros((qobj.N, qobj.N))
        row, col = qobj.grid_env.val_to_grid(qobj.grid_env.object_pos)
        grid[row, col] = 1
        print(grid)
    
    return qobj


def large_grid_example():
    """
    Example of a larger 15x15 grid for complex scenarios.
    """
    print("\n" + "="*60)
    print("LARGE GRID ENVIRONMENT (15x15)")
    print("="*60)
    
    # Create a 15x15 environment
    qobj = learning_grid_sarsa_0(
        run_number=1002,
        N=15,                           # Large 15x15 grid
        num_trans=6,                    # More transition types
        state_trans_cum_prob=[0.1, 0.25, 0.4, 0.6, 0.8, 1.0],  # Complex probabilities
        max_sensors=8,                  # More sensors available
        max_sensors_null=8,
        time_limit=2,                   # Longer time limit
        time_limit_max=2
    )
    
    print(f"Environment Configuration:")
    print(f"  Grid Size: {qobj.N}x{qobj.N}")
    print(f"  Total States: {qobj.N * qobj.N}")
    print(f"  Missing State: {qobj.missing_state}")
    print(f"  Transitions: {qobj.num_trans}")
    print(f"  Max Sensors: {qobj.grid_env.max_sensors}")
    print(f"  Time Limit: {qobj.time_limit}")
    
    # Show transition matrix sample
    print(f"\nSample Transition Matrices:")
    print("-" * 30)
    center_state = (qobj.N // 2) * qobj.N + (qobj.N // 2)
    corner_state = 0
    
    print(f"  Center state ({center_state}): {qobj.grid_env.obj_trans_matrix[center_state]}")
    print(f"  Corner state ({corner_state}): {qobj.grid_env.obj_trans_matrix[corner_state]}")
    
    return qobj


def high_mobility_example():
    """
    Example of high mobility environment with frequent movement.
    """
    print("\n" + "="*60)
    print("HIGH MOBILITY ENVIRONMENT")
    print("="*60)
    
    # Create environment with high movement probability
    qobj = learning_grid_sarsa_0(
        run_number=1003,
        N=8,                            # Medium grid
        num_trans=4,
        state_trans_cum_prob=[0.05, 0.15, 0.3, 0.9],  # High movement probability
        max_sensors=6,
        max_sensors_null=6,
        time_limit=1,
        time_limit_max=1
    )
    
    print(f"Environment Configuration:")
    print(f"  Grid Size: {qobj.N}x{qobj.N}")
    print(f"  Movement Probabilities: {qobj.grid_env.prob_list_cum}")
    print(f"  Terminal Probability: {1.0 - qobj.grid_env.prob_list_cum[-1]:.3f}")
    
    # Test mobility by running multiple episodes
    print(f"\nMobility Test (5 episodes):")
    print("-" * 30)
    
    episode_lengths = []
    
    for episode in range(5):
        qobj.grid_env.reset_object_state()
        steps = 0
        
        while qobj.grid_env.object_pos < qobj.N * qobj.N and steps < 50:
            qobj.grid_env.object_move()
            steps += 1
        
        episode_lengths.append(steps)
        status = "terminal" if qobj.grid_env.object_pos == qobj.N * qobj.N else "timeout"
        print(f"  Episode {episode + 1}: {steps} steps ({status})")
    
    avg_length = np.mean(episode_lengths)
    print(f"  Average episode length: {avg_length:.2f} steps")
    
    return qobj


def custom_reward_example():
    """
    Example of customizing reward structure.
    """
    print("\n" + "="*60)
    print("CUSTOM REWARD STRUCTURE")
    print("="*60)
    
    # Create environment
    qobj = learning_grid_sarsa_0(
        run_number=1004,
        N=6,
        num_trans=3,
        state_trans_cum_prob=[0.4, 0.7, 1.0],
        max_sensors=4,
        max_sensors_null=4,
        time_limit=1,
        time_limit_max=1
    )
    
    # Show default rewards
    print(f"Default Reward Structure:")
    print(f"  Tracking Success: {qobj.grid_env.tracking_rew}")
    print(f"  Tracking Miss: {qobj.grid_env.tracking_miss_rew}")
    print(f"  Sensor Cost: {qobj.grid_env.sensor_rew}")
    print(f"  Missing State Success: {qobj.grid_env.tracking_rew_missing}")
    print(f"  Missing State Miss: {qobj.grid_env.tracking_miss_rew_missing}")
    
    # Customize rewards for different scenarios
    scenarios = {
        "Energy Conscious": {
            "sensor_rew": -0.3,      # Higher sensor penalty
            "tracking_rew": 1.0,
            "tracking_miss_rew": -0.1
        },
        "Accuracy Focused": {
            "sensor_rew": -0.05,     # Lower sensor penalty
            "tracking_rew": 2.0,     # Higher tracking reward
            "tracking_miss_rew": -0.5 # Higher miss penalty
        },
        "Balanced": {
            "sensor_rew": -0.16,     # Default sensor penalty
            "tracking_rew": 1.0,
            "tracking_miss_rew": 0.0
        }
    }
    
    print(f"\nCustom Reward Scenarios:")
    print("-" * 28)
    
    for scenario_name, rewards in scenarios.items():
        # Create a copy of environment with custom rewards
        test_qobj = learning_grid_sarsa_0(
            run_number=1004 + hash(scenario_name) % 1000,
            N=6, num_trans=3, state_trans_cum_prob=[0.4, 0.7, 1.0],
            max_sensors=4, max_sensors_null=4, time_limit=1, time_limit_max=1
        )
        
        # Apply custom rewards
        for reward_type, value in rewards.items():
            setattr(test_qobj.grid_env, reward_type, value)
        
        print(f"\n  {scenario_name}:")
        print(f"    Sensor Cost: {test_qobj.grid_env.sensor_rew}")
        print(f"    Success Reward: {test_qobj.grid_env.tracking_rew}")
        print(f"    Miss Penalty: {test_qobj.grid_env.tracking_miss_rew}")
        
        # Simulate reward calculation
        sensor_count = 3
        success_reward = test_qobj.grid_env.tracking_rew + sensor_count * test_qobj.grid_env.sensor_rew
        miss_reward = test_qobj.grid_env.tracking_miss_rew + sensor_count * test_qobj.grid_env.sensor_rew
        
        print(f"    Example Rewards (3 sensors):")
        print(f"      Success: {success_reward:.2f}")
        print(f"      Miss: {miss_reward:.2f}")
    
    return qobj


def train_custom_environment(qobj):
    """
    Example of training on a custom environment.
    """
    print("\n" + "="*60)
    print("TRAINING ON CUSTOM ENVIRONMENT")
    print("="*60)
    
    try:
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Create environment config
        env_config = {
            "qobj": qobj,
            "time_limit_schedule": [100],  # Short schedule for demo
            "time_limit_max": qobj.time_limit_max
        }
        
        # Create PPO configuration
        config = PPOConfig()
        config = config.environment(grid_environment, env_config=env_config)
        config = config.training(lr=0.001, grad_clip=30.0)
        config = config.resources(num_gpus=0)
        config = config.env_runners(num_env_runners=0)  # Single worker for demo
        config = config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        
        # Build algorithm
        algo = config.build()
        
        print(f"Training Configuration:")
        print(f"  Algorithm: PPO")
        print(f"  Learning Rate: 0.001")
        print(f"  Grid Size: {qobj.N}x{qobj.N}")
        print(f"  Action Space: MultiDiscrete")
        
        # Run short training
        print(f"\nRunning short training (10 iterations for demo)...")
        
        for i in range(10):
            result = algo.train()
            
            # Handle metric location change
            if 'episode_reward_mean' in result:
                reward_mean = result['episode_reward_mean']
                len_mean = result['episode_len_mean']
            else:
                reward_mean = result.get('env_runners', {}).get('episode_reward_mean', 0.0)
                len_mean = result.get('env_runners', {}).get('episode_len_mean', 0.0)
                
            print(f"  Iteration {i+1:2d}: reward_mean = {reward_mean:8.4f}, "
                  f"episode_len_mean = {len_mean:6.2f}")
        
        # Quick evaluation
        print(f"\nQuick Evaluation (10 episodes):")
        accuracy, sensors = evaluate_policy(
            algo, qobj.grid_env, num_episodes=10, verbose=False
        )
        
        print(f"  Tracking Accuracy: {accuracy:.4f}")
        print(f"  Avg Sensors/Step: {sensors:.2f}")
        
        print(f"\n✅ Custom environment training completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()

def iobt_max_environment():
    """
    Configuration for the IoBT-MAX testbed at R2C2.
    10 Sensor Nodes + 1 Terminal/Exit State.
    """
    print("\n" + "="*60)
    print("IoBT-MAX TESTBED ENVIRONMENT (R2C2 SITE)")
    print("="*60)

    N_size = 4 # 4*3 = 12 cells
    
    # The object can move from each cell to any other cell.
    iobt_transitions = [0.1, 0.3, 0.6, 0.9, 1.0] 

    qobj = learning_grid_sarsa_0(
        run_number=2026,
        N=N_size,                       
        num_trans=5,                    
        state_trans_cum_prob=iobt_transitions,
        max_sensors=11,                 
        max_sensors_null=11,
        time_limit=2,                   
        time_limit_max=5
    )

    qobj.grid_env.sensor_rew = -0.25 
    qobj.grid_env.tracking_rew = 1.5  
    qobj.grid_env.tracking_miss_rew = -0.5 

    print(f"IoBT-MAX Configuration:")
    print(f"  Nodes Deployed: 11 fixed structures ")
    print(f"  Total Grid Cells: {qobj.N * qobj.N}")
    print(f"  Mobility: Fully connected (Any-to-Any movement)")
    print(f"  Primary Sensors: Zed 2i RGBD, TI mmWave Radar")
    
    return qobj


def main():
    """
    Main function to demonstrate custom environment configurations.
    """
    print("TRACK-MDP CUSTOM ENVIRONMENT EXAMPLES")
    print("=====================================\n")
    
    print("This example demonstrates different environment configurations:")
    print("• Small grid (5x5) for quick testing")
    print("• Large grid (15x15) for complex scenarios")  
    print("• High mobility environment")
    print("• Custom reward structures")
    print("• IoBT-MAX Testbed environment")
    print("• Training on custom environments")
    print()
    
    # Run environment examples
    print("Creating different environment configurations...\n")
    
    # Small grid
    small_env = small_grid_example()
    
    # Large grid
    large_env = large_grid_example()
    
    # High mobility
    mobile_env = high_mobility_example()
    
    # Custom rewards
    custom_env = custom_reward_example()
    
    # IoBT-MAX Testbed
    iobt_env = iobt_max_environment()
    
    # Ask user if they want to run training example
    print("\n" + "-"*60)
    response = input("Do you want to run training on a custom environment? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\nSelect environment to train on:")
        environments = {
            '1': ('Small Grid (5x5)', small_env),
            '2': ('Large Grid (15x15)', large_env),
            '3': ('High Mobility', mobile_env),
            '4': ('Custom Rewards', custom_env),
            '5': ('IoBT-MAX Testbed', iobt_env)
        }
        
        for key, (name, _) in environments.items():
            print(f"  {key}. {name}")
        
        while True:
            choice = input("Select environment (1-5): ")
            if choice in environments:
                break
            print("Invalid choice. Please enter 1-5.")
        
        name, env = environments[choice]
        print(f"\nTraining on {name}...")
        
        success = train_custom_environment(env)
        
        if success:
            print(f"\n🎉 Training on {name} completed successfully!")
        else:
            print(f"\n❌ Training on {name} failed.")
    else:
        print("\nSkipping training example.")
    
    print(f"\n📚 Summary:")
    print(f"Created {5} different environment configurations")
    print(f"Each environment can be used for different research scenarios:")
    print(f"• Small grids for rapid prototyping and testing")
    print(f"• Large grids for complex tracking challenges")
    print(f"• High mobility for dynamic tracking scenarios")
    print(f"• Custom rewards for specific optimization objectives")
    print(f"• IoBT-MAX Testbed for real-world deployment emulation")
    
    print(f"\nNext steps:")
    print(f"• Modify the parameters in this script to create your own configurations")
    print(f"• Use these environments in your own training scripts")
    print(f"• Compare performance across different environment types")


if __name__ == "__main__":
    main()