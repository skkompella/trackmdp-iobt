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
import pdb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.environment import learning_grid_sarsa_0
from src.core.gym_wrapper import grid_environment
from src.evaluation.evaluator import evaluate_policy
from src.core.iobt_environment import learning_iobt_sarsa
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from src.core.gym_wrapper import iobt_gym_wrapper


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


def iobt_max_environment_grid(): # preexisting grid version of the iobt max environment
    """
    Configuration for the IoBT-MAX testbed at R2C2.
    10 Sensor Nodes + 1 Terminal/Exit State.
    """
    print("\n" + "="*60)
    print("IoBT-MAX TESTBED ENVIRONMENT (R2C2 SITE)")
    print("="*60)

    
    qobj = learning_grid_sarsa_0(
        run_number=1004,
        N=4,                       
        num_trans=10,  # originally 5                  
        # state_trans_cum_prob=[0.05, 0.14, 0.23, 0.32, 0.41, 0.50, 0.59, 0.68, 0.77, 0.86, 0.95], # originally [0.1, 0.3, 0.6, 0.9, 1.0]
        state_trans_cum_prob=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90] , # originally [0.1, 0.3, 0.6, 0.9, 1.0]
        max_sensors=10,  # originally 11                 
        max_sensors_null=10, # originally 11
        time_limit=1,                   
        time_limit_max=1
    )

    print(f"IoBT-MAX Configuration:")
    print(f"  Nodes Deployed: 11 fixed structures ")
    print(f"  Total Grid Cells: {qobj.N * qobj.N}")
    print(f"  Mobility: Fully connected (Any-to-Any movement)")
    print(f"  Primary Sensors: Zed 2i RGBD, TI mmWave Radar")

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

def iobt_max_environment():
    """
    Configuration for the IoBT-MAX testbed at R2C2.
    10 Sensor Nodes + 1 Terminal/Exit State (Fully Connected K-10 Graph).
    """
    print("\n" + "="*60)
    print("IoBT-MAX TESTBED ENVIRONMENT (K-10 GRAPH)")
    print("="*60)

    # 10% chance of reaching terminal state at each step (0.9 to 1.0)
    iobt_transitions = [0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.90]
    # iobt_transitions = [0.18, 0.36, 0.54, 0.72, 0.90]

    qobj = learning_iobt_sarsa(
        run_number=2026,
        num_nodes=10,
        state_trans_cum_prob=iobt_transitions,
        max_sensors=6,       # RESTRICTED to 2 to prevent Action Space explosion
        max_sensors_null=6, 
        time_limit=4,
        time_limit_max=4
    )
    
    # Adding a dummy 'N' property just in case the underlying Gym Wrapper 
    # (grid_environment) explicitly looks for qobj.N during initialization.
    qobj.N = qobj.num_nodes

    print(f"IoBT-MAX Configuration:")
    print(f"  Nodes Deployed: {qobj.num_nodes} fixed structures (R2C2)")
    print(f"  Mobility: Fully connected graph (Any-to-Any movement)")
    print(f"  Max Sensors allowed per step: {qobj.max_sensors}")
    print(f"  Total Actions available: {qobj.total_actions}")

    # Quick mobility test to ensure termination works
    print(f"\nMobility Test (5 episodes):")
    print("-" * 30)
    episode_lengths = []
    
    for episode in range(5):
        qobj.grid_env.reset_object_state()
        steps = 0
        
        while qobj.grid_env.object_pos < qobj.num_nodes and steps < 50:
            qobj.grid_env.object_move()
            steps += 1
            
        episode_lengths.append(steps)
        status = "terminal" if qobj.grid_env.object_pos == qobj.num_nodes else "timeout"
        print(f"  Episode {episode + 1}: {steps} steps ({status})")
    
    avg_length = np.mean(episode_lengths)
    print(f"  Average episode length: {avg_length:.2f} steps")

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


def train_iobt_environment(qobj):
    """
    Example of training on the non-spatial IoBT environment.
    Updated to handle num_nodes instead of N, and forces episode horizons.
    """
    print("\n" + "="*60)
    print("TRAINING ON IOBT-MAX ENVIRONMENT")
    print("="*60)
    
    try:
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # 1. Register the environment explicitly
        register_env("iobt_env_graph", lambda config: iobt_gym_wrapper(config))
        
        # Create environment config
        env_config = {
            "qobj": qobj,
            "time_limit_schedule": [100],  
            "time_limit_max": qobj.time_limit_max
        }
        
        # Create PPO configuration
        config = PPOConfig()
        
        # 2. Pass the registered string name instead of the class
        config = config.environment("iobt_env_graph", env_config=env_config)
        config = config.training(lr=0.001, grad_clip=30.0)
        config = config.resources(num_gpus=0)
        config = config.env_runners(num_env_runners=0) 
        config = config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        
        # Build algorithm
        algo = config.build()
        
        print(f"Training Configuration:")
        print(f"  Algorithm: PPO")
        print(f"  Learning Rate: 0.001")
        print(f"  Graph Nodes: {qobj.num_nodes}")
        print(f"  Action Space: MultiDiscrete")
        
        # Run short training
        print(f"\nRunning short training (10 iterations for demo)...")
        # pdb.set_trace()
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
        
        print(f"\nQuick Evaluation (10 episodes):")
        eval_env = iobt_gym_wrapper(env_config)
        success_rates = []
        sensors_per_step = []

        for _ in range(10):
            obs, _ = eval_env.reset()
            done = False
            objects_found = 0
            total_steps = 0
            total_sensors_used = 0

            while not done and total_steps < 50:
                # 1. Get action from the trained policy
                action = algo.compute_single_action(obs, explore=False)
                
                # 2. Decode the action to find exactly WHICH sensors were turned on
                action_idx = int(action)
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

                # 3. Explicitly check if the target is found BEFORE taking the step
                # We reach into the base environment to get the ground truth position
                current_target_pos = eval_env.unwrapped.qobj.grid_env.object_pos
                
                # If target is not in terminal state AND its position is in our active sensors
                if current_target_pos < qobj.num_nodes and current_target_pos in active_sensor_indices:
                    objects_found += 1
                    
                total_sensors_used += num_sensors_active
                total_steps += 1

                # 4. Take the environment step to transition to the next state
                obs, reward, done, truncated, info = eval_env.step(action)

            # Record episode metrics
            if total_steps > 0:
                success_rates.append(objects_found / total_steps)
                sensors_per_step.append(total_sensors_used / total_steps)

        accuracy = np.mean(success_rates) if success_rates else 0.0
        sensors = np.mean(sensors_per_step) if sensors_per_step else 0.0

        print(f"  Tracking Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Avg Sensors/Step: {sensors:.2f}")
        
        print(f"\n✅ IoBT-MAX environment training completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


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
        
        if choice == '5':
            success = train_iobt_environment(env)
        else:
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