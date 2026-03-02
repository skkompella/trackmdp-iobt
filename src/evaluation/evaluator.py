"""
Policy Evaluation Module

This module provides functions for evaluating trained policies on the tracking task.
"""

import numpy as np


def evaluate_policy(algo, environment, num_episodes=1000, verbose=True):
    """
    Evaluate a trained policy on the tracking environment.
    
    This function runs the policy for multiple episodes and computes the average
    tracking success rate (proportion of steps where object is successfully detected)
    and average sensors used per step.
    
    Args:
        algo: Ray RLlib algorithm/trainer instance with compute_single_action method
        environment: grid_env instance from our environment module
        num_episodes (int): Number of evaluation episodes (default: 1000)
        verbose (bool): Print progress updates (default: True)
        
    Returns:
        tuple: (average_success_rate, average_sensors_used) - Average tracking success 
               rate and average number of sensors used per step across all episodes
    """
    success_rates = []
    sensors_per_step = []
    
    if verbose:
        print(f"Evaluating policy on {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        environment.reset_object_state()
        current_state = environment.missing_state
        done = False
        
        # Episode statistics
        objects_found = 0
        total_steps = 0
        total_sensors_used = 0
        time_delay = 0
        max_steps = 1000  # Prevent infinite episodes
        
        while not done and total_steps < max_steps:
            num_sensors = (2 * time_delay + 3) ** 2
            
            # Compute action based on current state
            if current_state != environment.missing_state:
                # Create the tuple observation that the policy expects
                state_pos = current_state // (environment.time_limit + 1)
                state_time = current_state % (environment.time_limit + 1)
                action_history = np.array([1] * 34)  # Default action history
                tuple_observation = (state_pos, state_time, action_history)
                
                # Get action from policy
                action_from_policy = algo.compute_single_action(tuple_observation, explore=False)

                # Extract relevant sensors for current time delay
                action_clip = action_from_policy[-num_sensors:]
                
                # Apply valid sensor mask
                action_sensors = np.multiply(
                    action_clip,
                    environment.valid_q_indices_dict[time_delay][current_state]
                )
                
                # Check if object is in sensor range
                obj_rel_pos, obj_in_grid = environment.realign_obj(
                    environment.object_pos, 
                    current_state, 
                    time_delay
                )
            else:
                # Missing state: activate all sensors
                obj_rel_pos, obj_in_grid = 0, 1
                action_sensors = [1] * num_sensors
            
            # Check if object was detected
            object_detected = 0
            if (obj_in_grid == 1) and (action_sensors[int(obj_rel_pos)] == 1):
                object_detected = 1
                objects_found += 1
            
            # Track sensor usage
            total_sensors_used += np.sum(action_sensors)
            total_steps += 1
            
            # Execute environment step
            reward, next_state, terminal_flag, time_delay = \
                environment.get_reward_next_state(
                    current_state, 
                    action_sensors, 
                    time_delay
                )
            
            # Check if episode is done
            if terminal_flag:
                done = True
            else:
                current_state = next_state
        
        # Calculate episode metrics
        if total_steps > 0:
            success_rate = objects_found / total_steps
            avg_sensors = total_sensors_used / total_steps
            success_rates.append(success_rate)
            sensors_per_step.append(avg_sensors)
    
    # Compute average metrics
    if len(success_rates) > 0:
        average_success_rate = sum(success_rates) / len(success_rates)
        average_sensors_used = sum(sensors_per_step) / len(sensors_per_step)
    else:
        average_success_rate = 0.0
        average_sensors_used = 0.0
    
    if verbose:
        print(f"  Average Tracking Success Rate: {average_success_rate:.4f} ({average_success_rate*100:.2f}%)")
        print(f"  Min Success Rate: {min(success_rates) if success_rates else 0:.4f}")
        print(f"  Max Success Rate: {max(success_rates) if success_rates else 0:.4f}")
        print(f"  Average Sensors Used per Step: {average_sensors_used:.2f}")
    
    return average_success_rate, average_sensors_used


def evaluate_policy_detailed(algo, environment, num_episodes=100):
    """
    Evaluate policy with detailed statistics.
    
    Args:
        algo: Ray RLlib algorithm/trainer instance
        environment: GridEnvironment instance
        num_episodes (int): Number of evaluation episodes
        
    Returns:
        dict: Dictionary containing detailed evaluation metrics
    """
    episode_stats = {
        'success_rates': [],
        'episode_lengths': [],
        'total_rewards': [],
        'sensor_usage': []
    }
    
    for episode in range(num_episodes):
        environment.reset_object_state()
        current_state = environment.missing_state
        done = False
        
        objects_found = 0
        total_steps = 0
        total_reward = 0
        total_sensors_used = 0
        time_delay = 0
        
        while not done:
            num_sensors = (2 * time_delay + 3) ** 2
            
            if current_state != environment.missing_state:
                current_action = algo.compute_single_action(current_state)
                action_clip = current_action[-num_sensors:]
                action_sensors = np.multiply(
                    action_clip,
                    environment.valid_q_indices_dict[time_delay][current_state]
                )
                obj_rel_pos, obj_in_grid = environment.realign_obj(
                    environment.object_pos, current_state, time_delay
                )
            else:
                obj_rel_pos, obj_in_grid = 0, 1
                current_action = [1] * ((2 * environment.time_limit + 3) ** 2)
                action_sensors = [1] * num_sensors
            
            if (obj_in_grid == 1) and (action_sensors[int(obj_rel_pos)] == 1):
                objects_found += 1
            
            total_sensors_used += np.sum(action_sensors)
            total_steps += 1
            
            reward, next_state, terminal_flag, time_delay = \
                environment.get_reward_next_state(
                    current_state, action_sensors, time_delay
                )
            
            total_reward += reward
            
            if terminal_flag:
                episode_stats['success_rates'].append(objects_found / total_steps)
                episode_stats['episode_lengths'].append(total_steps)
                episode_stats['total_rewards'].append(total_reward)
                episode_stats['sensor_usage'].append(total_sensors_used / total_steps)
                done = True
                continue
            
            current_state = next_state
    
    # Compute summary statistics
    summary = {
        'mean_success_rate': np.mean(episode_stats['success_rates']),
        'std_success_rate': np.std(episode_stats['success_rates']),
        'mean_episode_length': np.mean(episode_stats['episode_lengths']),
        'mean_total_reward': np.mean(episode_stats['total_rewards']),
        'mean_sensor_usage': np.mean(episode_stats['sensor_usage']),
        'detailed_stats': episode_stats
    }
    
    return summary


def print_evaluation_summary(summary):
    """
    Print formatted evaluation summary.
    
    Args:
        summary (dict): Evaluation summary from evaluate_policy_detailed
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Mean Success Rate:     {summary['mean_success_rate']:.4f} ± {summary['std_success_rate']:.4f}")
    print(f"Mean Episode Length:   {summary['mean_episode_length']:.2f}")
    print(f"Mean Total Reward:     {summary['mean_total_reward']:.2f}")
    print(f"Mean Sensor Usage:     {summary['mean_sensor_usage']:.2f} sensors/step")
    print("="*60 + "\n")