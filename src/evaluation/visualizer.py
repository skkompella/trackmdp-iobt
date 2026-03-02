"""
Interactive Visualization for Track-MDP

This module provides visualization functionality for evaluating and rendering
trained policies in real-time using pygame.
"""

import os
import numpy as np
import torch

try:
    from src.visualization.renderer import TrackingRenderer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def find_latest_checkpoint(checkpoint_root):
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_root (str): Root directory containing checkpoints
        
    Returns:
        str: Path to latest checkpoint directory, or None if none found
    """
    if not os.path.exists(checkpoint_root):
        print(f"Checkpoint directory does not exist: {checkpoint_root}")
        return None
    
    # List all checkpoint directories
    checkpoints = []
    for item in os.listdir(checkpoint_root):
        item_path = os.path.join(checkpoint_root, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint_'):
            try:
                # Extract checkpoint number
                checkpoint_num = int(item.split('_')[1])
                checkpoints.append((checkpoint_num, item_path))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        print(f"No checkpoints found in: {checkpoint_root}")
        return None
    
    # Sort by checkpoint number and return the latest
    checkpoints.sort(key=lambda x: x[0])
    latest = checkpoints[-1][1]
    print(f"Found {len(checkpoints)} checkpoints. Latest: {latest}")
    return latest


def visualize_policy_interactive(algo, environment, renderer=None, num_episodes=10, 
                                fps=5, wait_for_input=False, verbose=True):
    """
    Visualize a trained policy with interactive pygame rendering.
    
    Args:
        algo: Ray RLlib algorithm/trainer instance
        environment: grid_env instance from our environment module
        renderer: TrackingRenderer instance (optional, will create if None)
        num_episodes (int): Number of episodes to visualize (default: 10)
        fps (int): Frames per second for rendering (default: 5)
        wait_for_input (bool): Wait for key press after each step (default: False)
        verbose (bool): Print detailed statistics (default: True)
        
    Returns:
        dict: Summary statistics from visualization
    """
    if not PYGAME_AVAILABLE:
        raise ImportError("Pygame not available. Install pygame to use visualization.")
    
    grid_size = environment.N
    
    # Create renderer if not provided
    if renderer is None:
        renderer = TrackingRenderer(
            grid_size=grid_size,
            sq_pixels=40,
            fps=fps,
            title='Track-MDP Interactive Visualization'
        )
        should_close_renderer = True
    else:
        should_close_renderer = False
    
    if verbose:
        print(f"\n{'='*60}")
        print("TRACK-MDP INTERACTIVE VISUALIZATION")
        print(f"{'='*60}")
        print(f"Grid Size: {grid_size}x{grid_size}")
        print(f"Episodes: {num_episodes}")
        print(f"FPS: {fps}")
        if wait_for_input:
            print("Mode: Step-by-step (press any key to advance)")
        else:
            print("Mode: Real-time")
        print("Press ESC to exit")
        print(f"{'='*60}\n")
    
    total_success_rates = []
    total_sensors_used = []
    total_episodes_completed = 0
    
    try:
        for episode in range(num_episodes):
            # Reset environment
            environment.reset_object_state()
            current_state = environment.missing_state
            done = False
            
            # Episode statistics
            objects_found = 0
            total_steps = 0
            episode_sensors = 0
            time_delay = 0
            
            if verbose:
                print(f"\n{'='*40}")
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"Object starting at position: {environment.object_pos}")
                print(f"{'='*40}")
            
            step_count = 0
            max_steps_per_episode = 500  # Prevent infinite episodes
            
            while not done and step_count < max_steps_per_episode:
                step_count += 1
                num_sensors = (2 * time_delay + 3) ** 2
                
                # Get action from policy
                if current_state != environment.missing_state:
                    # Create the tuple observation that the policy expects
                    state_pos = current_state // (environment.time_limit + 1)
                    state_time = current_state % (environment.time_limit + 1)
                    action_history = np.array([1] * 34)  # Default action history
                    tuple_observation = (state_pos, state_time, action_history)
                    
                    current_action = algo.compute_single_action(tuple_observation,explore=False)
                    action_clip = current_action[-num_sensors:]
                    action_sensors = np.multiply(
                        action_clip,
                        environment.valid_q_indices_dict[time_delay][current_state]
                    )
                    obj_rel_pos, obj_in_grid = environment.realign_obj(
                        environment.object_pos,
                        current_state,
                        time_delay
                    )
                else:
                    # Missing state: activate all sensors
                    obj_rel_pos, obj_in_grid = 0, 1
                    current_action = [1] * ((2 * environment.time_limit + 3) ** 2)
                    action_sensors = [1] * num_sensors
                
                # Check if object was detected
                object_detected = (obj_in_grid == 1) and (action_sensors[int(obj_rel_pos)] == 1)
                if object_detected:
                    objects_found += 1
                
                # Count active sensors
                sensors_active = int(np.sum(action_sensors))
                episode_sensors += sensors_active
                total_steps += 1
                
                # Print step-level information (every 10 steps or when object detected)
                if verbose and (step_count % 10 == 0 or object_detected):
                    current_accuracy = objects_found / total_steps if total_steps > 0 else 0
                    current_avg_sensors = episode_sensors / total_steps if total_steps > 0 else 0
                    detection_status = "✓ DETECTED" if object_detected else "✗ missed"
                    print(f"  Step {step_count}: {detection_status} | Sensors: {sensors_active} | Accuracy: {current_accuracy:.3f}")
                
                # Create sensor visualization array
                if current_state != environment.missing_state:
                    # Create the action mapped to full grid
                    state_grid_pos = current_state // (environment.time_limit + 1)
                    state_x, state_y = environment.val_to_grid(state_grid_pos)
                    
                    # Get valid sensor positions for this state
                    window_size = 2 * time_delay + 3
                    center = window_size // 2
                    
                    # Create full grid sensor array
                    temp = np.zeros((grid_size, grid_size))
                    
                    # Map action_sensors to global grid positions
                    for sensor_idx in range(len(action_sensors)):
                        if action_sensors[sensor_idx] > 0:
                            # Convert sensor index to local offset
                            local_y = sensor_idx // window_size
                            local_x = sensor_idx % window_size
                            
                            # Convert to global position
                            global_y = state_y + (local_y - center)
                            global_x = state_x + (local_x - center)
                            
                            # Set if within bounds
                            if 0 <= global_y < grid_size and 0 <= global_x < grid_size:
                                temp[global_y, global_x] = 1
                    
                    anim_action = temp
                else:
                    # Missing state: all sensors on
                    anim_action = np.ones((grid_size, grid_size))
                
                # Update renderer
                renderer.update_sensors(anim_action, environment.object_pos)
                
                # Render frame
                if not renderer.render(wait_for_input=wait_for_input):
                    if verbose:
                        print("\nVisualization stopped by user.")
                    return _compile_visualization_summary(total_success_rates, total_sensors_used, interrupted=True)
                
                # Execute environment step
                reward, next_state, terminal_flag, time_delay = \
                    environment.get_reward_next_state(
                        current_state,
                        action_sensors,
                        time_delay
                    )
                
                # Check if episode is done
                if terminal_flag:
                    success_rate = objects_found / total_steps if total_steps > 0 else 0
                    avg_sensors = episode_sensors / total_steps if total_steps > 0 else 0
                    total_success_rates.append(success_rate)
                    total_sensors_used.append(avg_sensors)
                    total_episodes_completed += 1
                    
                    if verbose:
                        print(f"\n🏁 Episode {episode + 1} Complete:")
                        print(f"  📊 Steps: {total_steps}")
                        print(f"  🎯 Objects Found: {objects_found}")
                        print(f"  ✅ Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
                        print(f"  📡 Avg Sensors/Step: {avg_sensors:.2f}")
                        
                        # Running average statistics
                        if total_episodes_completed > 0:
                            running_accuracy = np.mean(total_success_rates)
                            running_sensors = np.mean(total_sensors_used)
                            print(f"\n📈 Running Averages ({total_episodes_completed} episodes):")
                            print(f"  🎯 Average Accuracy: {running_accuracy:.4f} ({running_accuracy*100:.2f}%)")
                            print(f"  📡 Average Sensors: {running_sensors:.2f}")
                    
                    done = True
                    continue
                
                current_state = next_state
            
            # Handle case where episode didn't terminate naturally
            if not done:
                success_rate = objects_found / total_steps if total_steps > 0 else 0
                avg_sensors = episode_sensors / total_steps if total_steps > 0 else 0
                total_success_rates.append(success_rate)
                total_sensors_used.append(avg_sensors)
                total_episodes_completed += 1
                
                if verbose:
                    print(f"\n⏰ Episode {episode + 1} Timed Out:")
                    print(f"  📊 Steps: {total_steps}")
                    print(f"  🎯 Objects Found: {objects_found}")
                    print(f"  ✅ Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
                    print(f"  📡 Avg Sensors/Step: {avg_sensors:.2f}")
    
    finally:
        if should_close_renderer:
            renderer.close()
    
    # Compile and return summary
    summary = _compile_visualization_summary(total_success_rates, total_sensors_used)
    if verbose:
        _print_visualization_summary(summary)
    
    return summary


def _compile_visualization_summary(success_rates, sensors_used, interrupted=False):
    """Compile visualization statistics into a summary dictionary."""
    if not success_rates:
        return {
            'episodes_completed': 0,
            'interrupted': interrupted,
            'mean_success_rate': 0.0,
            'std_success_rate': 0.0,
            'mean_sensors_used': 0.0,
            'std_sensors_used': 0.0
        }
    
    return {
        'episodes_completed': len(success_rates),
        'interrupted': interrupted,
        'mean_success_rate': np.mean(success_rates),
        'std_success_rate': np.std(success_rates),
        'best_success_rate': max(success_rates),
        'worst_success_rate': min(success_rates),
        'mean_sensors_used': np.mean(sensors_used),
        'std_sensors_used': np.std(sensors_used),
        'best_efficiency': min(sensors_used),
        'worst_efficiency': max(sensors_used),
        'success_rates': success_rates,
        'sensors_used': sensors_used
    }


def _print_visualization_summary(summary):
    """Print formatted visualization summary."""
    if summary['episodes_completed'] == 0:
        print("\n⚠ No episodes completed successfully.")
        return
    
    print("\n" + "="*60)
    print("🎯 VISUALIZATION SUMMARY")
    print("="*60)
    print(f"📊 Episodes Completed: {summary['episodes_completed']}")
    print(f"📈 Average Success Rate: {summary['mean_success_rate']:.4f} ({summary['mean_success_rate']*100:.2f}%)")
    print(f"📉 Success Rate Std Dev: ± {summary['std_success_rate']:.4f}")
    print(f"🎯 Best Episode Accuracy: {summary['best_success_rate']:.4f} ({summary['best_success_rate']*100:.2f}%)")
    print(f"🎯 Worst Episode Accuracy: {summary['worst_success_rate']:.4f} ({summary['worst_success_rate']*100:.2f}%)")
    print(f"📡 Average Sensors/Step: {summary['mean_sensors_used']:.2f}")
    print(f"📡 Sensors Std Dev: ± {summary['std_sensors_used']:.2f}")
    print(f"🔋 Most Efficient Episode: {summary['best_efficiency']:.2f} sensors/step")
    print(f"🔋 Least Efficient Episode: {summary['worst_efficiency']:.2f} sensors/step")
    
    # Performance categories
    success_rates = summary['success_rates']
    excellent_episodes = sum(1 for rate in success_rates if rate >= 0.8)
    good_episodes = sum(1 for rate in success_rates if 0.6 <= rate < 0.8)
    poor_episodes = sum(1 for rate in success_rates if rate < 0.6)
    total_episodes = len(success_rates)
    
    print(f"\n📋 Performance Breakdown:")
    print(f"  🌟 Excellent (≥80%): {excellent_episodes} episodes ({excellent_episodes/total_episodes*100:.1f}%)")
    print(f"  👍 Good (60-79%): {good_episodes} episodes ({good_episodes/total_episodes*100:.1f}%)")
    print(f"  👎 Poor (<60%): {poor_episodes} episodes ({poor_episodes/total_episodes*100:.1f}%)")
    
    if summary['interrupted']:
        print(f"\n⚠ Visualization was interrupted by user.")
    
    print("="*60 + "\n")


def evaluate_and_visualize(algo, environment, num_episodes=10, fps=5, 
                          evaluation_episodes=100, show_visualization=True):
    """
    Combined evaluation and visualization function.
    
    First runs a standard evaluation without rendering, then provides
    interactive visualization.
    
    Args:
        algo: Ray RLlib algorithm/trainer instance
        environment: grid_env instance
        num_episodes (int): Number of episodes for visualization
        fps (int): Frames per second for visualization
        evaluation_episodes (int): Number of episodes for evaluation
        show_visualization (bool): Whether to show interactive visualization
        
    Returns:
        tuple: (evaluation_summary, visualization_summary)
    """
    from .evaluator import evaluate_policy_detailed
    
    print("="*60)
    print("COMBINED EVALUATION AND VISUALIZATION")
    print("="*60)
    
    # First run standard evaluation
    print("Step 1: Running standard evaluation...")
    eval_summary = evaluate_policy_detailed(algo, environment, evaluation_episodes)
    
    print(f"\n📊 Evaluation Results ({evaluation_episodes} episodes):")
    print(f"  🎯 Mean Success Rate: {eval_summary['mean_success_rate']:.4f} ({eval_summary['mean_success_rate']*100:.2f}%)")
    print(f"  📡 Mean Sensor Usage: {eval_summary['mean_sensor_usage']:.2f} sensors/step")
    print(f"  🔄 Mean Episode Length: {eval_summary['mean_episode_length']:.1f} steps")
    print(f"  🏆 Mean Total Reward: {eval_summary['mean_total_reward']:.2f}")
    
    viz_summary = None
    if show_visualization and PYGAME_AVAILABLE:
        print(f"\nStep 2: Starting interactive visualization ({num_episodes} episodes)...")
        print("Press ESC at any time to stop visualization.\n")
        
        viz_summary = visualize_policy_interactive(
            algo=algo,
            environment=environment,
            num_episodes=num_episodes,
            fps=fps,
            verbose=True
        )
    elif show_visualization:
        print("\n⚠ Visualization skipped: pygame not available")
        print("Install pygame to enable visualization: pip install pygame")
    else:
        print("\nVisualization skipped (show_visualization=False)")
    
    return eval_summary, viz_summary