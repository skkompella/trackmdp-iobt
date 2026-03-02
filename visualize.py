"""
Visualization Script for Track-MDP

This script loads a trained model and visualizes its performance on the tracking task
using pygame rendering.
"""

import os
import sys
import argparse
import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.environment import learning_grid_sarsa_0
from src.core.gym_wrapper import grid_environment
from src.visualization.renderer import TrackingRenderer

# Add compatibility for old checkpoints that expect old import paths
import sys
import src.core.gym_wrapper
import src.core.environment
sys.modules['src.gym_wrapper'] = src.core.gym_wrapper
sys.modules['src.environment'] = src.core.environment


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


def visualize_policy(checkpoint_path, num_episodes=10, fps=5, 
                     wait_for_input=False, grid_size=None, time_limit=None):
    """
    Visualize a trained policy using pygame rendering.
    Uses the exact same approach as original Track-MDP.py but with our updated implementation.
    
    Args:
        checkpoint_path (str): Path to saved checkpoint
        num_episodes (int): Number of episodes to visualize (default: 10)
        fps (int): Frames per second for rendering (default: 5)
        wait_for_input (bool): Wait for key press after each step (default: False)
        grid_size (int, optional): Override grid size from config
        time_limit (int, optional): Override time limit from config
    """
    print("\n" + "="*60)
    print("TRACK-MDP VISUALIZATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print("="*60 + "\n")
    
    # Try to load environment parameters from saved file
    checkpoint_root = os.path.dirname(checkpoint_path)
    env_file = os.path.join(checkpoint_root, f'env_s1.pt')
    
    if os.path.exists(env_file):
        print(f"Loading environment configuration from: {env_file}")
        try:
            # Try loading with weights_only=False for our custom classes
            env_data = torch.load(env_file, weights_only=False)
            qobj = env_data['environment']
            print("✓ Environment configuration loaded from saved file")
            grid_size = qobj.N
            time_limit = qobj.time_limit
        except Exception as e:
            print(f"⚠ Warning: Could not load saved environment: {e}")
            print("⚠ Using default parameters instead...")
            # Fall back to defaults
            grid_size = grid_size or 5
            time_limit = time_limit or 1
            
            run_number = 194
            N, num_trans = grid_size, 3
            terminal_st_prob = 0.005
            state_prob_run = 0.15
            state_trans_cum_prob = [i*(1-terminal_st_prob-state_prob_run)/float(num_trans-1) for i in range(1, num_trans)]
            state_trans_cum_prob += [state_trans_cum_prob[-1] + state_prob_run] 
            max_sensors, max_sensors_null = 6, 6
            
            qobj = learning_grid_sarsa_0(run_number, N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, time_limit, time_limit)
    else:
        print("⚠ No saved environment found. Using parameters from config...")
        # Use provided parameters or defaults
        grid_size = grid_size or 5  # Default 5x5 grid
        time_limit = time_limit or 1  # Default time limit 1
        
        # Create default environment
        run_number = 194
        N, num_trans = grid_size, 3
        terminal_st_prob = 0.005
        state_prob_run = 0.15
        state_trans_cum_prob = [i*(1-terminal_st_prob-state_prob_run)/float(num_trans-1) for i in range(1, num_trans)]
        state_trans_cum_prob += [state_trans_cum_prob[-1] + state_prob_run] 
        max_sensors, max_sensors_null = 6, 6
        
        qobj = learning_grid_sarsa_0(run_number, N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, time_limit, time_limit)
    
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Time Limit: {time_limit}")
    print(f"Episodes: {num_episodes}")
    print(f"FPS: {fps}")
    print("="*60 + "\n")
    
    # Initialize Ray
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    
    try:
        # Load the trained algorithm
        print(f"Loading checkpoint from: {checkpoint_path}")
        algo = PPO.from_checkpoint(checkpoint_path)
        print("Checkpoint loaded successfully!\n")
        
        # Create renderer (matching original Track-MDP.py exactly)
        renderer = TrackingRenderer(
            grid_size=grid_size,
            sq_pixels=40,  # Original uses 40
            fps=fps,
            title=f'Track-MDP Visualization'
        )
        
        # Run visualization
        print("Starting visualization...")
        print("Press ESC to exit")
        if wait_for_input:
            print("Press any key to advance to next step\n")
        else:
            print()
        
        total_success_rates = []
        total_sensors_used = []
        total_episodes_completed = 0
        
        for episode in range(num_episodes):
            # Reset environment
            qobj.grid_env.reset_object_state()
            current_state = qobj.missing_state
            done = False
            
            # Episode statistics
            objects_found = 0
            total_steps = 0
            episode_sensors = 0
            time_delay = 0
            
            print(f"\n{'='*40}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Object starting at position: {qobj.grid_env.object_pos}")
            print(f"{'='*40}")
            
            step_count = 0
            max_steps_per_episode = 500  # Prevent infinite episodes
            
            while not done and step_count < max_steps_per_episode:
                step_count += 1
                num_sensors = (2 * time_delay + 3) ** 2
                
                # Get action from policy (matching our gym wrapper interface)
                if current_state != qobj.missing_state:
                    # Create the tuple observation that the policy expects
                    state_pos = current_state // (time_limit + 1)
                    state_time = current_state % (time_limit + 1)
                    action_history = np.array([1] * 34)  # Default action history
                    tuple_observation = (state_pos, state_time, action_history)

                    current_action = algo.compute_single_action(tuple_observation, explore=False)
                    action_clip = current_action[-num_sensors:]
                    action_sensors = np.multiply(
                        action_clip,
                        qobj.grid_env.valid_q_indices_dict[time_delay][current_state]
                    )
                    obj_rel_pos, obj_in_grid = qobj.grid_env.realign_obj(
                        qobj.grid_env.object_pos,
                        current_state,
                        time_delay
                    )
                else:
                    # Missing state: activate all sensors
                    obj_rel_pos, obj_in_grid = 0, 1
                    current_action = [1] * ((2 * time_limit + 3) ** 2)
                    action_sensors = [1] * num_sensors
                
                # Check if object was detected
                object_detected = (obj_in_grid == 1) and (action_sensors[int(obj_rel_pos)] == 1)
                if object_detected:
                    objects_found += 1
                
                # Count active sensors
                sensors_active = int(np.sum(action_sensors))
                episode_sensors += sensors_active
                total_steps += 1
                
                # Print step-level information every 10 steps or when object is detected
                if step_count % 10 == 0 or object_detected:
                    current_accuracy = objects_found / total_steps if total_steps > 0 else 0
                    current_avg_sensors = episode_sensors / total_steps if total_steps > 0 else 0
                    detection_status = "✓ DETECTED" if object_detected else "✗ missed"
                    print(f"  Step {step_count}: {detection_status} | Sensors: {sensors_active} | Accuracy: {current_accuracy:.3f} | Avg Sensors: {current_avg_sensors:.1f}")
                
                # Create sensor visualization array
                if current_state != qobj.missing_state:
                    # Create the action mapped to full grid
                    state_grid_pos = current_state // (time_limit + 1)
                    state_x, state_y = qobj.grid_env.val_to_grid(state_grid_pos)
                    
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
                renderer.update_sensors(anim_action, qobj.grid_env.object_pos)
                
                # Render frame
                if not renderer.render(wait_for_input=wait_for_input):
                    print("\nVisualization stopped by user.")
                    renderer.close()
                    return
                
                # Execute environment step
                reward, next_state, terminal_flag, time_delay = \
                    qobj.grid_env.get_reward_next_state(
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
                    
                    print(f"\n🏁 Episode {episode + 1} Complete:")
                    print(f"  📊 Steps: {total_steps}")
                    print(f"  🎯 Objects Found: {objects_found}")
                    print(f"  ✅ Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
                    print(f"  📡 Avg Sensors/Step: {avg_sensors:.2f}")
                    print(f"  🔄 Reason: Object reached terminal state")
                    
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
                
                print(f"\n⏰ Episode {episode + 1} Timed Out:")
                print(f"  📊 Steps: {total_steps}")
                print(f"  🎯 Objects Found: {objects_found}")
                print(f"  ✅ Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
                print(f"  📡 Avg Sensors/Step: {avg_sensors:.2f}")
                print(f"  🔄 Reason: Maximum steps reached ({max_steps_per_episode})")
        
        # Print comprehensive summary statistics
        if total_success_rates:
            print("\n" + "="*60)
            print("🎯 FINAL VISUALIZATION SUMMARY")
            print("="*60)
            print(f"📊 Episodes Completed: {len(total_success_rates)}")
            print(f"📈 Average Success Rate: {np.mean(total_success_rates):.4f} ({np.mean(total_success_rates)*100:.2f}%)")
            print(f"📉 Success Rate Std Dev: ± {np.std(total_success_rates):.4f}")
            print(f"🎯 Best Episode Accuracy: {max(total_success_rates):.4f} ({max(total_success_rates)*100:.2f}%)")
            print(f"🎯 Worst Episode Accuracy: {min(total_success_rates):.4f} ({min(total_success_rates)*100:.2f}%)")
            print(f"📡 Average Sensors/Step: {np.mean(total_sensors_used):.2f}")
            print(f"📡 Sensors Std Dev: ± {np.std(total_sensors_used):.2f}")
            print(f"🔋 Most Efficient Episode: {min(total_sensors_used):.2f} sensors/step")
            print(f"🔋 Least Efficient Episode: {max(total_sensors_used):.2f} sensors/step")
            
            # Performance categories
            excellent_episodes = sum(1 for rate in total_success_rates if rate >= 0.8)
            good_episodes = sum(1 for rate in total_success_rates if 0.6 <= rate < 0.8)
            poor_episodes = sum(1 for rate in total_success_rates if rate < 0.6)
            
            print(f"\n📋 Performance Breakdown:")
            print(f"  🌟 Excellent (≥80%): {excellent_episodes} episodes ({excellent_episodes/len(total_success_rates)*100:.1f}%)")
            print(f"  👍 Good (60-79%): {good_episodes} episodes ({good_episodes/len(total_success_rates)*100:.1f}%)")
            print(f"  👎 Poor (<60%): {poor_episodes} episodes ({poor_episodes/len(total_success_rates)*100:.1f}%)")
            
            # Efficiency analysis
            efficient_episodes = sum(1 for sensors in total_sensors_used if sensors <= 5)
            moderate_episodes = sum(1 for sensors in total_sensors_used if 5 < sensors <= 10)
            inefficient_episodes = sum(1 for sensors in total_sensors_used if sensors > 10)
            
            print(f"\n⚡ Efficiency Breakdown:")
            print(f"  🟢 Efficient (≤5 sensors): {efficient_episodes} episodes ({efficient_episodes/len(total_sensors_used)*100:.1f}%)")
            print(f"  🟡 Moderate (5-10 sensors): {moderate_episodes} episodes ({moderate_episodes/len(total_sensors_used)*100:.1f}%)")
            print(f"  🔴 Inefficient (>10 sensors): {inefficient_episodes} episodes ({inefficient_episodes/len(total_sensors_used)*100:.1f}%)")
            
            print("="*60 + "\n")
        else:
            print("\n⚠ No episodes completed successfully.")
        
        # Clean up
        renderer.close()
        
    finally:
        ray.shutdown()
        print("\nRay shutdown complete.")


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description='Visualize a trained Track-MDP policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize latest checkpoint
  python visualize.py --checkpoint ./agent_run194_ppo
  
  # Visualize specific checkpoint
  python visualize.py --checkpoint ./agent_run194_ppo/checkpoint_001000
  
  # Visualize with step-by-step control
  python visualize.py --checkpoint ./agent_run194_ppo --step
  
  # Visualize with slower animation
  python visualize.py --checkpoint ./agent_run194_ppo --fps 2
        """
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to checkpoint directory (will use latest checkpoint if directory contains multiple)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=10,
        help='Number of episodes to visualize (default: 10)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='Frames per second for rendering (default: 5)'
    )
    
    parser.add_argument(
        '--step', '-s',
        action='store_true',
        help='Wait for key press after each step (step-by-step mode)'
    )
    
    parser.add_argument(
        '--grid-size',
        type=int,
        default=None,
        help='Override grid size from config'
    )
    
    parser.add_argument(
        '--time-limit',
        type=int,
        default=None,
        help='Override time limit from config'
    )
    
    args = parser.parse_args()
    
    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    
    # If path is a directory, find latest checkpoint
    if os.path.isdir(checkpoint_path):
        # Check if this is already a specific checkpoint
        if os.path.exists(os.path.join(checkpoint_path, 'algorithm_state.pkl')):
            # This is already a checkpoint directory
            pass
        else:
            # Find latest checkpoint in directory
            latest = find_latest_checkpoint(checkpoint_path)
            if latest is None:
                print("Error: No valid checkpoints found.")
                return
            checkpoint_path = latest
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        return
    
    # Run visualization
    try:
        visualize_policy(
            checkpoint_path=checkpoint_path,
            num_episodes=args.episodes,
            fps=args.fps,
            wait_for_input=args.step,
            grid_size=args.grid_size,
            time_limit=args.time_limit
        )
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
