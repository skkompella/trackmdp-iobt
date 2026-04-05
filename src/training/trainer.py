"""
Training Script for Track-MDP

This script trains a PPO agent to track a moving object in a grid environment
using Ray RLlib. Matches the original training approach from coding_grid_v15_md_run194.py
"""

import os
import sys
import random
import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.environment import learning_grid_sarsa_0
from src.core.gym_wrapper import grid_environment
from src.evaluation.evaluator import evaluate_policy

# Import visualization (with fallback if not available)
try:
    from src.evaluation.visualizer import visualize_policy_interactive
    from src.visualization.renderer import TrackingRenderer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

SEED = 1
rnd = random
rnd.seed(SEED)
np.random.seed(SEED)


def save_environment(env, file_path):
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_name = os.path.join(file_path, f'env_s{SEED}.pt')
    data = {'environment': env}
    # Use weights_only=False explicitly to avoid future PyTorch compatibility issues
    torch.save(data, file_name)


def _run_visualization(algo, environment, iteration, viz_config, context=""):
    """
    Run visualization and handle interruptions.
    
    Args:
        algo: RLlib algorithm instance
        environment: Grid environment instance  
        iteration: Current training iteration
        viz_config: Visualization configuration
        context: Context string for display
        
    Returns:
        bool: True to continue training, False to stop
    """
    if not VISUALIZATION_AVAILABLE:
        return True
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION AT ITERATION {iteration} ({context})")
    print(f"{'='*60}")
    
    try:
        viz_summary = visualize_policy_interactive(
            algo=algo,
            environment=environment,
            num_episodes=viz_config['episodes'],
            fps=viz_config['fps'],
            wait_for_input=viz_config['step_by_step'],
            verbose=True
        )
        
        if viz_summary and viz_summary.get('interrupted', False):
            print("Training stopped by user during visualization")
            return False
        
        print(f"{'='*60}\n")
        return True
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted - stopping training")
        return False
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing training...")
        return True


def train(visualization_mode='none', viz_config=None):
    """
    Main training function with integrated visualization options.
    
    Args:
        visualization_mode (str): Visualization mode
            - 'none': No visualization
            - 'evaluation': Only visualize during evaluation cycles  
            - 'all': Continuous real-time visualization during training
        viz_config (dict): Visualization configuration
            - episodes (int): Number of episodes for evaluation visualization
            - fps (int): Frames per second for continuous visualization
            - step_by_step (bool): Step-by-step mode for evaluation
            - evaluation_episodes (int): Episodes for evaluation visualization
    """
    
    # Default visualization config
    default_viz_config = {
        'episodes': 3,
        'fps': 10,
        'step_by_step': False,
        'evaluation_episodes': 5
    }
    viz_config = {**default_viz_config, **(viz_config or {})}
    
    # Check if visualization is available
    if visualization_mode != 'none' and not VISUALIZATION_AVAILABLE:
        print("⚠ Warning: Visualization requested but not available. Install pygame.")
        print("⚠ Continuing training without visualization...")
        visualization_mode = 'none'
    
    # Visualization modes: none, evaluation, all
    if visualization_mode == 'all':
        print("Warning: 'all' mode (continuous during training) is not practical and would slow training significantly.")
        print("Using 'evaluation' mode instead - visualization will show during evaluation episodes.")
        visualization_mode = 'evaluation'
    
    run_number = 194  # 99999 for testing
    run_number_load = 194
    
    print(f"\n Run number = {run_number}")
    print(f" Visualization mode = {visualization_mode}")
    if visualization_mode == 'all':
        print(f" Continuous visualization FPS = {viz_config['fps']}")
    elif visualization_mode == 'evaluation':
        print(f" Evaluation visualization episodes = {viz_config['evaluation_episodes']}")
    print()
    N, num_trans = 10, 4
    terminal_st_prob = 0.005

    state_prob_run = 0.15
    state_trans_cum_prob = [i*(1-terminal_st_prob-state_prob_run)/float(num_trans-1) for i in range(1, num_trans)]
    state_trans_cum_prob += [state_trans_cum_prob[-1] + state_prob_run] 

    max_sensors, max_sensors_null = 6, 6
    no_of_episodes = 10 + 1  # 2*10**5 + 1
    time_limit_start = 1 
    time_limit_max = 1
    frequency_save = 10

    max_run_episodes_tschedule = 6000
    time_limit_schedule = [2000]
    
    qobj = learning_grid_sarsa_0(run_number, N, num_trans, state_trans_cum_prob, max_sensors, max_sensors_null, time_limit_start, time_limit_max)
    qobj.no_of_episodes = no_of_episodes
    
    file_path = "./agent_run{}_ppo".format(run_number)
    save_environment(qobj, file_path)
    env_config = {"qobj": qobj, "time_limit_schedule": time_limit_schedule,
                  "time_limit_max": time_limit_max}

    # Initialize Ray
    # Ensure workers can find the src module
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": project_root}})

    config = PPOConfig()
    config = config.environment(grid_environment, env_config=env_config)
    config = config.training(lr=0.0001, grad_clip=30.0)  
    config = config.resources(num_gpus=0)  

    #### Change no of rollout workers
    config = config.env_runners(num_env_runners=4) 
    config = config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)

    
    algo = config.build()  
    
    mean_ppo = []
    tracking_accuracies = []
    avg_sensors_per_step = []
    evaluation_iterations = []
    
    # [0,4000,8000,12000]
    run_episodes = [0, 2005, 8000, 12000]
    for time_lim in range(1, time_limit_max+1):

        for i in range(run_episodes[time_lim] - run_episodes[time_lim-1]):    
            result = algo.train()
            iteration = result['training_iteration']
            
            # Handle metric location change in newer Ray versions
            if 'episode_reward_mean' in result:
                reward_mean = result['episode_reward_mean']
            elif 'env_runners' in result and 'episode_reward_mean' in result['env_runners']:
                reward_mean = result['env_runners']['episode_reward_mean']
            else:
                reward_mean = 0.0
            
            print("episode reward mean:", i, reward_mean)
            mean_ppo.append(reward_mean)
            
            if (i+1) % 10 == 0:
                save_dir = "./agent_run{}_ppo".format(run_number)
                algo.save(save_dir)
                
                # Evaluate the policy
                print(f"\n{'='*60}")
                print(f"EVALUATION AT ITERATION {result['training_iteration']}")
                print(f"{'='*60}")
                
                # Run evaluation on current policy
                if visualization_mode == 'evaluation':
                    # First run standard evaluation
                    tracking_accuracy, avg_sensors = evaluate_policy(
                        algo, 
                        qobj.grid_env, 
                        num_episodes=100,
                        verbose=True
                    )
                    # Then show visualization
                    print("Starting evaluation visualization...")
                    try:
                        from src.evaluation.visualizer import visualize_policy_interactive
                        visualize_policy_interactive(
                            algo=algo,
                            environment=qobj.grid_env,
                            num_episodes=viz_config.get('evaluation_episodes', 3),
                            fps=viz_config.get('fps', 10),
                            wait_for_input=viz_config.get('step_by_step', False),
                            verbose=True
                        )
                    except Exception as e:
                        print(f"Visualization error: {e}")
                        print("Continuing with training...")
                else:
                    tracking_accuracy, avg_sensors = evaluate_policy(
                        algo, 
                        qobj.grid_env, 
                        num_episodes=100,  # Use 100 episodes for faster evaluation
                        verbose=True
                    )
                
                # Store evaluation metrics
                tracking_accuracies.append(tracking_accuracy)
                avg_sensors_per_step.append(avg_sensors)
                evaluation_iterations.append(result['training_iteration'])
                
                print(f"Tracking Accuracy: {tracking_accuracy:.4f} ({tracking_accuracy*100:.2f}%)")
                print(f"Average Sensors per Step: {avg_sensors:.2f}")
                print(f"{'='*60}\n")
                
                # Save metrics including evaluation results
                mean_save = {
                    "mean_ppo_array": mean_ppo,
                    "tracking_accuracies": tracking_accuracies,
                    "avg_sensors_per_step": avg_sensors_per_step,
                    "evaluation_iterations": evaluation_iterations
                }
                torch.save(mean_save, save_dir + "/{}.pt".format(result["training_iteration"]))
                
        print("\n updating time_limit = ", time_lim+1, "\n")
        qobj.update_time_limit(time_lim+1)        
        env_config = {"qobj": qobj, "time_limit_schedule": time_limit_schedule,
                     "time_limit_max": time_limit_max}

        config = config.environment(grid_environment, env_config=env_config)
        algo = config.build()

        algo.restore('./agent_run{}_ppo/checkpoint_{:06d}'.format(run_number_load, run_episodes[time_lim]))
        try:
            saved_data = torch.load('./agent_run{}_ppo/{}.pt'.format(run_number_load, run_episodes[time_lim]), weights_only=False)
        except Exception as e:
            print(f"⚠ Warning: Could not load metrics with custom classes: {e}")
            try:
                saved_data = torch.load('./agent_run{}_ppo/{}.pt'.format(run_number_load, run_episodes[time_lim]), weights_only=True)
            except Exception as e2:
                print(f"⚠ Warning: Could not load metrics: {e2}")
                saved_data = {"mean_ppo_array": []}
        
        mean_ppo = saved_data.get("mean_ppo_array", [])
        if "tracking_accuracies" in saved_data:
            tracking_accuracies = saved_data["tracking_accuracies"]
            avg_sensors_per_step = saved_data["avg_sensors_per_step"]
            evaluation_iterations = saved_data["evaluation_iterations"]
    
    # Final evaluation summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - FINAL EVALUATION SUMMARY")
    print("="*60)
    
    if tracking_accuracies:
        print(f"Training iterations completed: {len(mean_ppo)}")
        print(f"Evaluations performed: {len(tracking_accuracies)}")
        print(f"Best tracking accuracy: {max(tracking_accuracies):.4f} ({max(tracking_accuracies)*100:.2f}%)")
        print(f"Final tracking accuracy: {tracking_accuracies[-1]:.4f} ({tracking_accuracies[-1]*100:.2f}%)")
        print(f"Best avg sensors per step: {min(avg_sensors_per_step):.2f}")
        print(f"Final avg sensors per step: {avg_sensors_per_step[-1]:.2f}")
        
        # Save final summary
        save_dir = "./agent_run{}_ppo".format(run_number)
        final_summary = {
            "final_mean_ppo_array": mean_ppo,
            "final_tracking_accuracies": tracking_accuracies,
            "final_avg_sensors_per_step": avg_sensors_per_step,
            "final_evaluation_iterations": evaluation_iterations,
            "best_tracking_accuracy": max(tracking_accuracies),
            "final_tracking_accuracy": tracking_accuracies[-1],
            "best_avg_sensors": min(avg_sensors_per_step),
            "final_avg_sensors": avg_sensors_per_step[-1]
        }
        torch.save(final_summary, save_dir + "/final_summary.pt")
        print(f"Final summary saved to: {save_dir}/final_summary.pt")
    else:
        print("No evaluations were performed during training.")
    
    print("="*60)
    
    print("\n Done")    


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Track-MDP with optional visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualization modes:
  none        - No visualization (default)
  evaluation  - Visualize only during evaluation cycles  
  periodic    - Visualize every N training iterations
  all         - Visualize every training iteration

Examples:
  # Train without visualization
  python -m src.training.trainer
  
  # Train with visualization only during evaluations
  python -m src.training.trainer --viz evaluation
  
  # Train with visualization every 10 iterations
  python -m src.training.trainer --viz periodic --freq 10
  
  # Train with visualization every iteration (step-by-step)
  python -m src.training.trainer --viz all --step
        """
    )
    
    parser.add_argument(
        '--viz', '--visualization',
        choices=['none', 'evaluation', 'periodic', 'all'],
        default='none',
        help='Visualization mode (default: none)'
    )
    
    parser.add_argument(
        '--freq', '--frequency',
        type=int,
        default=10,
        help='For periodic mode: visualize every N iterations (default: 10)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Number of episodes to visualize each time (default: 3)'
    )
    
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=5,
        help='Number of episodes for evaluation visualization (default: 5)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for visualization (default: 10)'
    )
    
    parser.add_argument(
        '--step', '--step-by-step',
        action='store_true',
        help='Step-by-step visualization mode'
    )
    
    args = parser.parse_args()
    
    viz_config = {
        'frequency': args.freq,
        'episodes': args.episodes,
        'evaluation_episodes': args.eval_episodes,
        'fps': args.fps,
        'step_by_step': args.step
    }
    
    train(visualization_mode=args.viz, viz_config=viz_config)