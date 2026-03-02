"""
Training Monitor for Track-MDP

This module provides a training monitor that can add visualization and 
enhanced monitoring to existing training scripts.
"""

import time
import threading
from queue import Queue, Empty
import numpy as np

try:
    from src.visualization.renderer import TrackingRenderer
    from src.evaluation.visualizer import visualize_policy_interactive
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class TrainingMonitor:
    """
    A monitor that can add visualization and enhanced logging to training.
    
    Can be used to:
    - Visualize policy during training at specified intervals
    - Log detailed training statistics
    - Provide real-time feedback
    - Allow interactive control during training
    """
    
    def __init__(self, environment, visualization_config=None):
        """
        Initialize the training monitor.
        
        Args:
            environment: The grid environment instance
            visualization_config (dict): Configuration for visualization
                - enabled (bool): Enable visualization
                - frequency (int): Visualize every N iterations
                - episodes (int): Episodes per visualization
                - fps (int): Frames per second
                - step_by_step (bool): Step-by-step mode
        """
        self.environment = environment
        self.viz_config = visualization_config or {}
        
        # Training statistics
        self.iteration_count = 0
        self.rewards_history = []
        self.evaluation_history = []
        
        # Control flags
        self.should_stop = False
        self.pause_training = False
        
        # Visualization state
        self.viz_enabled = self.viz_config.get('enabled', False) and VISUALIZATION_AVAILABLE
        self.viz_frequency = self.viz_config.get('frequency', 10)
        self.viz_episodes = self.viz_config.get('episodes', 3)
        self.viz_fps = self.viz_config.get('fps', 10)
        self.viz_step_by_step = self.viz_config.get('step_by_step', False)
        
        if self.viz_enabled:
            print(f"✓ Training Monitor initialized with visualization")
            print(f"  Frequency: Every {self.viz_frequency} iterations")
            print(f"  Episodes: {self.viz_episodes}")
            print(f"  FPS: {self.viz_fps}")
            print(f"  Step-by-step: {self.viz_step_by_step}")
        else:
            print("✓ Training Monitor initialized (no visualization)")
    
    def on_iteration_end(self, algo, result):
        """
        Called at the end of each training iteration.
        
        Args:
            algo: The RLlib algorithm instance
            result: Training result dictionary
            
        Returns:
            bool: True to continue training, False to stop
        """
        self.iteration_count += 1
        iteration = result['training_iteration']
        reward_mean = result['episode_reward_mean']
        
        self.rewards_history.append(reward_mean)
        
        # Print progress
        print(f"Iteration {iteration}: reward_mean = {reward_mean:.4f}")
        
        # Check if we should visualize
        if self.viz_enabled and (iteration % self.viz_frequency == 0):
            success = self._run_visualization(algo, iteration)
            if not success:
                print("Stopping training due to visualization interruption")
                return False
        
        # Check for stop condition
        if self.should_stop:
            print("Stopping training due to monitor stop signal")
            return False
        
        # Handle pause
        while self.pause_training and not self.should_stop:
            time.sleep(0.1)
        
        return True
    
    def _run_visualization(self, algo, iteration):
        """Run visualization and handle interruptions."""
        print(f"\n{'='*50}")
        print(f"VISUALIZATION AT ITERATION {iteration}")
        print(f"{'='*50}")
        
        try:
            viz_summary = visualize_policy_interactive(
                algo=algo,
                environment=self.environment,
                num_episodes=self.viz_episodes,
                fps=self.viz_fps,
                wait_for_input=self.viz_step_by_step,
                verbose=True
            )
            
            if viz_summary and viz_summary.get('interrupted', False):
                return False
            
            print(f"{'='*50}\n")
            return True
            
        except KeyboardInterrupt:
            print("\nVisualization interrupted")
            return False
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Continuing training...")
            return True
    
    def stop_training(self):
        """Signal to stop training."""
        self.should_stop = True
    
    def pause_training(self):
        """Pause training."""
        self.pause_training = True
    
    def resume_training(self):
        """Resume training."""
        self.pause_training = False
    
    def get_statistics(self):
        """Get training statistics."""
        if not self.rewards_history:
            return {}
        
        recent_rewards = self.rewards_history[-10:]  # Last 10 iterations
        
        return {
            'total_iterations': self.iteration_count,
            'current_reward_mean': self.rewards_history[-1],
            'recent_reward_mean': np.mean(recent_rewards),
            'best_reward': max(self.rewards_history),
            'reward_trend': np.mean(recent_rewards) - np.mean(self.rewards_history[:10]) if len(self.rewards_history) > 10 else 0
        }


def integrate_monitor_with_training():
    """
    Example of how to integrate the monitor with your existing training script.
    """
    
    # This would go in your training script
    example_code = '''
    # In your training function:
    
    # Create monitor
    viz_config = {
        'enabled': True,
        'frequency': 10,
        'episodes': 3,
        'fps': 10,
        'step_by_step': False
    }
    monitor = TrainingMonitor(qobj.grid_env, viz_config)
    
    # Training loop
    for i in range(no_of_episodes):
        result = algo.train()
        
        # Let monitor handle visualization and logging
        should_continue = monitor.on_iteration_end(algo, result)
        if not should_continue:
            break
        
        # Your existing evaluation code...
        if (i+1) % 50 == 0:
            # ... evaluation logic ...
            pass
    
    # Get final statistics
    stats = monitor.get_statistics()
    print(f"Training completed: {stats}")
    '''
    
    print("Integration example:")
    print(example_code)


# Convenience function for quick visualization during training
def quick_visualize_during_training(algo, environment, episodes=3, fps=10):
    """
    Quick visualization function that can be called during training.
    
    Args:
        algo: RLlib algorithm instance
        environment: Grid environment
        episodes (int): Number of episodes to visualize
        fps (int): Frames per second
        
    Returns:
        bool: True if completed normally, False if interrupted
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available")
        return True
    
    try:
        viz_summary = visualize_policy_interactive(
            algo=algo,
            environment=environment,
            num_episodes=episodes,
            fps=fps,
            wait_for_input=False,
            verbose=False
        )
        
        return not viz_summary.get('interrupted', False)
        
    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"Visualization error: {e}")
        return True