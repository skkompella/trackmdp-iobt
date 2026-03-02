#!/usr/bin/env python3
"""
Example 2: Advanced Training with Visualization

This example demonstrates advanced training features including:
- Real-time visualization during training
- Custom configuration parameters
- Performance monitoring
- Detailed evaluation metrics

Usage:
    python examples/advanced_training.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.trainer import train
from src.evaluation.evaluator import evaluate_policy_detailed
from src.evaluation.visualizer import evaluate_and_visualize
from src.core.environment import learning_grid_sarsa_0
import ray
from ray.rllib.algorithms.ppo import PPO


def advanced_training_example():
    """
    Example of advanced training with monitoring and visualization.
    """
    print("="*70)
    print("TRACK-MDP ADVANCED TRAINING EXAMPLE")
    print("="*70)
    
    print("This example demonstrates:")
    print("• Training with real-time visualization")
    print("• Performance monitoring during training")
    print("• Detailed evaluation with statistics")
    print("• Custom parameter configuration")
    print()
    
    # Display current configuration (inline for this example)
    print("Current Configuration:")
    print("-" * 30)
    current_config = {
        'grid_size': 10,
        'time_limit': 1, 
        'training_iterations': 2000,
        'learning_rate': 0.0001,
        'num_rollout_workers': 4
    }
    for key, value in current_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Custom configuration for this example
    custom_config = {
        'episodes': 3,              # Visualization episodes
        'evaluation_episodes': 5,   # Evaluation episodes during training
        'fps': 15,                  # Higher FPS for smoother visualization
        'step_by_step': False       # Real-time visualization
    }
    
    print("Custom Visualization Config:")
    print("-" * 35)
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Ask user for visualization preference
    viz_modes = {
        '1': 'none',
        '2': 'evaluation', 
        '3': 'periodic'
    }
    
    print("Visualization Options:")
    print("  1. None - No visualization (fastest)")
    print("  2. Evaluation - Show visualization during evaluation cycles")
    print("  3. Periodic - Show visualization every few training iterations")
    
    while True:
        choice = input("\nSelect visualization mode (1-3): ")
        if choice in viz_modes:
            viz_mode = viz_modes[choice]
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\nSelected mode: {viz_mode}")
    
    if viz_mode == 'periodic':
        custom_config['frequency'] = 5  # Visualize every 5 iterations
        print("  - Will visualize every 5 training iterations")
    
    # Start training
    print(f"\nStarting training with {viz_mode} visualization...")
    print("=" * 50)
    
    try:
        train(
            visualization_mode=viz_mode,
            viz_config=custom_config
        )
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user.")
        return False
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check dependencies and system requirements.")
        return False
    
    return True


def detailed_evaluation_example():
    """
    Example of detailed evaluation with comprehensive metrics.
    """
    print("\n" + "="*70)
    print("DETAILED EVALUATION EXAMPLE")
    print("="*70)
    
    try:
        # Create environment
        run_number = 194
        qobj = learning_grid_sarsa_0(
            run_number, 10, 4, [0.1, 0.3, 0.6, 1.0], 6, 6, 1, 1
        )
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Load trained model
        model_path = f"./agent_run{run_number}_ppo"
        
        # Find latest checkpoint
        checkpoints = []
        if os.path.exists(model_path):
            for item in os.listdir(model_path):
                item_path = os.path.join(model_path, item)
                if os.path.isdir(item_path) and item.startswith('checkpoint_'):
                    try:
                        checkpoint_num = int(item.split('_')[1])
                        checkpoints.append((checkpoint_num, item_path))
                    except (ValueError, IndexError):
                        continue
        
        if not checkpoints:
            print("❌ No trained model found. Run training first.")
            return False
        
        # Use latest checkpoint
        checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = checkpoints[-1][1]
        print(f"Loading model from: {latest_checkpoint}")
        
        algo = PPO.from_checkpoint(latest_checkpoint)
        
        # Detailed evaluation
        print("\nRunning detailed evaluation...")
        print("This may take a moment...\n")
        
        detailed_summary = evaluate_policy_detailed(
            algo, qobj.grid_env, num_episodes=200
        )
        
        # Display comprehensive results
        print("="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        
        # Basic metrics
        print("Basic Performance Metrics:")
        print("-" * 30)
        print(f"  Mean Success Rate:    {detailed_summary['mean_success_rate']:.4f} ± {detailed_summary['std_success_rate']:.4f}")
        print(f"  Mean Episode Length:  {detailed_summary['mean_episode_length']:.2f}")
        print(f"  Mean Total Reward:    {detailed_summary['mean_total_reward']:.2f}")
        print(f"  Mean Sensor Usage:    {detailed_summary['mean_sensor_usage']:.2f}")
        
        # Performance analysis
        success_rates = detailed_summary['detailed_stats']['success_rates']
        sensor_usage = detailed_summary['detailed_stats']['sensor_usage']
        
        print(f"\nPerformance Distribution:")
        print("-" * 25)
        excellent = sum(1 for r in success_rates if r >= 0.8)
        good = sum(1 for r in success_rates if 0.6 <= r < 0.8)
        fair = sum(1 for r in success_rates if 0.4 <= r < 0.6)
        poor = sum(1 for r in success_rates if r < 0.4)
        total = len(success_rates)
        
        print(f"  Excellent (≥80%): {excellent:3d} episodes ({excellent/total*100:5.1f}%)")
        print(f"  Good (60-79%):    {good:3d} episodes ({good/total*100:5.1f}%)")
        print(f"  Fair (40-59%):    {fair:3d} episodes ({fair/total*100:5.1f}%)")
        print(f"  Poor (<40%):      {poor:3d} episodes ({poor/total*100:5.1f}%)")
        
        # Efficiency analysis
        print(f"\nEfficiency Analysis:")
        print("-" * 20)
        print(f"  Best Success Rate:    {max(success_rates):.4f}")
        print(f"  Worst Success Rate:   {min(success_rates):.4f}")
        print(f"  Most Efficient:       {min(sensor_usage):.2f} sensors/step")
        print(f"  Least Efficient:      {max(sensor_usage):.2f} sensors/step")
        
        # Calculate efficiency score
        efficiency_score = detailed_summary['mean_success_rate'] / detailed_summary['mean_sensor_usage']
        print(f"  Efficiency Score:     {efficiency_score:.4f} (success/sensor)")
        
        # Performance recommendations
        print(f"\nPerformance Assessment:")
        print("-" * 25)
        if detailed_summary['mean_success_rate'] > 0.8:
            print("  🌟 Outstanding! Excellent tracking performance.")
        elif detailed_summary['mean_success_rate'] > 0.7:
            print("  🎯 Very Good! Strong tracking with room for minor improvements.")
        elif detailed_summary['mean_success_rate'] > 0.6:
            print("  👍 Good! Solid performance with moderate efficiency.")
        else:
            print("  📈 Needs Improvement. Consider retraining or parameter tuning.")
        
        if detailed_summary['mean_sensor_usage'] < 3.0:
            print("  ⚡ Highly efficient sensor usage!")
        elif detailed_summary['mean_sensor_usage'] < 5.0:
            print("  🔋 Good sensor efficiency.")
        else:
            print("  ⚠️ High sensor usage - efficiency could be improved.")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def visualization_demo():
    """
    Demonstrate combined evaluation and visualization.
    """
    print("\n" + "="*70)
    print("INTERACTIVE VISUALIZATION DEMO")
    print("="*70)
    
    try:
        # Check for pygame availability
        try:
            import pygame
            pygame_available = True
        except ImportError:
            pygame_available = False
        
        if not pygame_available:
            print("❌ Pygame not available. Install pygame for visualization:")
            print("   pip install pygame")
            return False
        
        # Create environment
        run_number = 194
        qobj = learning_grid_sarsa_0(
            run_number, 10, 4, [0.1, 0.3, 0.6, 1.0], 6, 6, 1, 1
        )
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Load model (same as previous examples)
        model_path = f"./agent_run{run_number}_ppo"
        checkpoints = []
        
        if os.path.exists(model_path):
            for item in os.listdir(model_path):
                item_path = os.path.join(model_path, item)
                if os.path.isdir(item_path) and item.startswith('checkpoint_'):
                    try:
                        checkpoint_num = int(item.split('_')[1])
                        checkpoints.append((checkpoint_num, item_path))
                    except (ValueError, IndexError):
                        continue
        
        if not checkpoints:
            print("❌ No trained model found. Run training first.")
            return False
        
        checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = checkpoints[-1][1]
        
        algo = PPO.from_checkpoint(latest_checkpoint)
        
        # Combined evaluation and visualization
        print("Starting combined evaluation and visualization...")
        print("\nControls:")
        print("  - ESC: Exit visualization")
        print("  - Close window: Stop visualization")
        print()
        
        eval_summary, viz_summary = evaluate_and_visualize(
            algo=algo,
            environment=qobj.grid_env,
            num_episodes=5,          # Visualization episodes
            fps=10,                  # Frame rate
            evaluation_episodes=50,   # Background evaluation episodes
            show_visualization=True
        )
        
        if viz_summary and not viz_summary.get('interrupted', False):
            print("\n✅ Visualization completed successfully!")
            print(f"Visualized {viz_summary['episodes_completed']} episodes")
        else:
            print("\n⏹ Visualization was interrupted or skipped.")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization demo failed: {e}")
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main():
    """
    Main function to run advanced examples.
    """
    print("TRACK-MDP ADVANCED EXAMPLES")
    print("===========================\n")
    
    examples = {
        '1': ('Advanced Training', advanced_training_example),
        '2': ('Detailed Evaluation', detailed_evaluation_example),
        '3': ('Visualization Demo', visualization_demo),
        '4': ('All Examples', None)
    }
    
    print("Available Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    while True:
        choice = input(f"\nSelect example to run (1-{len(examples)}): ")
        if choice in examples:
            break
        print(f"Invalid choice. Please enter 1-{len(examples)}.")
    
    if choice == '4':
        # Run all examples
        print("\nRunning all examples in sequence...\n")
        
        for i, (name, func) in enumerate(examples.values()):
            if func is None:  # Skip the "All Examples" entry
                continue
                
            print(f"\n{'='*50}")
            print(f"RUNNING EXAMPLE {i+1}: {name.upper()}")
            print(f"{'='*50}")
            
            success = func()
            
            if not success:
                print(f"\n❌ Example {i+1} failed. Stopping sequence.")
                break
            
            if i < len(examples) - 2:  # Don't pause after the last example
                input("\nPress Enter to continue to next example...")
        
        print("\n🎉 All examples completed!")
    
    else:
        # Run selected example
        name, func = examples[choice]
        print(f"\nRunning: {name}\n")
        success = func()
        
        if success:
            print(f"\n✅ {name} completed successfully!")
        else:
            print(f"\n❌ {name} failed.")
    
    print("\nFor more examples and documentation, visit:")
    print("https://github.com/yourusername/Track-MDP-final")


if __name__ == "__main__":
    main()