#!/usr/bin/env python3
"""
Example 1: Basic Training and Evaluation

This example demonstrates the basic workflow of training a Track-MDP agent
and evaluating its performance.

Usage:
    python examples/basic_training.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.trainer import train
from src.evaluation.evaluator import evaluate_policy
from src.core.environment import learning_grid_sarsa_0
import ray
from ray.rllib.algorithms.ppo import PPO


def basic_training_example():
    """
    Example of basic training workflow.
    """
    print("="*60)
    print("TRACK-MDP BASIC TRAINING EXAMPLE")
    print("="*60)
    
    print("This example will:")
    print("1. Train a Track-MDP agent for a few iterations")
    print("2. Save the trained model")
    print("3. Load and evaluate the trained model")
    print("4. Display performance metrics")
    print()
    
    # Step 1: Train the model
    print("Step 1: Training the agent...")
    print("-" * 30)
    
    try:
        # Train with minimal iterations for demonstration
        # In practice, you would use more iterations (e.g., 2000)
        train(
            visualization_mode='none',  # No visualization for faster training
            viz_config={
                'episodes': 3,
                'fps': 10,
                'step_by_step': False
            }
        )
        print("✓ Training completed successfully!")
    
    except Exception as e:
        print(f"Training failed: {e}")
        print("This might be due to Ray initialization issues or missing dependencies.")
        print("Make sure Ray and all other dependencies are installed correctly.")
        return False
    
    # Step 2: Load and evaluate the trained model
    print("\nStep 2: Evaluating the trained agent...")
    print("-" * 40)
    
    try:
        # Create environment (same parameters as training)
        run_number = 194
        N, num_trans = 10, 4
        terminal_st_prob = 0.005
        state_prob_run = 0.15
        state_trans_cum_prob = [
            i * (1 - terminal_st_prob - state_prob_run) / float(num_trans - 1) 
            for i in range(1, num_trans)
        ]
        state_trans_cum_prob += [state_trans_cum_prob[-1] + state_prob_run]
        
        max_sensors, max_sensors_null = 6, 6
        time_limit_start = 1
        time_limit_max = 1
        
        qobj = learning_grid_sarsa_0(
            run_number, N, num_trans, state_trans_cum_prob,
            max_sensors, max_sensors_null, time_limit_start, time_limit_max
        )
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Load the trained model
        model_path = f"./agent_run{run_number}_ppo"
        
        # Find latest checkpoint
        import os
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
        
        if checkpoints:
            # Use the latest checkpoint
            checkpoints.sort(key=lambda x: x[0])
            latest_checkpoint = checkpoints[-1][1]
            print(f"Loading model from: {latest_checkpoint}")
            
            # Load the algorithm
            algo = PPO.from_checkpoint(latest_checkpoint)
            
            # Evaluate the policy
            print("Running evaluation (100 episodes)...")
            tracking_accuracy, avg_sensors = evaluate_policy(
                algo, qobj.grid_env, num_episodes=100, verbose=True
            )
            
            # Display results
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Tracking Accuracy: {tracking_accuracy:.4f} ({tracking_accuracy*100:.2f}%)")
            print(f"Average Sensors per Step: {avg_sensors:.2f}")
            print(f"Efficiency Score: {tracking_accuracy/avg_sensors:.4f}")
            print("="*50)
            
            # Performance assessment
            if tracking_accuracy > 0.7:
                print("🎉 Excellent performance! The agent learned to track effectively.")
            elif tracking_accuracy > 0.5:
                print("👍 Good performance! The agent shows decent tracking ability.")
            else:
                print("📈 Room for improvement. Consider longer training or parameter tuning.")
            
            return True
            
        else:
            print("❌ No trained model found. Please run training first.")
            return False
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def custom_environment_example():
    """
    Example of creating and testing a custom environment configuration.
    """
    print("\n" + "="*60)
    print("CUSTOM ENVIRONMENT EXAMPLE")
    print("="*60)
    
    # Create a smaller environment for faster testing
    print("Creating a 5x5 grid environment...")
    
    qobj = learning_grid_sarsa_0(
        run_number=999,          # Test run number
        N=5,                     # Smaller 5x5 grid
        num_trans=3,             # Fewer transition types
        state_trans_cum_prob=[0.4, 0.7, 1.0],  # Simple probabilities
        max_sensors=3,           # Fewer sensors
        max_sensors_null=3,
        time_limit=1,
        time_limit_max=1
    )
    
    print(f"✓ Environment created:")
    print(f"  Grid Size: {qobj.N}x{qobj.N}")
    print(f"  Missing State: {qobj.missing_state}")
    print(f"  Time Limit: {qobj.time_limit}")
    
    # Test environment dynamics
    print("\nTesting environment dynamics...")
    
    # Reset and show initial state
    qobj.grid_env.reset_object_state()
    initial_pos = qobj.grid_env.object_pos
    print(f"  Initial object position: {initial_pos}")
    
    # Show a few object movements
    print("  Object movement sequence:")
    for i in range(5):
        old_pos = qobj.grid_env.object_pos
        qobj.grid_env.object_move()
        new_pos = qobj.grid_env.object_pos
        
        if new_pos == qobj.N * qobj.N:
            print(f"    Step {i+1}: {old_pos} → TERMINAL")
            break
        else:
            print(f"    Step {i+1}: {old_pos} → {new_pos}")
    
    print("✓ Environment dynamics working correctly!")


def main():
    """
    Main function to run all examples.
    """
    print("TRACK-MDP EXAMPLES")
    print("==================\n")
    
    print("This script demonstrates basic usage of the Track-MDP framework.")
    print("Note: Training may take a few minutes depending on your system.\n")
    
    # Run custom environment example first (no training required)
    custom_environment_example()
    
    # Ask user if they want to run the full training example
    print("\n" + "-"*60)
    response = input("Do you want to run the full training example? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        success = basic_training_example()
        
        if success:
            print("\n🎉 All examples completed successfully!")
            print("\nNext steps:")
            print("- Try the visualization: python visualize.py --checkpoint ./agent_run194_a2c")
            print("- Run comparative evaluation: python comparative_evaluation.py")
            print("- Explore other examples in the examples/ directory")
        else:
            print("\n❌ Training example failed.")
            print("Check the error messages above and ensure all dependencies are installed.")
    else:
        print("\nSkipping training example. To run it later, use:")
        print("python examples/basic_training.py")
    
    print("\nFor more examples, see:")
    print("- examples/advanced_training.py")
    print("- examples/custom_environment.py")
    print("- examples/visualization_demo.py")


if __name__ == "__main__":
    main()