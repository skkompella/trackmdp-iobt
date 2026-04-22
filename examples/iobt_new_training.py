#!/usr/bin/env python3
"""
Example 1 (IoBT Adapted): Basic Training and Evaluation

This example demonstrates the basic workflow of training a Track-MDP agent
and evaluating its performance using the custom Camp Buckner IoBT environment.
"""




import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.training.trainer import train 
from src.evaluation.evaluator import evaluate_policy
# from src.core.iobt_new_env import learning_grid_sarsa_0
# from src.core.iobt_equal_env import learning_grid_sarsa_0
from src.core.iobt_6node_env import learning_grid_sarsa_0
from src.core.gym_wrapper import grid_environment
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO


def basic_training_example():
    """
    Example of basic training workflow.
    """
    print("="*60)
    print("TRACK-MDP IOBT TRAINING EXAMPLE")
    print("="*60)
    
    print("This example will:")
    print("1. Train a Track-MDP agent on the IoBT Camp Buckner environment")
    print("2. Save the trained model")
    print("3. Load and evaluate the trained model")
    print("4. Display performance metrics\n")
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
        # Initialize shared environment parameters for the 4x4 IoBT map
        run_number = 197  # Changed run number to avoid overwriting original 10x10 model
        N, num_trans = 4, 6
        terminal_st_prob = 0.005
        state_prob_run = 0.15
        state_trans_cum_prob = [round((i+1)/num_trans, 4) for i in range(num_trans)]
        max_sensors, max_sensors_null = 6, 6
        time_limit_start, time_limit_max = 1, 1






        
        qobj = learning_grid_sarsa_0(
            run_number, N, num_trans, state_trans_cum_prob,
            max_sensors, max_sensors_null, time_limit_start, time_limit_max
        )

        # Initialize Ray
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": project_root}})
        

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
    Example of testing the custom IoBT dynamics.
    """
    print("\n" + "="*60)
    print("CUSTOM ENVIRONMENT EXAMPLE: CAMP BUCKNER")
    print("="*60)
    
    print("Creating the Camp Buckner 10-node IoBT environment...")
    
    qobj = learning_grid_sarsa_0(
        run_number=999,
        N=4,
        num_trans=6,
        state_trans_cum_prob=[round((i+1)/6, 4) for i in range(6)],
        max_sensors=6,
        max_sensors_null=6,
        time_limit=1,
        time_limit_max=1
    )
    
    # # Inject new environment
    # qobj.grid_env = iobt_env(6, 6, qobj.missing_state, 1)
    
    print(f"✓ Environment created:")
    print(f"  Underlying Grid Matrix: {qobj.N}x{qobj.N}")
    print(f"  Missing State: {qobj.missing_state}")
    
    print("\nTesting IoBT environment dynamics...")
    qobj.grid_env.reset_object_state()
    initial_pos = qobj.grid_env.object_pos
    print(f"  Initial object position: Node {initial_pos}")
    
    print("  Object movement sequence:")
    for i in range(5):
        old_pos = qobj.grid_env.object_pos
        qobj.grid_env.object_move()
        new_pos = qobj.grid_env.object_pos
        
        if new_pos == qobj.N * qobj.N:
            print(f"    Step {i+1}: Node {old_pos} → TERMINAL")
            break
        else:
            print(f"    Step {i+1}: Node {old_pos} → Node {new_pos}")
    
    print("✓ Environment dynamics working correctly!")


def main():
    print("TRACK-MDP IOBT EXAMPLES")
    print("=======================\n")
    
    custom_environment_example()
    
    print("\n" + "-"*60)
    response = input("Do you want to run the full training example on the IoBT map? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        success = basic_training_example()
        if success:
            print("\n🎉 All examples completed successfully!")
        else:
            print("\n❌ Training example failed.")
    else:
        print("\nSkipping training example.")


if __name__ == "__main__":
    main()