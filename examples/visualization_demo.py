#!/usr/bin/env python3
"""
Example 4: Visualization Demo

This example demonstrates the visualization capabilities of Track-MDP,
including interactive visualization, real-time rendering, and performance
monitoring displays.

Usage:
    python examples/visualization_demo.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.environment import learning_grid_sarsa_0
from src.evaluation.visualizer import visualize_policy_interactive, find_latest_checkpoint
from src.visualization.renderer import TrackingRenderer
import ray
from ray.rllib.algorithms.ppo import PPO
import numpy as np


def check_visualization_requirements():
    """
    Check if visualization requirements are available.
    """
    print("Checking Visualization Requirements...")
    print("-" * 40)
    
    requirements = {
        'pygame': False,
        'ray': False,
        'numpy': False
    }
    
    # Check pygame
    try:
        import pygame
        requirements['pygame'] = True
        print("✅ pygame: Available")
    except ImportError:
        print("❌ pygame: Not available")
        print("   Install with: pip install pygame")
    
    # Check ray
    try:
        import ray
        requirements['ray'] = True
        print("✅ ray: Available")
    except ImportError:
        print("❌ ray: Not available")
        print("   Install with: pip install 'ray[rllib]'")
    
    # Check numpy
    try:
        import numpy
        requirements['numpy'] = True
        print("✅ numpy: Available")
    except ImportError:
        print("❌ numpy: Not available")
        print("   Install with: pip install numpy")
    
    all_available = all(requirements.values())
    
    if all_available:
        print("\n🎉 All visualization requirements are available!")
    else:
        print("\n⚠️ Some requirements are missing. Install them to use visualization.")
    
    return all_available


def basic_renderer_demo():
    """
    Demonstrate basic renderer functionality without a trained model.
    """
    print("\n" + "="*60)
    print("BASIC RENDERER DEMO")
    print("="*60)
    
    try:
        import pygame
    except ImportError:
        print("❌ pygame not available. Cannot run renderer demo.")
        return False
    
    print("This demo shows basic visualization without a trained model.")
    print("A simple pattern will be displayed to test the renderer.")
    print("\nControls:")
    print("  - ESC: Exit visualization")
    print("  - Close window: Stop visualization")
    
    input("\nPress Enter to start the demo...")
    
    try:
        # Create renderer
        renderer = TrackingRenderer(
            grid_size=8,
            sq_pixels=60,
            fps=2,
            title='Track-MDP Renderer Demo'
        )
        
        print("Renderer created. Starting animation...")
        
        # Simple animation demo
        for frame in range(20):
            # Create a simple moving pattern
            sensors = np.zeros((8, 8))
            
            # Moving diagonal line
            for i in range(8):
                x = (i + frame) % 8
                y = i
                sensors[y, x] = 1
            
            # Moving object position
            obj_x = (frame * 2) % 8
            obj_y = (frame) % 8
            object_pos = obj_y * 8 + obj_x
            
            # Update renderer
            renderer.update_sensors(sensors, object_pos)
            
            # Render frame
            if not renderer.render():
                print("Visualization stopped by user.")
                break
        
        renderer.close()
        print("✅ Basic renderer demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Renderer demo failed: {e}")
        return False


def interactive_visualization_demo():
    """
    Demonstrate interactive visualization with a trained model.
    """
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATION DEMO")
    print("="*60)
    
    # Check for trained model
    model_path = "./agent_run194_ppo"
    
    if not os.path.exists(model_path):
        print("❌ No trained model found at:", model_path)
        print("Run training first or specify a different model path.")
        return False
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(model_path)
    
    if latest_checkpoint is None:
        print("❌ No valid checkpoints found in model directory.")
        return False
    
    print(f"Found trained model: {latest_checkpoint}")
    
    try:
        # Create environment
        qobj = learning_grid_sarsa_0(194, 10, 4, [0.15, 0.3, 0.45, 0.6], 6, 6, 1, 1)
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Load trained model
        algo = PPO.from_checkpoint(latest_checkpoint)
        
        print("\nInteractive Visualization Options:")
        print("1. Real-time (smooth animation)")
        print("2. Step-by-step (manual advancement)")
        print("3. Fast (high speed)")
        print("4. Slow (detailed observation)")
        
        while True:
            choice = input("\nSelect visualization mode (1-4): ")
            if choice in ['1', '2', '3', '4']:
                break
            print("Invalid choice. Please enter 1-4.")
        
        # Configure based on choice
        configs = {
            '1': {'fps': 5, 'step_by_step': False, 'episodes': 3},
            '2': {'fps': 1, 'step_by_step': True, 'episodes': 2},
            '3': {'fps': 20, 'step_by_step': False, 'episodes': 5},
            '4': {'fps': 1, 'step_by_step': False, 'episodes': 2}
        }
        
        config = configs[choice]
        mode_names = {
            '1': 'Real-time',
            '2': 'Step-by-step', 
            '3': 'Fast',
            '4': 'Slow'
        }
        
        print(f"\nStarting {mode_names[choice]} visualization...")
        print(f"Episodes: {config['episodes']}")
        print(f"FPS: {config['fps']}")
        if config['step_by_step']:
            print("Press any key to advance each step")
        print("\nControls:")
        print("  - ESC: Exit visualization")
        print("  - Close window: Stop visualization")
        
        input("\nPress Enter to start...")
        
        # Run interactive visualization
        viz_summary = visualize_policy_interactive(
            algo=algo,
            environment=qobj.grid_env,
            num_episodes=config['episodes'],
            fps=config['fps'],
            wait_for_input=config['step_by_step'],
            verbose=True
        )
        
        if viz_summary:
            if viz_summary.get('interrupted', False):
                print("\n⏹ Visualization was interrupted by user.")
            else:
                print(f"\n✅ Interactive visualization completed!")
                print(f"Episodes shown: {viz_summary['episodes_completed']}")
                print(f"Mean success rate: {viz_summary['mean_success_rate']:.4f}")
                print(f"Mean sensor usage: {viz_summary['mean_sensors_used']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive visualization failed: {e}")
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def performance_visualization_demo():
    """
    Demonstrate visualization with performance monitoring overlay.
    """
    print("\n" + "="*60)
    print("PERFORMANCE VISUALIZATION DEMO")
    print("="*60)
    
    print("This demo shows visualization with real-time performance statistics.")
    print("Statistics will be printed to console during visualization.")
    
    # Check for model
    model_path = "./agent_run194_ppo"
    
    if not os.path.exists(model_path):
        print("❌ No trained model found. Run basic renderer demo instead.")
        return basic_renderer_demo()
    
    try:
        # Load environment and model (same as previous demo)
        qobj = learning_grid_sarsa_0(194, 10, 4, [0.15, 0.3, 0.45, 0.6], 6, 6, 1, 1)
        
        ray.init(ignore_reinit_error=True)
        
        latest_checkpoint = find_latest_checkpoint(model_path)
        algo = PPO.from_checkpoint(latest_checkpoint)
        
        print("Starting performance visualization...")
        print("Watch the console for real-time statistics!")
        
        input("Press Enter to start...")
        
        # Run with verbose output for performance monitoring
        viz_summary = visualize_policy_interactive(
            algo=algo,
            environment=qobj.grid_env,
            num_episodes=5,
            fps=8,
            wait_for_input=False,
            verbose=True  # This enables detailed console output
        )
        
        if viz_summary:
            print("\n📊 Performance Summary:")
            print("=" * 25)
            stats = viz_summary
            
            print(f"Episodes Completed: {stats['episodes_completed']}")
            print(f"Success Rate: {stats['mean_success_rate']:.4f} ± {stats['std_success_rate']:.4f}")
            print(f"Sensor Usage: {stats['mean_sensors_used']:.2f} ± {stats['std_sensors_used']:.2f}")
            print(f"Best Performance: {stats['best_success_rate']:.4f}")
            print(f"Most Efficient: {stats['best_efficiency']:.2f} sensors/step")
            
            # Performance assessment
            if stats['mean_success_rate'] > 0.8:
                print("🌟 Excellent tracking performance!")
            elif stats['mean_success_rate'] > 0.6:
                print("👍 Good tracking performance!")
            else:
                print("📈 Performance could be improved.")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance visualization failed: {e}")
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def standalone_visualization_demo():
    """
    Demonstrate standalone visualization script usage.
    """
    print("\n" + "="*60)
    print("STANDALONE VISUALIZATION DEMO")
    print("="*60)
    
    print("This demo shows how to use the standalone visualization script.")
    print("The script 'visualize.py' can be run from command line.")
    
    # Check if visualize.py exists
    visualize_script = os.path.join(project_root, 'visualize.py')
    
    if not os.path.exists(visualize_script):
        print("❌ visualize.py not found in project root.")
        return False
    
    print(f"✅ Found visualization script: {visualize_script}")
    
    print("\nCommand Line Usage Examples:")
    print("-" * 30)
    
    examples = [
        ("Basic visualization", "python visualize.py --checkpoint ./agent_run194_ppo"),
        ("5 episodes, 10 FPS", "python visualize.py --checkpoint ./agent_run194_ppo --episodes 5 --fps 10"),
        ("Step-by-step mode", "python visualize.py --checkpoint ./agent_run194_ppo --step"),
        ("Custom grid size", "python visualize.py --checkpoint ./agent_run194_ppo --grid-size 8"),
        ("Specific checkpoint", "python visualize.py --checkpoint ./agent_run194_ppo/checkpoint_001000"),
    ]
    
    for i, (description, command) in enumerate(examples, 1):
        print(f"{i}. {description}:")
        print(f"   {command}")
        print()
    
    # Check for trained model
    model_path = "./agent_run194_ppo"
    
    if os.path.exists(model_path):
        print("✅ Trained model found. You can run these commands directly!")
        
        response = input("\nWould you like to run the basic visualization now? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            print("\nRunning: python visualize.py --checkpoint ./agent_run194_ppo --episodes 3")
            
            try:
                # Import and run the main function from visualize.py
                sys.argv = ['visualize.py', '--checkpoint', './agent_run194_ppo', '--episodes', '3']
                
                # Import the visualize module
                visualize_module_path = visualize_script
                spec = __import__('importlib.util').util.spec_from_file_location("visualize", visualize_module_path)
                visualize_module = __import__('importlib.util').util.module_from_spec(spec)
                spec.loader.exec_module(visualize_module)
                
                # Run the main function
                visualize_module.main()
                
                print("✅ Standalone visualization completed!")
                
            except Exception as e:
                print(f"❌ Standalone visualization failed: {e}")
                print("Try running the command manually from the terminal.")
                return False
    else:
        print("⚠️ No trained model found. Train a model first to test visualization.")
    
    return True


def main():
    """
    Main function to run visualization demos.
    """
    print("TRACK-MDP VISUALIZATION DEMOS")
    print("=============================\n")
    
    # Check requirements first
    if not check_visualization_requirements():
        print("\nCannot run visualization demos without required packages.")
        print("Please install the missing requirements and try again.")
        return
    
    demos = {
        '1': ('Basic Renderer Demo', basic_renderer_demo),
        '2': ('Interactive Visualization', interactive_visualization_demo),
        '3': ('Performance Monitoring', performance_visualization_demo),
        '4': ('Standalone Script Demo', standalone_visualization_demo),
        '5': ('All Demos', None)
    }
    
    print("\nAvailable Visualization Demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    
    while True:
        choice = input(f"\nSelect demo to run (1-{len(demos)}): ")
        if choice in demos:
            break
        print(f"Invalid choice. Please enter 1-{len(demos)}.")
    
    if choice == '5':
        # Run all demos
        print("\nRunning all visualization demos...\n")
        
        for i, (name, func) in enumerate(demos.values()):
            if func is None:  # Skip the "All Demos" entry
                continue
            
            print(f"\n{'='*50}")
            print(f"DEMO {i+1}: {name.upper()}")
            print(f"{'='*50}")
            
            success = func()
            
            if not success:
                print(f"\n❌ Demo {i+1} failed or was skipped.")
            
            if i < len(demos) - 2:  # Don't pause after the last demo
                input("\nPress Enter to continue to next demo...")
        
        print("\n🎉 All visualization demos completed!")
    
    else:
        # Run selected demo
        name, func = demos[choice]
        print(f"\nRunning: {name}\n")
        success = func()
        
        if success:
            print(f"\n✅ {name} completed successfully!")
        else:
            print(f"\n❌ {name} failed or was skipped.")
    
    print("\nVisualization Tips:")
    print("• Use step-by-step mode to analyze agent behavior in detail")
    print("• Adjust FPS based on your observation needs")
    print("• Monitor console output for real-time performance statistics")
    print("• Close the visualization window or press ESC to exit")
    
    print("\nFor more visualization options, see:")
    print("• src/visualization/renderer.py - Renderer implementation")
    print("• src/evaluation/visualizer.py - Visualization functions")
    print("• visualize.py - Standalone visualization script")


if __name__ == "__main__":
    main()