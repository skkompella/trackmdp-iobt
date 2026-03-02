# Track-MDP Examples

This directory contains comprehensive examples demonstrating different aspects of the Track-MDP framework.

## Quick Start

Run the basic example to get started:

```bash
python examples/basic_training.py
```

## Available Examples

### 1. `basic_training.py` - Getting Started
- **Purpose**: Introduction to Track-MDP workflow
- **Features**: Basic training and evaluation
- **Time**: ~5-10 minutes
- **Requirements**: Ray RLlib, basic dependencies

**What it demonstrates:**
- Training a simple tracking agent
- Loading and evaluating trained models
- Basic performance metrics
- Environment configuration

### 2. `advanced_training.py` - Advanced Features
- **Purpose**: Advanced training with monitoring
- **Features**: Real-time visualization, detailed evaluation
- **Time**: ~10-15 minutes
- **Requirements**: pygame (for visualization)

**What it demonstrates:**
- Training with real-time visualization
- Performance monitoring during training
- Comprehensive evaluation metrics
- Custom configuration options

### 3. `custom_environment.py` - Environment Customization
- **Purpose**: Creating custom tracking environments
- **Features**: Different grid sizes, reward structures
- **Time**: ~5-10 minutes
- **Requirements**: Basic dependencies

**What it demonstrates:**
- Small vs large grid environments
- High mobility scenarios
- Custom reward structures
- Environment parameter effects

### 4. `visualization_demo.py` - Visualization Features
- **Purpose**: Interactive visualization capabilities
- **Features**: Real-time rendering, performance overlays
- **Time**: ~10-15 minutes
- **Requirements**: pygame

**What it demonstrates:**
- Interactive policy visualization
- Real-time performance monitoring
- Different visualization modes
- Standalone script usage

## Running Examples

### Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

For visualization examples, ensure pygame is installed:
```bash
pip install pygame
```

### Individual Examples

Run any example directly:
```bash
# Basic training and evaluation
python examples/basic_training.py

# Advanced training with visualization
python examples/advanced_training.py

# Custom environment configurations
python examples/custom_environment.py

# Visualization demonstrations
python examples/visualization_demo.py
```

### All Examples

Some examples offer an "All Examples" mode to run multiple demonstrations in sequence.

## Example Outputs

### Training Output
```
TRACK-MDP BASIC TRAINING EXAMPLE
============================================
Step 1: Training the agent...
------------------------------
Episode reward mean: 0 -2.1234
Episode reward mean: 1 -1.8765
...
✓ Training completed successfully!

Step 2: Evaluating the trained agent...
----------------------------------------
Running evaluation (100 episodes)...
  Average Tracking Success Rate: 0.7543 (75.43%)
  Average Sensors Used per Step: 4.21

EVALUATION RESULTS
==================================================
Tracking Accuracy: 0.7543 (75.43%)
Average Sensors per Step: 4.21
Efficiency Score: 0.1792
==================================================
🎉 Excellent performance! The agent learned to track effectively.
```

### Visualization Output
```
INTERACTIVE VISUALIZATION DEMO
============================================
Found trained model: ./agent_run194_a2c/checkpoint_001000

Starting Real-time visualization...
Episodes: 3
FPS: 5

Controls:
  - ESC: Exit visualization
  - Close window: Stop visualization

========================================
Episode 1/3
Object starting at position: 42
========================================
  Step 10: ✓ DETECTED | Sensors: 3 | Accuracy: 0.800
  Step 20: ✗ missed | Sensors: 2 | Accuracy: 0.750
...
```

## Understanding the Examples

### Training Flow
1. **Environment Creation**: Configure grid size, transitions, rewards
2. **Algorithm Setup**: Initialize A2C with custom parameters
3. **Training Loop**: Iterative policy improvement
4. **Evaluation**: Performance assessment on test episodes
5. **Visualization**: Real-time policy demonstration

### Key Concepts Demonstrated

#### Environment Configuration
- Grid sizes: 5×5 (fast), 10×10 (standard), 15×15 (complex)
- Transition probabilities: Object movement patterns
- Sensor constraints: Maximum sensors per time step
- Reward structures: Balancing accuracy vs efficiency

#### Training Parameters
- Learning rates: 0.0001 (stable) to 0.001 (fast)
- Rollout workers: 0 (single) to 4 (parallel)
- Training iterations: 10 (demo) to 2000 (full)

#### Evaluation Metrics
- **Tracking Accuracy**: Fraction of successful detections
- **Sensor Efficiency**: Average sensors used per step
- **Episode Length**: Steps until object reaches terminal state
- **Efficiency Score**: Accuracy divided by sensor usage

### Performance Expectations

#### Small Grid (5×5)
- Training time: 1-2 minutes
- Expected accuracy: 80-95%
- Sensor usage: 2-4 per step

#### Standard Grid (10×10)  
- Training time: 5-10 minutes
- Expected accuracy: 70-85%
- Sensor usage: 3-6 per step

#### Large Grid (15×15)
- Training time: 15-30 minutes
- Expected accuracy: 60-75% 
- Sensor usage: 4-8 per step

## Troubleshooting

### Common Issues

#### "Ray not available"
```bash
pip install 'ray[rllib]>=2.0.0'
```

#### "pygame not available"
```bash
pip install pygame>=2.1.0
```

#### "No trained model found"
- Run `basic_training.py` first to create a model
- Check that `./agent_run194_a2c/` directory exists

#### Visualization window doesn't appear
- Ensure you have a display available (not running headless)
- Try reducing FPS or grid size
- Check pygame installation

#### Training is slow
- Reduce grid size (N=5 instead of N=10)
- Decrease number of rollout workers
- Use fewer training iterations for testing

### Platform-Specific Notes

#### Windows
- PowerShell recommended for running examples
- May need Visual C++ redistributable for Ray

#### macOS
- Might need to allow Python through firewall for Ray
- Use `python3` instead of `python` if needed

#### Linux
- Usually works out of the box
- May need `python3-dev` package for some dependencies

## Next Steps

After running the examples:

1. **Modify Parameters**: Edit inline parameters in the scripts to test different scenarios
2. **Custom Environments**: Create your own environment configurations using inline parameters
3. **Algorithm Tuning**: Experiment with different hyperparameters
4. **Performance Analysis**: Use `comparative_evaluation.py` for detailed comparisons
5. **Integration**: Integrate Track-MDP components into your own projects

## Additional Resources

- **Main Documentation**: `../README.md`
- **API Reference**: `../src/` directory
- **Configuration**: Inline parameters in example scripts
- **Comparative Analysis**: `../comparative_evaluation.py`
- **Standalone Visualization**: `../visualize.py`

For questions or issues, please check the main repository documentation or create an issue on GitHub.