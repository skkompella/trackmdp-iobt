# Track-MDP: Reinforcement Learning for Target Tracking with Controlled Sensing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ray RLlib](https://img.shields.io/badge/Ray-RLlib-orange.svg)](https://docs.ray.io/en/latest/rllib/index.html)

Implementation of the paper: "Track-MDP: Reinforcement Learning for Target Tracking with Controlled Sensing" (**ICASSP 2025**)

Adarsh M. Subramaniam, **Argyrios Gerogiannis**, James Z. Hare, Venugopal Veeravalli 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)

Paper link: [https://ieeexplore.ieee.org/document/10890122](https://ieeexplore.ieee.org/document/10890122)

## 🌟 Features

- **Grid-based Object Tracking**: Stochastic object movement in configurable NxN grid environments
- **Sensor Network Optimization**: Intelligent sensor activation to minimize energy consumption while maximizing tracking accuracy
- **Ray RLlib Integration**: Built on Ray RLlib for scalable distributed training
- **Real-time Visualization**: Interactive pygame-based visualization of tracking performance
- **Comparative Evaluation**: Built-in QMDP baseline comparison for performance analysis
- **Modular Architecture**: Clean separation of environment, training, evaluation, and visualization components
- **Flexible Configuration**: Easy parameter tuning for different scenarios and environments


## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Install from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Track-MDP-final.git
   cd Track-MDP-final
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv track_mdp_env
   source track_mdp_env/bin/activate  # On Windows: track_mdp_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

## 🏃 Quick Start

### Training

Train a tracking agent with default parameters:

```python
from src.training.trainer import train

# Train wihout visualization
train(visualization_mode='none')
```

### Visualization and Evaluation

Visualize and evaluate a trained policy:

```bash
python visualize.py --checkpoint ./agent_run194_a2c --episodes 100 --fps 10
```

## 📁 Project Structure

```
Track-MDP-final/
├── src/
│   ├── core/                    # Core environment and gym wrapper
│   │   ├── environment.py       # Grid environment implementation
│   │   └── gym_wrapper.py       # Gymnasium environment wrapper
│   ├── training/                # Training modules
│   │   ├── trainer.py           # Main training script
│   │   └── monitor.py           # Training monitoring and visualization
│   ├── evaluation/              # Evaluation and testing
│   │   ├── evaluator.py         # Policy evaluation functions
│   │   └── visualizer.py        # Interactive visualization
│   ├── visualization/           # Rendering and display
│   │   └── renderer.py          # Pygame-based visualization
│   └── utils/                   # Utility functions
├── assets/                      # Image assets for visualization
│   ├── robot4.png               # Robot/object sprite
│   ├── antenna_on.png           # Active sensor sprite
│   └── antenna_off.png          # Inactive sensor sprite
├── examples/                    # Usage examples
├── visualize.py                 # Standalone visualization script
├── comparative_evaluation.py    # QMDP vs Track-MDP comparison
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```


## 📊 Performance Evaluation

### Metrics

The framework provides comprehensive performance metrics:

- **Tracking Accuracy**: Proportion of time steps where object is successfully detected
- **Sensor Efficiency**: Average number of sensors activated per time step
- **Average Reward**: Average cumulative reward including tracking success and sensor costs
- **Cost per Detection**: Average sensor cost per successful object detection

### Baseline Comparison

Use the built-in QMDP baseline for performance comparison:

```bash
python comparative_evaluation.py
```

This will output a detailed comparison showing:
- Tracking accuracy improvements
- Sensor efficiency gains  
- Reward optimization



## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `python -m pytest`
5. **Submit a pull request**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Track-MDP-final.git
cd Track-MDP-final

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting  
- **MyPy** for type checking
- **Pytest** for testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Ray RLlib](https://docs.ray.io/en/latest/rllib/) for distributed reinforcement learning
- Visualization powered by [Pygame](https://www.pygame.org/)


## 🔗 Related Projects

- [Ray RLlib](https://github.com/ray-project/ray)
- [OpenAI Gym](https://github.com/openai/gym)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

---

**Happy Tracking!** 🎯
