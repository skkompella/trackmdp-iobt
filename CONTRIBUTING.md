# Contributing to Track-MDP

We welcome contributions to the Track-MDP project! This document provides guidelines for contributing.

## Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/Track-MDP-final.git
   cd Track-MDP-final
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test them
5. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Track-MDP-final.git
cd Track-MDP-final

# Create virtual environment
python -m venv track_mdp_dev
source track_mdp_dev/bin/activate  # On Windows: track_mdp_dev\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install development tools
pip install pytest black flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- **Environment details**: OS, Python version, dependency versions
- **Minimal reproduction case**: Simplest code that reproduces the issue
- **Expected vs actual behavior**: What you expected vs what happened
- **Error messages**: Full stack trace if applicable

Use the bug report template in GitHub Issues.

### Feature Requests

For new features, please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** clearly
- **Propose an implementation** approach if possible
- **Consider backwards compatibility**

### Code Contributions

#### Areas for Contribution

1. **Core Environment**
   - New environment configurations
   - Performance optimizations
   - Additional reward structures

2. **Training Algorithms**
   - Support for new RL algorithms (PPO, SAC, etc.)
   - Hyperparameter optimization
   - Distributed training improvements

3. **Evaluation & Visualization**
   - New evaluation metrics
   - Enhanced visualization features
   - Performance analysis tools

4. **Documentation**
   - Code documentation improvements
   - Tutorial additions
   - Example enhancements

5. **Testing**
   - Unit test coverage
   - Integration tests
   - Performance benchmarks

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use isort for import sorting
- **Type hints**: Required for new functions
- **Docstrings**: Google style for all public functions

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Automatically run on commit
git commit -m "Your commit message"

# Run manually
pre-commit run --all-files
```

The hooks include:
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/):

- **Be respectful** and inclusive
- **Be patient** with new contributors
- **Be constructive** in feedback
- **Focus on the work**, not the person

### Mentorship

New contributors are welcome! If you're new to:

- **Open source**: Check our "good first issue" labels
- **Machine learning**: See our documentation and examples
- **Ray RLlib**: We can help with integration questions

Thank you for contributing to Track-MDP! 🎯
