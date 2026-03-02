#!/usr/bin/env python3
"""
Setup script for Track-MDP: Object Tracking with Reinforcement Learning

This package provides a modular framework for training reinforcement learning agents
to track moving objects in grid environments using sensor networks.
"""

import os
import sys
from setuptools import setup, find_packages

# Ensure we're in the right directory
here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get version from src/__init__.py
def get_version():
    version_file = os.path.join(here, 'src', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '1.0.0'

# Read requirements from requirements.txt
def get_requirements():
    requirements_file = os.path.join(here, 'requirements.txt')
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and platform-specific requirements
            if line and not line.startswith('#') and not line.startswith('-'):
                # Remove platform-specific markers for now
                if ';' in line:
                    line = line.split(';')[0].strip()
                requirements.append(line)
    return requirements

# Optional dependencies
extras_require = {
    'dev': [
        'pytest>=7.0.0',
        'black>=22.0.0',
        'flake8>=5.0.0',
        'mypy>=0.991',
        'pre-commit>=2.20.0',
    ],
    'docs': [
        'sphinx>=5.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'jupyter': [
        'jupyter>=1.0.0',
        'ipywidgets>=7.6.0',
    ],
    'gpu': [
        'torch-audio>=0.12.0',
        'torchvision>=0.13.0',
    ]
}

# Add 'all' extra that includes everything
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name='track-mdp',
    version=get_version(),
    description='Object Tracking with Reinforcement Learning using Sensor Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # Author and contact info
    author='Track-MDP Team',
    author_email='your-email@example.com',
    
    # URLs
    url='https://github.com/yourusername/Track-MDP-final',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/Track-MDP-final/issues',
        'Source': 'https://github.com/yourusername/Track-MDP-final',
        'Documentation': 'https://github.com/yourusername/Track-MDP-final/wiki',
    },
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'track_mdp': [
            'assets/*.png',
        ],
    },
    
    # Requirements
    python_requires='>=3.8',
    install_requires=get_requirements(),
    extras_require=extras_require,
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'track-mdp-train=src.training.trainer:main',
            'track-mdp-visualize=visualize:main',
            'track-mdp-evaluate=comparative_evaluation:main',
        ],
    },
    
    # Classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords
    keywords='reinforcement-learning, object-tracking, sensor-networks, ray-rllib, a2c, pomdp',
    
    # Additional metadata
    zip_safe=False,
    platforms=['any'],
    
    # Testing
    test_suite='tests',
    tests_require=['pytest>=7.0.0'],
)


