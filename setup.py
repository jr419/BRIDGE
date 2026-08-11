"""
Setup script for the GraphRewiring library.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bridge",
    version="1.1.0",
    description="BRIDGE: Block Rewiring from Inference-Derived Graph Ensembles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jonathan Rubin, Sahil Loomba, Nick S. Jones",
    url="https://github.com/jr419/BRIDGE",
    packages=find_packages(),
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'bridge=bridge.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
