#!/usr/bin/env python3
"""
Setup script for FNSFT package.

This script provides an alternative installation method for environments
that don't support pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="fnsft",
    version="0.1.0",
    description="Fine-tuning for Neural Supervised Fine-Tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Byrne",
    author_email="realdanielbyrne@icloud.com",
    url="https://github.com/realdanielbyrne/fnsft",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fnsft-train=fnsft.sft_trainer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning, deep-learning, transformers, fine-tuning, llm, lora, quantization",
    project_urls={
        "Bug Reports": "https://github.com/realdanielbyrne/fnsft/issues",
        "Source": "https://github.com/realdanielbyrne/fnsft",
        "Documentation": "https://github.com/realdanielbyrne/fnsft/blob/main/docs/README.md",
    },
)
