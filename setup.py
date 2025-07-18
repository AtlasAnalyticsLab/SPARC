#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="sparc",
    version="0.1.0",
    author="Ali Nasiri-Sarvi",
    description="SPARC: Cross-Mode Interpretabilty",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "h5py>=3.1.0",
        "Pillow>=8.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pycocotools>=2.0.4",
        "wandb>=0.12.0",
        "opencv-python>=4.5.0",
        "open_clip_torch>=2.0.0",
        "timm>=0.9.0",
        "captum>=0.6.0",
        "grad-cam>=1.4.0",
        "ttach>=0.0.3",
    ],
)