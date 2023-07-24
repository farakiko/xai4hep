from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="xai4hep",
    packages=find_packages(),
    version="1.0.0",
    description="XAI toolbox for interpreting state-of-the-art ML algorithms for high energy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Farouk Mokhtar",
    author_email="farouk.mokhtar@gmail.com",
    url="https://github.com/farakiko/xai4hep",
    license="BSD-3-Clause",
    install_requires=[
        "torch>=1.8",
        "numpy>=1.21",
        "torch_geometric",
        "torch-cluster",
        "pandas",
        "matplotlib",
        "mplhep",
        "captum",
        "tqdm",
        "fastjet",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA :: 11.3",
        "Environment :: GPU :: NVIDIA CUDA :: 11.6",
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Typing :: Typed",
    ],
)
