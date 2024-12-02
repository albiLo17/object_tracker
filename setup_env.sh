#!/bin/bash

# Exit on any error
set -e

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Torch Geometric core library
echo "Installing Torch Geometric..."
pip install torch-geometric

# Install Torch Geometric dependencies for PyTorch 2.2.0 and CUDA 12.1
echo "Installing Torch Geometric dependencies..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Install additional requirements from requirements.txt
echo "Installing additional dependencies..."
pip install -r requirements.txt

echo "Environment setup complete!"