#!/bin/bash
# Install PyTorch and TorchVision with CUDA 13 support
# Run this AFTER: conda env create -f environment.yml && conda activate veisac-env

pip install \
    torch==2.10.0+cu130 \
    torchvision==0.25.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130
