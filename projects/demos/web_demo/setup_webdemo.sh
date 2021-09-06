#!/bin/bash

# For grcuda-data repo
git submodule init
cd ../../../grcuda-data
git submodule update --remote

cd -

# Create symbolic link for the images
cd frontend
ln -s ../../../../grcuda-data/datasets/web_demo/images images
cd -

# install cmake (required for opencv4nodejs)
sudo apt-get install cmake libopencv-dev python3

# Compile cuda binary
echo "Compiling CUDA binary"
mkdir ../image_pipeline/cuda/build
cd ../image_pipeline/cuda/build
cmake ..
make

cd -

# Build backend 
echo "Building and running backend"
cd backend 
npm i
npm run build
npm run runall &

# Run frontend
python -m http.server 8085 --directory ../frontend



