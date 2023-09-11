#!/bin/bash

# create venv:
VENVNAME=gromvenv
virtualenv $VENVNAME

source $VENVNAME/bin/activate

# # requirements:
pip install numpy
pip install packaging
pip install dgl
pip install torch
pip install vtk
git install hydra

git clone https://github.com/NVIDIA/modulus.git
cd modulus 
pip install .
cd ..
rm -rf modulus

git clone https://github.com/NVIDIA/modulus-launch.git
cd modulus-launch 
pip install .
cd ..
rm -rf modulus-launch

pip install pydantic==1.10.9