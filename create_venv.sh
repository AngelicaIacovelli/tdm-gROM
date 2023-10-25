#!/bin/bash

module purge
module load gpu/0.17.3b gcc/10.2.0/i62tgso 
module load py-virtualenv/16.7.6/4kzv6ad

# create venv:
VENVNAME=gromvenv
virtualenv $VENVNAME

source $VENVNAME/bin/activate

# # requirements:
pip install numpy
pip install packaging
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
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
pip install ray
pip install optuna