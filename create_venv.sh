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