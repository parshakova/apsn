#!/usr/bin/env bash

conda create -n py35 pip python=3.5
source activate py35

# pytorch
conda install pytorch torchvision cuda90 -c pytorch

# dependencies
pip install -U spacy==1.10.1 && python -m spacy.en.download
conda install -c anaconda cython numpy  pandas scikit-learn
conda install msgpack
conda install -c conda-forge matplotlib
pip install tensorboardX pynvrtc cupy-cuda90


# prepare dataset
python prepro.py
python semisup_labels.py

