#!/usr/bin/env bash

#cd ~
#wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
#chmod +x Anaconda3-2019.07-Linux-x86_64.sh
#./Anaconda3-2019.07-Linux-x86_64.sh

conda create python=3.6 --name disetgl
conda activate disetgl
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

cd ~
git clone https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit.git
cd neurips2019_disentanglement_challenge_starter_kit
pip install -r requirements.txt

cd ~
git clone https://github.com/google-research/disentanglement_lib.git
cd disentanglement_lib
pip install .[tf_gpu]
dlib_tests
dlib_download_data
